# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of the Policy Gradient algorithm."""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn
from rich.progress import track
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.adapter import OnPolicyAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils import distributed


@registry.register
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods,line-too-long
class PolicyGradient(BaseAlgo):
    """The Policy Gradient algorithm.

    References:
        - Title: Policy Gradient Methods for Reinforcement Learning with Function Approximation
        - Authors: Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour.
        - URL: `PG <https://proceedings.neurips.cc/paper/1999/file64d828b85b0bed98e80ade0a5c43b0f-Paper.pdf>`_
    """

    def _init_env(self) -> None:
        """Initialize the environment.

        OmniSafe uses :class:`omnisafe.adapter.OnPolicyAdapter` to adapt the environment to the
        algorithm.

        User can customize the environment by inheriting this method.

        Examples:
            >>> def _init_env(self) -> None:
            ...     self._env = CustomAdapter()

        Raises:
            AssertionError: If the number of steps per epoch is not divisible by the number of
                environments.
        """
        self._env: OnPolicyAdapter = OnPolicyAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, 'The number of steps per epoch is not divisible by the number of environments.'
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch
            // distributed.world_size()
            // self._cfgs.train_cfgs.vector_env_nums
        )

    def _init_model(self) -> None:
        """Initialize the model.

        OmniSafe uses :class:`omnisafe.models.actor_critic.constraint_actor_critic.ConstraintActorCritic`
        as the default model.

        User can customize the model by inheriting this method.

        Examples:
            >>> def _init_model(self) -> None:
            ...     self._actor_critic = CustomActorCritic()
        """
        self._actor_critic: ConstraintActorCritic = ConstraintActorCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._cfgs.train_cfgs.epochs,
        ).to(self._device)

        if distributed.world_size() > 1:
            distributed.sync_params(self._actor_critic)

        if self._cfgs.model_cfgs.exploration_noise_anneal:
            self._actor_critic.set_annealing(
                epochs=[0, self._cfgs.train_cfgs.epochs],
                std=self._cfgs.model_cfgs.std_range,
            )

    def _init(self) -> None:
        """The initialization of the algorithm.

        User can define the initialization of the algorithm by inheriting this method.

        Examples:
            >>> def _init(self) -> None:
            ...     super()._init()
            ...     self._buffer = CustomBuffer()
            ...     self._model = CustomModel()
        """
        self._buf: VectorOnPolicyBuffer = VectorOnPolicyBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=self._steps_per_epoch,
            gamma=self._cfgs.algo_cfgs.gamma,
            lam=self._cfgs.algo_cfgs.lam,
            lam_c=self._cfgs.algo_cfgs.lam_c,
            advantage_estimator=self._cfgs.algo_cfgs.adv_estimation_method,
            standardized_adv_r=self._cfgs.algo_cfgs.standardized_rew_adv,
            standardized_adv_c=self._cfgs.algo_cfgs.standardized_cost_adv,
            penalty_coefficient=self._cfgs.algo_cfgs.penalty_coef,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._device,
        )

    def _init_log(self) -> None:
        """Log info about epoch.

        +-----------------------+----------------------------------------------------------------------+
        | Things to log         | Description                                                          |
        +=======================+======================================================================+
        | Train/Epoch           | Current epoch.                                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpCost        | Average cost of the epoch.                                           |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpRet         | Average return of the epoch.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpLen         | Average length of the epoch.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Values/reward         | Average value in :meth:`rollout` (from critic network) of the epoch. |
        +-----------------------+----------------------------------------------------------------------+
        | Values/cost           | Average cost in :meth:`rollout` (from critic network) of the epoch.  |
        +-----------------------+----------------------------------------------------------------------+
        | Values/Adv            | Average reward advantage of the epoch.                               |
        +-----------------------+----------------------------------------------------------------------+
        | Loss/Loss_pi          | Loss of the policy network.                                          |
        +-----------------------+----------------------------------------------------------------------+
        | Loss/Loss_cost_critic | Loss of the cost critic network.                                     |
        +-----------------------+----------------------------------------------------------------------+
        | Train/Entropy         | Entropy of the policy network.                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Train/StopIters       | Number of iterations of the policy network.                          |
        +-----------------------+----------------------------------------------------------------------+
        | Train/PolicyRatio     | Ratio of the policy network.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Train/LR              | Learning rate of the policy network.                                 |
        +-----------------------+----------------------------------------------------------------------+
        | Misc/Seed             | Seed of the experiment.                                              |
        +-----------------------+----------------------------------------------------------------------+
        | Misc/TotalEnvSteps    | Total steps of the experiment.                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Time                  | Total time.                                                          |
        +-----------------------+----------------------------------------------------------------------+
        | FPS                   | Frames per second of the epoch.                                      |
        +-----------------------+----------------------------------------------------------------------+
        """
        self._logger = Logger(
            output_dir=self._cfgs.logger_cfgs.log_dir,
            exp_name=self._cfgs.exp_name,
            seed=self._cfgs.seed,
            use_tensorboard=self._cfgs.logger_cfgs.use_tensorboard,
            use_wandb=self._cfgs.logger_cfgs.use_wandb,
            config=self._cfgs,
        )

        what_to_save: dict[str, Any] = {}
        what_to_save['pi'] = self._actor_critic.actor
        if self._cfgs.algo_cfgs.obs_normalize:
            obs_normalizer = self._env.save()['obs_normalizer']
            what_to_save['obs_normalizer'] = obs_normalizer
        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()

        self._logger.register_key(
            'Metrics/EpRet',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )
        self._logger.register_key(
            'Metrics/EpCost',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )
        self._logger.register_key(
            'Metrics/EpLen',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )

        self._logger.register_key('Train/Epoch')
        self._logger.register_key('Train/Entropy')
        self._logger.register_key('Train/KL')
        self._logger.register_key('Train/StopIter')
        self._logger.register_key('Train/PolicyRatio', min_and_max=True)
        self._logger.register_key('Train/LR')
        if self._cfgs.model_cfgs.actor_type == 'gaussian_learning':
            self._logger.register_key('Train/PolicyStd')

        self._logger.register_key('TotalEnvSteps')

        # log information about actor
        self._logger.register_key('Loss/Loss_pi', delta=True)
        self._logger.register_key('Value/Adv')

        # log information about critic
        self._logger.register_key('Loss/Loss_reward_critic', delta=True)
        self._logger.register_key('Value/reward')

        if self._cfgs.algo_cfgs.use_cost:
            # log information about cost critic
            self._logger.register_key('Loss/Loss_cost_critic', delta=True)
            self._logger.register_key('Value/cost')

        self._logger.register_key('Time/Total')
        self._logger.register_key('Time/Rollout')
        self._logger.register_key('Time/Update')
        self._logger.register_key('Time/Epoch')
        self._logger.register_key('Time/FPS')

        # register environment specific keys
        for env_spec_key in self._env.env_spec_keys:
            self.logger.register_key(env_spec_key)

    def learn(self) -> tuple[float, float, float]:
        """This is main function for algorithm update.

        It is divided into the following steps:

        - :meth:`rollout`: collect interactive data from environment.
        - :meth:`update`: perform actor/critic updates.
        - :meth:`log`: epoch/update information for visualization and terminal log print.

        Returns:
            ep_ret: Average episode return in final epoch.
            ep_cost: Average episode cost in final epoch.
            ep_len: Average episode length in final epoch.
        """
        start_time = time.time()
        self._logger.log('INFO: Start training')
        
        # self._logger.log('INFO: running sanity check')
        # sanity_passed = self._sanity_check()
        # if not sanity_passed:
        #     self._logger.log('ERROR: Sanity check failed, aborting training.')
        #     return 0.0, 0.0, 0.0
        # self._logger.log('INFO: Sanity check passed, proceeding with training.')

        for epoch in range(self._cfgs.train_cfgs.epochs):
            epoch_time = time.time()

            rollout_time = time.time()
            self._env.rollout(
                steps_per_epoch=self._steps_per_epoch,
                agent=self._actor_critic,
                buffer=self._buf,
                logger=self._logger,
            )
            self._logger.store({'Time/Rollout': time.time() - rollout_time})

            update_time = time.time()
            self._update()
            self._logger.store({'Time/Update': time.time() - update_time})

            if self._cfgs.model_cfgs.exploration_noise_anneal:
                self._actor_critic.annealing(epoch)

            if self._cfgs.model_cfgs.actor.lr is not None:
                self._actor_critic.actor_scheduler.step()

            self._logger.store(
                {
                    'TotalEnvSteps': (epoch + 1) * self._cfgs.algo_cfgs.steps_per_epoch,
                    'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                    'Train/LR': (
                        0.0
                        if self._cfgs.model_cfgs.actor.lr is None
                        else self._actor_critic.actor_scheduler.get_last_lr()[0]
                    ),
                },
            )

            self._logger.dump_tabular()

            # save model to disk
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0 or (
                epoch + 1
            ) == self._cfgs.train_cfgs.epochs:
                self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()
        self._env.close()

        return ep_ret, ep_cost, ep_len

    def _update(self) -> None:
        """Update actor, critic.

        -  Get the ``data`` from buffer

        .. hint::

            +----------------+------------------------------------------------------------------+
            | obs            | ``observation`` sampled from buffer.                             |
            +================+==================================================================+
            | act            | ``action`` sampled from buffer.                                  |
            +----------------+------------------------------------------------------------------+
            | target_value_r | ``target reward value`` sampled from buffer.                     |
            +----------------+------------------------------------------------------------------+
            | target_value_c | ``target cost value`` sampled from buffer.                       |
            +----------------+------------------------------------------------------------------+
            | logp           | ``log probability`` sampled from buffer.                         |
            +----------------+------------------------------------------------------------------+
            | adv_r          | ``estimated advantage`` (e.g. **GAE**) sampled from buffer.      |
            +----------------+------------------------------------------------------------------+
            | adv_c          | ``estimated cost advantage`` (e.g. **GAE**) sampled from buffer. |
            +----------------+------------------------------------------------------------------+


        -  Update value net by :meth:`_update_reward_critic`.
        -  Update cost net by :meth:`_update_cost_critic`.
        -  Update policy net by :meth:`_update_actor`.

        The basic process of each update is as follows:

        #. Get the data from buffer.
        #. Shuffle the data and split it into mini-batch data.
        #. Get the loss of network.
        #. Update the network by loss.
        #. Repeat steps 2, 3 until the number of mini-batch data is used up.
        #. Repeat steps 2, 3, 4 until the KL divergence violates the limit.
        """
        data = self._buf.get()
        print(f"\n=== Buffer Data Check ===")
        print(f"Target_value_r: mean={data['target_value_r'].mean():.3f}, range=[{data['target_value_r'].min():.3f}, {data['target_value_r'].max():.3f}]")
        print(f"Adv_r: mean={data['adv_r'].mean():.3f}, std={data['adv_r'].std():.3f}")
        print(f"Obs shape: {data['obs'].shape}, Act shape: {data['act'].shape}")
        obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
        )
        rewards = data['reward'] if 'reward' in data else None
        if rewards is not None:
            print(f"Reward分布: mean={rewards.mean():.4f}, std={rewards.std():.4f}, nonzero={(rewards!=0).sum()}/{rewards.numel()}")

        original_obs = obs
        old_distribution = self._actor_critic.actor(obs)

        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, logp, target_value_r, target_value_c, adv_r, adv_c),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        update_counts = 0
        final_kl = 0.0

        for i in track(range(self._cfgs.algo_cfgs.update_iters), description='Updating...'):
            for (
                obs,
                act,
                logp,
                target_value_r,
                target_value_c,
                adv_r,
                adv_c,
            ) in dataloader:
                self._update_reward_critic(obs, target_value_r)
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic(obs, target_value_c)
                self._update_actor(obs, act, logp, adv_r, adv_c)

            new_distribution = self._actor_critic.actor(original_obs)

            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
            )
            kl = distributed.dist_avg(kl)

            final_kl = kl.item()
            update_counts += 1

            if self._cfgs.algo_cfgs.kl_early_stop and kl.item() > self._cfgs.algo_cfgs.target_kl:
                self._logger.log(f'Early stopping at iter {i + 1} due to reaching max kl')
                break

        self._logger.store(
            {
                'Train/StopIter': update_counts,  # pylint: disable=undefined-loop-variable
                'Value/Adv': adv_r.mean().item(),
                'Train/KL': final_kl,
            },
        )

    def _update_reward_critic(self, obs: torch.Tensor, target_value_r: torch.Tensor) -> None:
        r"""Update value network under a double for loop.

        The loss function is ``MSE loss``, which is defined in ``torch.nn.MSELoss``.
        Specifically, the loss function is defined as:

        .. math::

            L = \frac{1}{N} \sum_{i=1}^N (\hat{V} - V)^2

        where :math:`\hat{V}` is the predicted cost and :math:`V` is the target cost.

        #. Compute the loss function.
        #. Add the ``critic norm`` to the loss function if ``use_critic_norm`` is ``True``.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            target_value_r (torch.Tensor): The ``target_value_r`` sampled from buffer.
        """
        device = next(self._actor_critic.reward_critic.parameters()).device
        if obs.device != device:
            obs = obs.to(device)
        if target_value_r.device != device:
            target_value_r = target_value_r.to(device)
        # if not hasattr(self, '_printed_obs'):
        #     self._printed_obs = True
        #     print(f"\n=== Actual Training Obs ===")
        #     print(f"Obs[0]: {obs[0]}")
        #     print(f"Obs stats: mean={obs.mean():.4f}, std={obs.std():.4f}, range=[{obs.min():.4f}, {obs.max():.4f}]")
        self._actor_critic.reward_critic_optimizer.zero_grad()
        loss = nn.functional.mse_loss(self._actor_critic.reward_critic(obs)[0], target_value_r)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        loss.backward()
        model_param_ids = [id(p) for p in self._actor_critic.reward_critic.parameters()]
        opt_param_ids = []
        for group in self._actor_critic.reward_critic_optimizer.param_groups:
            opt_param_ids += [id(p) for p in group['params']]

        if not hasattr(self, '_checked_critic_grad'):
            self._checked_critic_grad = True
            total_norm = 0
            for param in self._actor_critic.reward_critic.parameters():
                if param.grad is not None:
                    total_norm += param.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            # print(f"Critic grad norm: {total_norm:.4f}")
        # values = self._actor_critic.reward_critic(obs)[0]
        # print(f"Raw values: {values.detach()}")
        # print(f"Value range: [{values.std():.3f}, {values.min():.3f}, {values.max():.3f}, {values.mean():.3f}]")
        # print(f"target_values mean/std: {target_value_r.mean():.3f}/{target_value_r.std():.3f}, range: [{target_value_r.min():.3f}, {target_value_r.max():.3f}]")
        # print(f"Critic loss: {loss.item():.4f}, target_value range: [{target_value_r.min():.3f}, {target_value_r.max():.3f}]")
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.reward_critic)
        # first = next(self._actor_critic.reward_critic.parameters()).clone().detach()
        # state = self._actor_critic.reward_critic_optimizer.state
        # first_param = next(self._actor_critic.reward_critic.parameters())
        # sid = None
        # for k in state.keys():
        #     # keys are parameter objects
        #     if k.shape == first_param.shape:
        #         sid = k
        #         break
        # print(">> CHECK: optimizer state entries:", len(state))
        # if sid is not None:
        #     st = state[sid]
        #     print("  state for a param: has exp_avg?", 'exp_avg' in st, 'exp_avg_sq' in st, 'step' in st)
        #     if 'exp_avg' in st:
        #         print("   exp_avg mean", st['exp_avg'].mean().item(), "exp_avg_sq mean", st['exp_avg_sq'].mean().item(), "step", st.get('step', None))
        # else:
        #     print("  could not find matching state by shape; printing few keys' shapes:")
        #     for k in list(state.keys())[:6]:
        #         print("   key shape", getattr(k,'shape', None))

        self._actor_critic.reward_critic_optimizer.step()
        # second = next(self._actor_critic.reward_critic.parameters()).clone().detach()
        # print("param diff:", (first - second).abs().mean().item())
        self._logger.store({'Loss/Loss_reward_critic': loss.mean().item()})

    def _update_cost_critic(self, obs: torch.Tensor, target_value_c: torch.Tensor) -> None:
        r"""Update value network under a double for loop.

        The loss function is ``MSE loss``, which is defined in ``torch.nn.MSELoss``.
        Specifically, the loss function is defined as:

        .. math::

            L = \frac{1}{N} \sum_{i=1}^N (\hat{V} - V)^2

        where :math:`\hat{V}` is the predicted cost and :math:`V` is the target cost.

        #. Compute the loss function.
        #. Add the ``critic norm`` to the loss function if ``use_critic_norm`` is ``True``.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            target_value_c (torch.Tensor): The ``target_value_c`` sampled from buffer.
        """
        device = next(self._actor_critic.reward_critic.parameters()).device
        if obs.device != device:
            obs = obs.to(device)
        if target_value_c.device != device:
            target_value_c = target_value_c.to(device)
        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss = nn.functional.mse_loss(self._actor_critic.cost_critic(obs)[0], target_value_c)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.cost_critic)
        self._actor_critic.cost_critic_optimizer.step()

        self._logger.store({'Loss/Loss_cost_critic': loss.mean().item()})

    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        """Update policy network under a double for loop.

        #. Compute the loss function.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        .. warning::
            For some ``KL divergence`` based algorithms (e.g. TRPO, CPO, etc.),
            the ``KL divergence`` between the old policy and the new policy is calculated.
            And the ``KL divergence`` is used to determine whether the update is successful.
            If the ``KL divergence`` is too large, the update will be terminated.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log_p`` sampled from buffer.
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.
        """
        device = next(self._actor_critic.actor.parameters()).device
        if obs.device != device: obs = obs.to(device)
        if act.device != device: act = act.to(device)
        if logp.device != device: logp = logp.to(device)
        if adv_r.device != device: adv_r = adv_r.to(device)
        if adv_c.device != device: adv_c = adv_c.to(device)
        adv = self._compute_adv_surrogate(adv_r, adv_c)
        loss = self._loss_pi(obs, act, logp, adv)
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.actor)
        self._actor_critic.actor_optimizer.step()
        # DEBUG12.7
        if hasattr(self, '_diagnose_counter'):
            self._diagnose_counter += 1
        else:
            self._diagnose_counter = 0
        
        if self._diagnose_counter % 500 == 0:  # 每500次更新
            self._diagnose_training(obs, act, adv_r)

    def _diagnose_training(self, obs, act, adv_r):
        """添加此方法到PolicyGradient类"""
        with torch.no_grad():
            dist = self._actor_critic.actor(obs)
            logits = dist.logits if hasattr(dist, 'logits') else None
            
            if logits is not None:
                print(f"\n=== 诊断 (Step {self._diagnose_counter}) ===")
                print(f"Logits std: {logits.std():.4f}")
                print(f"Entropy: {dist.entropy().mean():.4f}")
                
                # 检查value
                values = self._actor_critic.reward_critic(obs)[0]
                print(f"Raw values: {values.detach()}")
                print(f"Value range: [{values.min():.3f}, {values.max():.3f}, {values.mean():.3f}]")
                print(f"Value std: {values.std():.4f}")
                print(f"Adv std: {adv_r.std():.4f}")
                print(f"Adv_r: mean={adv_r.mean():.3f}, std={adv_r.std():.3f}, min={adv_r.min():.3f}, max={adv_r.max():.3f}")

    def _sanity_check(self) -> bool:
        """单样本过拟合测试"""
        # 获取一个固定样本
        device = next(self._actor_critic.actor.parameters()).device
        obs, _ = self._env.reset()
        obs_sample = obs[0]  # 取第一个env
        print(f"obs shape: {obs_sample.shape}, obs sample: {obs_sample}")
        obs_tensor = torch.tensor(obs_sample, dtype=torch.float32, device=device).unsqueeze(0)
        action_tensor = torch.tensor([0], dtype=torch.long).to(device)
        target_tensor = torch.tensor([15.0], dtype=torch.float32).to(device)
        
        print(f"\n=== Sanity Check ===")
        print(f"Device: {device}, Obs shape: {obs_tensor.shape}")
        print(f"Obs shape: {obs_tensor.shape}, target action: 0, target value: 15.0")
        temp_optimizer = torch.optim.Adam(self._actor_critic.reward_critic.parameters(), lr=1e-3)
    
        # 测试Critic
        print("\n[1] Testing Critic...")
        for i in range(500):
            temp_optimizer.zero_grad()
            pred_value = self._actor_critic.reward_critic(obs_tensor)[0]
            loss = nn.functional.mse_loss(pred_value, target_tensor)
            loss.backward()
            if i == 0:
                total_norm = 0
                for p in self._actor_critic.reward_critic.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                print(f"  Initial gradient norm: {total_norm:.4f}")
                if total_norm < 1e-6:
                    print("  ❌ Gradient is too small, possible issue with backpropagation.")
                    # return False
            temp_optimizer.step()
            # self._actor_critic.reward_critic_optimizer.zero_grad()
            # pred_value = self._actor_critic.reward_critic(obs_tensor)[0]
            # loss = nn.functional.mse_loss(pred_value, target_tensor)
            # loss.backward()
            # self._actor_critic.reward_critic_optimizer.step()
            
            if i % 100 == 0:
                print(f"  Iter {i}: loss={loss.item():.4f}, pred={pred_value.item():.4f}")
        
        if loss.item() > 1.0:
            print(f"❌ Critic FAILED: final loss={loss.item():.4f}")
            return False
        print(f"✓ Critic PASSED: final loss={loss.item():.4f}")
        
        # 测试Actor
        print("\n[2] Testing Actor...")
        for i in range(500):
            self._actor_critic.actor_optimizer.zero_grad()
            dist = self._actor_critic.actor(obs_tensor)
            log_prob = dist.log_prob(action_tensor)
            loss = -log_prob.mean()
            loss.backward()
            self._actor_critic.actor_optimizer.step()
            
            if i % 100 == 0:
                probs = dist.probs[0].detach().cpu().numpy()
                print(f"  Iter {i}: loss={loss.item():.4f}, probs={probs}")
        
        final_probs = dist.probs[0].detach().cpu().numpy()
        if final_probs[0] < 0.8:
            print(f"❌ Actor FAILED: action 0 prob={final_probs[0]:.3f}")
            return False
        print(f"✓ Actor PASSED: probs={final_probs}")
        
        print("===================\n")
        return True

    def _compute_adv_surrogate(  # pylint: disable=unused-argument
        self,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> torch.Tensor:
        """Compute surrogate loss.

        Policy Gradient only use reward advantage.

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The advantage function of reward to update policy network.
        """
        return adv_r

    def _loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computing pi/actor loss.

        In Policy Gradient, the loss is defined as:

        .. math::

            L = -\underset{s_t \sim \rho_{\theta}}{\mathbb{E}} [
                \sum_{t=0}^T ( \frac{\pi^{'}_{\theta}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)} )
                 A^{R}_{\pi_{\theta}}(s_t, a_t)
            ]

        where :math:`\pi_{\theta}` is the policy network, :math:`\pi^{'}_{\theta}`
        is the new policy network, :math:`A^{R}_{\pi_{\theta}}(s_t, a_t)` is the advantage.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log probability`` of action sampled from buffer.
            adv (torch.Tensor): The ``advantage`` processed. ``reward_advantage`` here.

        Returns:
            The loss of pi/actor.
        """
        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        ratio = torch.exp(logp_ - logp)
        loss = -(ratio * adv).mean()
        entropy = distribution.entropy().mean().item()
        if self._cfgs.model_cfgs.actor_type == 'gaussian_learning' and hasattr(self._actor_critic.actor, 'std'):
            print('!!!',self._cfgs.model_cfgs.actor_type)
            std = self._actor_critic.actor.std
            self._logger.store(
            {
                'Train/PolicyStd': std,
            },
        )
        self._logger.store(
            {
                'Train/Entropy': entropy,
                'Train/PolicyRatio': ratio,
                'Loss/Loss_pi': loss.mean().item(),
            },
        )
        return loss
