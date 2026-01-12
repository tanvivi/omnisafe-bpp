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
        self.diagnostics = OmniSafeDiagnostics(enable=True,
        log_freq=10)

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
        self._logger.register_key('Diagnostics/CriticGradNorm', delta=True)
        self._logger.register_key('Diagnostics/ActorGradNorm', delta=True)
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
            self._update(epoch)
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

    def _update(self, epoch) -> None:
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
        current_epoch = epoch
        if self.diagnostics.should_diagnose(current_epoch):
            self.diagnostics.diagnose_buffer_data(data, self._logger)
            self.diagnostics.diagnose_policy_before_update(
                self._actor_critic,
                data['obs'][:1024],
                data['act'][:1024],
                data['adv_r'][:1024],
                sample_size=1024
            )
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

        with torch.no_grad():
            # 注意：如果数据量太大，显存可能不够，这里假设显存足够
            # 使用 original_obs (你在代码前面定义的) 和 data['target_value_r']
            values_all_full = self._actor_critic.reward_critic(data['obs'])[0].squeeze(-1)
            
            y_true_full = data['target_value_r']
            var_y_full = torch.var(y_true_full)
            
            explained_var_full = 1.0 - torch.var(y_true_full - values_all_full) / (var_y_full + 1e-8)
            
        print(f"Global Explained var: {explained_var_full.item():.6f}")

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

        # Log gradient norms before clipping (for diagnostics)
        critic_grad_norm = 0.0
        for param in self._actor_critic.reward_critic.parameters():
            if param.grad is not None:
                critic_grad_norm += param.grad.data.norm(2).item() ** 2
        critic_grad_norm = critic_grad_norm ** 0.5

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

        # Store gradient norm and loss for monitoring
        self._logger.store({
            'Diagnostics/CriticGradNorm': critic_grad_norm,
            'Loss/Loss_reward_critic': loss.item(),
        })

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

        # Log gradient norms before clipping (for diagnostics)
        actor_grad_norm = 0.0
        for p in self._actor_critic.actor.parameters():
            if p.grad is not None:
                actor_grad_norm += p.grad.data.norm(2).item() ** 2
        actor_grad_norm = actor_grad_norm ** 0.5

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.actor)
        self._actor_critic.actor_optimizer.step()

        # Store gradient norm for monitoring
        self._logger.store({'Diagnostics/ActorGradNorm': actor_grad_norm})
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

class OmniSafeDiagnostics:
    """
    适配OmniSafe框架的诊断工具
    Diagnostic tool adapted for OmniSafe framework
    """
    
    def __init__(self, enable=True, log_freq=100):
        self.enable = enable
        self.log_freq = log_freq
        self.step_count = 0
        
    def should_diagnose(self, epoch: int) -> bool:
        """判断是否应该进行诊断 / Check if should diagnose"""
        return self.enable and (epoch % self.log_freq == 0)
    
    def diagnose_buffer_data(self, data: dict, logger) -> None:
        """
        诊断buffer中的数据质量
        Diagnose data quality in buffer
        
        在 _update() 开始时调用
        Call at the beginning of _update()
        """
        print("\n" + "="*80)
        print("📊 Buffer数据诊断 / Buffer Data Diagnosis")
        print("="*80)
        
        obs = data['obs']
        act = data['act']
        adv_r = data['adv_r']
        target_value_r = data['target_value_r']
        rewards = data.get('reward', None)
        
        # 1. 奖励统计 / Reward statistics
        if rewards is not None:
            print(f"\n奖励分布 / Reward distribution:")
            print(f"  Mean: {rewards.mean():.4f}")
            print(f"  Std: {rewards.std():.4f}")
            print(f"  Range: [{rewards.min():.4f}, {rewards.max():.4f}]")
            print(f"  非零比例 / Non-zero ratio: {(rewards != 0).sum().item() / rewards.numel():.2%}")
        
        # 2. Advantage统计 / Advantage statistics
        print(f"\nAdvantage分布 / Advantage distribution:")
        print(f"  Mean: {adv_r.mean():.4f}")
        print(f"  Std: {adv_r.std():.4f}")
        print(f"  Range: [{adv_r.min():.4f}, {adv_r.max():.4f}]")
        
        if adv_r.std() < 0.1:
            print("  ⚠️  警告：Advantage方差太小！")
            print("  ⚠️  Warning: Advantage variance too small!")
        
        # 3. Value target统计 / Value target statistics
        print(f"\nValue Target统计 / Value Target statistics:")
        print(f"  Mean: {target_value_r.mean():.4f}")
        print(f"  Std: {target_value_r.std():.4f}")
        print(f"  Range: [{target_value_r.min():.4f}, {target_value_r.max():.4f}]")
        
        # 4. 动作分布 / Action distribution
        print(f"\n动作分布 / Action distribution:")
        unique_actions, counts = torch.unique(act, return_counts=True)
        for action, count in zip(unique_actions, counts):
            print(f"  Action {action.item()}: {count.item()} ({count.item()/act.numel():.2%})")
        
        # 判断动作是否过于集中 / Check if actions are too concentrated
        max_ratio = counts.max().float() / act.numel()
        if max_ratio > 0.8:
            print(f"  ⚠️  警告：动作分布过于集中！最高频率={max_ratio:.2%}")
            print(f"  ⚠️  Warning: Action distribution too concentrated! Max freq={max_ratio:.2%}")
    
    def diagnose_policy_before_update(
        self, 
        actor_critic, 
        obs: torch.Tensor,
        act: torch.Tensor,
        adv_r: torch.Tensor,
        sample_size: int = 128
    ) -> dict:
        """
        在更新前诊断策略状态
        Diagnose policy state before update
        
        在 _update() 中，actor更新之前调用
        Call in _update() before actor update
        """
        print("\n" + "="*80)
        print("🔍 策略更新前诊断 / Pre-Update Policy Diagnosis")
        print("="*80)
        
        with torch.no_grad():
            # 采样一小部分数据进行分析 / Sample a subset for analysis
            indices = torch.randperm(obs.size(0))[:sample_size]
            obs_sample = obs[indices]
            act_sample = act[indices]
            adv_sample = adv_r[indices]
            
            # 获取当前策略分布 / Get current policy distribution
            distribution = actor_critic.actor(obs_sample)
            logits = distribution.logits if hasattr(distribution, 'logits') else None
            probs = distribution.probs if hasattr(distribution, 'probs') else None
            entropy = distribution.entropy()
            
            # 获取value估计 / Get value estimates
            values = actor_critic.reward_critic(obs_sample)[0]
            
            print(f"\n采样大小 / Sample size: {sample_size}")
            # 在获取 obs_sample 之后
            print(f"Obs 差异性检查 (Std): {obs_sample.float().std(dim=0).sum().item():.6f}")
            # 1. Logits分析 / Logits analysis
            if logits is not None:
                print(f"\nLogits统计 / Logits statistics:")
                print(f"  Mean: {logits.mean():.4f}")
                print(f"  Std: {logits.std():.4f}")
                print(f"  Range: [{logits.min():.4f}, {logits.max():.4f}]")
                
                # 计算logits区分度 / Calculate logits discrimination
                logit_diffs = []
                for i in range(min(10, logits.size(0))):  # 只看前10个样本
                    diff = logits[i].max() - logits[i].min()
                    logit_diffs.append(diff.item())
                
                avg_diff = sum(logit_diffs) / len(logit_diffs)
                print(f"  平均Logits差异 / Avg logits difference: {avg_diff:.4f}")
                
                if avg_diff < 0.5:
                    print("  🚨 严重：Logits差异极小！策略几乎随机")
                    print("  🚨 CRITICAL: Logits difference tiny! Policy nearly random")
                elif avg_diff < 2.0:
                    print("  ⚠️  警告：Logits差异较小，特征可能不够强")
                    print("  ⚠️  Warning: Logits difference small, features may be weak")
                else:
                    print("  ✅ Logits差异合理")
                    print("  ✅ Logits difference reasonable")
            
            # 2. 概率分布分析 / Probability distribution analysis
            if probs is not None:
                print(f"\n概率分布统计 / Probability distribution:")
                print(f"  最大概率均值 / Max prob mean: {probs.max(dim=-1)[0].mean():.4f}")
                print(f"  最小概率均值 / Min prob mean: {probs.min(dim=-1)[0].mean():.4f}")
            
            # 3. 熵分析 / Entropy analysis
            print(f"\n熵统计 / Entropy statistics:")
            print(f"  Mean: {entropy.mean():.4f}")
            print(f"  Std: {entropy.std():.4f}")
            
            # 计算理论最大熵 / Calculate theoretical max entropy
            num_actions = logits.size(-1) if logits is not None else probs.size(-1)
            max_entropy = torch.log(torch.tensor(float(num_actions)))
            normalized_entropy = entropy.mean() / max_entropy
            print(f"  归一化熵 / Normalized entropy: {normalized_entropy:.4f}")
            
            if normalized_entropy > 0.9:
                print("  ⚠️  警告：熵接近最大值，策略接近均匀分布")
                print("  ⚠️  Warning: Entropy close to max, policy nearly uniform")
            
            # # 4. 同一obs下不同action的对比 / Compare different actions for same obs
            # print(f"\n动作-Advantage对比分析 / Action-Advantage comparison:")
            # self._analyze_action_advantage_correlation(
            #     obs_sample[:20], act_sample[:20], adv_sample[:20], 
            #     probs[:20] if probs is not None else None
            # )
            
            # 5. Value估计质量 / Value estimation quality
            print(f"\nValue估计 / Value estimates:")
            print(f"  Mean: {values.mean():.4f}")
            print(f"  Std: {values.std():.4f}")
            
            return {
                'logit_diff': avg_diff if logits is not None else None,
                'entropy': entropy.mean().item(),
                'normalized_entropy': normalized_entropy.item(),
            }
    
    # def _analyze_action_advantage_correlation(
    #     self, 
    #     obs: torch.Tensor, 
    #     act: torch.Tensor, 
    #     adv: torch.Tensor,
    #     probs: torch.Tensor = None
    # ):
    #     """
    #     分析动作和advantage的相关性
    #     Analyze correlation between actions and advantages
    #     """
    #     print("\n  前5个样本的详细信息 / Details for first 5 samples:")
    #     for i in range(20):
    #         print(f"\n  样本 {i+1} / Sample {i+1}:")
    #         print(f"    Advantage mean: {adv.mean():.4f}, std: {adv.std():.4f}")
    #         print(f"    选择的动作 / Selected action: {act[i].item()}")
    #         print(f"    Advantage: {adv[i].item():.4f}")
            
    #         if probs is not None:
    #             print(f"    各动作概率 / Action probabilities:")
    #             for j in range(probs.size(-1)):
    #                 marker = " ← 选中" if j == act[i].item() else ""
    #                 print(f"      Action {j}: {probs[i, j].item():.4f}{marker}")
    
    def diagnose_update_dynamics(
        self,
        old_logp: torch.Tensor,
        new_logp: torch.Tensor,
        ratio: torch.Tensor,
        adv: torch.Tensor,
        loss: torch.Tensor,
    ):
        """
        诊断更新动态
        Diagnose update dynamics
        
        在 _loss_pi() 中调用
        Call in _loss_pi()
        """
        print("\n" + "="*80)
        print("📈 更新动态诊断 / Update Dynamics Diagnosis")
        print("="*80)
        
        print(f"\nPolicy Ratio统计 / Policy Ratio statistics:")
        print(f"  Mean: {ratio.mean():.4f}")
        print(f"  Std: {ratio.std():.4f}")
        print(f"  Range: [{ratio.min():.4f}, {ratio.max():.4f}]")
        
        # 检查ratio是否过大 / Check if ratio is too large
        if ratio.max() > 3.0 or ratio.min() < 0.33:
            print("  ⚠️  警告：Ratio范围过大，可能需要调整学习率或clip")
            print("  ⚠️  Warning: Ratio range too large, may need to adjust LR or clip")
        
        print(f"\nLoss信息 / Loss info:")
        print(f"  Loss value: {loss.item():.4f}")
        
        # 检查loss和advantage的关系 / Check relationship between loss and advantage
        print(f"\nAdvantage统计（用于更新）/ Advantage statistics (for update):")
        print(f"  Mean: {adv.mean():.4f}")
        print(f"  Std: {adv.std():.4f}")
        
        # 计算加权advantage / Calculate weighted advantage
        weighted_adv = (ratio * adv).mean()
        print(f"  加权Advantage / Weighted advantage: {weighted_adv:.4f}")


# 集成到PolicyGradient类中 / Integration into PolicyGradient class
def integrate_diagnostics_into_pg(pg_instance):
    """
    将诊断功能集成到PolicyGradient实例中
    Integrate diagnostics into PolicyGradient instance
    
    使用方法 / Usage:
        在 __init__ 中调用 / Call in __init__:
        integrate_diagnostics_into_pg(self)
    """
    # 添加诊断器 / Add diagnostics
    pg_instance.diagnostics = OmniSafeDiagnostics(
        enable=True,  # 可以从config读取 / Can read from config
        log_freq=10   # 每10个epoch诊断一次 / Diagnose every 10 epochs
    )
    
    # 保存原始方法 / Save original methods
    pg_instance._original_update = pg_instance._update
    pg_instance._original_loss_pi = pg_instance._loss_pi
    
    # 包装_update方法 / Wrap _update method
    def _update_with_diagnostics(self):
        data = self._buf.get()
        
        # 诊断buffer数据 / Diagnose buffer data
        if self.diagnostics.should_diagnose(self._logger.get_stats('Train/Epoch')[0]):
            self.diagnostics.diagnose_buffer_data(data, self._logger)
            
            # 诊断策略状态 / Diagnose policy state
            self.diagnostics.diagnose_policy_before_update(
                self._actor_critic,
                data['obs'][:1024],  # 采样512个 / Sample 512
                data['act'][:1024],
                data['adv_r'][:1024],
                sample_size=1024
            )
        
        # 调用原始更新 / Call original update
        self._original_update()
    
    # 包装_loss_pi方法 / Wrap _loss_pi method
    def _loss_pi_with_diagnostics(self, obs, act, logp, adv):
        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        ratio = torch.exp(logp_ - logp)
        loss = -(ratio * adv).mean()
        entropy = distribution.entropy().mean().item()
        
        # 诊断更新动态（只在特定epoch）/ Diagnose update dynamics (only certain epochs)
        if self.diagnostics.should_diagnose(self._logger.get_stats('Train/Epoch')[0]):
            with torch.no_grad():
                # 采样一小部分进行诊断 / Sample subset for diagnosis
                sample_size = min(128, obs.size(0))
                indices = torch.randperm(obs.size(0))[:sample_size]
                self.diagnostics.diagnose_update_dynamics(
                    logp[indices],
                    logp_[indices],
                    ratio[indices],
                    adv[indices],
                    loss,
                )
        
        if self._cfgs.model_cfgs.actor_type == 'gaussian_learning' and hasattr(self._actor_critic.actor, 'std'):
            std = self._actor_critic.actor.std
            self._logger.store({'Train/PolicyStd': std})
        
        self._logger.store({
            'Train/Entropy': entropy,
            'Train/PolicyRatio': ratio,
            'Loss/Loss_pi': loss.mean().item(),
        })
        
        return loss
    
    # 替换方法 / Replace methods
    import types
    pg_instance._update = types.MethodType(_update_with_diagnostics, pg_instance)
    pg_instance._loss_pi = types.MethodType(_loss_pi_with_diagnostics, pg_instance)