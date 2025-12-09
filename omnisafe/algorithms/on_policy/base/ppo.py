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
"""Implementation of the PPO algorithm."""

from __future__ import annotations

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.policy_gradient import PolicyGradient


@registry.register
class PPO(PolicyGradient):
    """The Proximal Policy Optimization (PPO) algorithm.

    References:
        - Title: Proximal Policy Optimization Algorithms
        - Authors: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov.
        - URL: `PPO <https://arxiv.org/abs/1707.06347>`_
    """

    def _loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computing pi/actor loss.

        In Proximal Policy Optimization, the loss is defined as:

        .. math::

            L^{CLIP} = \underset{s_t \sim \rho_{\theta}}{\mathbb{E}} \left[
                \min ( r_t A^{R}_{\pi_{\theta}} (s_t, a_t) , \text{clip} (r_t, 1 - \epsilon, 1 + \epsilon)
                A^{R}_{\pi_{\theta}} (s_t, a_t)
            \right]

        where :math:`r_t = \frac{\pi_{\theta}^{'} (a_t|s_t)}{\pi_{\theta} (a_t|s_t)}`,
        :math:`\epsilon` is the clip parameter, and :math:`A^{R}_{\pi_{\theta}} (s_t, a_t)` is the
        advantage.

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
        
        # with torch.no_grad():
        #     dist = distribution  # already computed
        #     print("=== DIST TYPE ===")
        #     print(type(dist), repr(dist))
        #     # Print common attrs if present
        #     for attr in ['logits', 'probs', 'mean', 'variance', 'scale', 'stddev', 'scale_tril', 'variance', 'entropy']:
        #         if hasattr(dist, attr):
        #             val = getattr(dist, attr)
        #             try:
        #                 # if tensor or returns tensor
        #                 if torch.is_tensor(val):
        #                     print(f" attr {attr}: shape={val.shape}, mean={val.mean().item():.6g}, std={val.std().item():.6g}, min={val.min().item():.6g}, max={val.max().item():.6g}")
        #                 else:
        #                     print(f" attr {attr}: {val}")
        #             except Exception as e:
        #                 print(f" attr {attr}: (error printing) {e}")

        #     # Print actor-side parameters that commonly control scale
        #     print("\n=== ACTOR PARAM STATS (first 200 params) ===")
        #     for name, p in list(self._actor_critic.actor.named_parameters())[:200]:
        #         if p is None: continue
        #         try:
        #             print(f"{name}: shape={tuple(p.shape)}, mean={p.data.mean().item():.6g}, std={p.data.std().item():.6g}, min={p.data.min().item():.6g}, max={p.data.max().item():.6g}, requires_grad={p.requires_grad}")
        #         except Exception as e:
        #             print(name, "print error", e)

        #     # If actor has an explicit std/log_std attribute
        #     if hasattr(self._actor_critic.actor, 'log_std'):
        #         ls = self._actor_critic.actor.log_std
        #         if torch.is_tensor(ls):
        #             print("actor.log_std:", ls.shape, "mean", ls.mean().item(), "std", ls.std().item(), "min", ls.min().item(), "max", ls.max().item())
        #         else:
        #             print("actor.log_std:", ls)

        #     if hasattr(self._actor_critic.actor, 'std'):
        #         s = self._actor_critic.actor.std
        #         if torch.is_tensor(s):
        #             print("actor.std:", s.shape, "mean", s.mean().item(), "std", s.std().item(), "min", s.min().item(), "max", s.max().item())
        #         else:
        #             print("actor.std:", s)

        
        # # === DEBUG: check whether forward() and log_prob() use the same distribution ===
        # with torch.no_grad():
        #     # distribution from forward(obs)
        #     dist_forward = distribution
        #     logp_from_forward = dist_forward.log_prob(act)

        #     # distribution from a *fresh forward* (forward again)
        #     dist_fresh = self._actor_critic.actor(obs)
        #     logp_from_fresh_forward = dist_fresh.log_prob(act)

        #     # distribution used inside actor.log_prob (whatever it computes)
        #     logp_from_logprob = self._actor_critic.actor.log_prob(act)

        #     print("[DEBUG] logp diff (forward vs log_prob):",
        #         (logp_from_forward - logp_from_logprob).abs().mean().item())

        #     print("[DEBUG] logp diff (forward vs fresh_forward):",
        #         (logp_from_forward - logp_from_fresh_forward).abs().mean().item())

        #     print("[DEBUG] logits std forward:",
        #         dist_forward.logits.std().item())

        #     print("[DEBUG] logits std fresh:", dist_fresh.logits.std().item())

        ratio = torch.exp(logp_ - logp)
        ratio_cliped = torch.clamp(
            ratio,
            1 - self._cfgs.algo_cfgs.clip,
            1 + self._cfgs.algo_cfgs.clip,
        )
        # print("[DEBUG] adv mean/std:", adv.mean().item(), adv.std().item())
        loss = -torch.min(ratio * adv, ratio_cliped * adv).mean()
        loss -= self._cfgs.algo_cfgs.entropy_coef * distribution.entropy().mean()
        # useful extra info
        entropy = distribution.entropy().mean().item()
        if self._cfgs.model_cfgs.actor_type == 'gaussian_learning' and hasattr(self._actor_critic.actor, 'std') :
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
