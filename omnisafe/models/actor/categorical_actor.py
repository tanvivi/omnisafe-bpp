from omnisafe.models.base import Actor
import torch
from torch.distributions import Categorical, Distribution
from omnisafe.utils.model import build_mlp_network

class CategoricalActor(Actor):
    def __init__(self, obs_space, act_space, hidden_sizes, activation = 'relu', weight_initialization_mode = 'kaiming_uniform'):
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)
        
        self.net: torch.nn.Module = build_mlp_network(
            sizes=[self._obs_dim, *self._hidden_sizes, self._act_dim],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )
        self._device = next(self.net.parameters()).device
        # line 17-21为了应对过大KL
        last_layer = None
        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                last_layer = m
        
        if last_layer is not None:
            # 使用很小的 gain 初始化权重，Bias 设为 0
            torch.nn.init.orthogonal_(last_layer.weight, gain=0.01)
            torch.nn.init.constant_(last_layer.bias, 0.0)
    
    def _distribution(self, obs: torch.Tensor) -> Categorical:
        """
        Args: obs
        Return: the categorical distribution
        """
        logits = self.net(obs)
        mask = obs[..., :self._act_dim]
        HUGE_NEG = -1e8
        masked_logits = torch.where(mask > 0.5, logits, torch.tensor(HUGE_NEG, device=logits.device, dtype=logits.dtype))
        all_masked = (mask <= 0.5).all(dim=-1, keepdim=True)
        masked_logits = torch.where(all_masked, logits, masked_logits)
        if masked_logits.dim() == 2:
            masked_logits = masked_logits.unsqueeze(1)
        return Categorical(logits=masked_logits) # important, used to avoid being considered as probs
    
    def forward(self, obs: torch.Tensor)-> Distribution:
        """
        Args: obs (torch.Tensor): Observation from environments.
        Returns: The current distribution.
        """
        self._current_dist = self._distribution(obs)
        self._after_inference = True
        return self._current_dist

    def predict(self, obs, deterministic = False)-> torch.Tensor:
        """Predict the action given observation
        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns: The mean of the distribution if deterministic is True, otherwise the sampled action.
        """
        self._current_dist = self._distribution(obs)
        self._after_inference = True
        if deterministic:
            # return self._current_dist.probs.argmax(dim=-1)
            action = torch.argmax(self._current_dist.probs, dim=-1)
        else:
            action = self._current_dist.sample()
        return action.squeeze(-1) if action.dim() > 1 else action
    
    def log_prob(self, act) ->torch.Tensor:
        """compute the log prob of action

        Args:
            act (_type_): _description_

        Returns:
            torch.Tensor: _description_
        """
        assert self._after_inference, 'log_prob() should be called after predict() or forward()'
        self._after_inference = False
        if act.dim() == 1:
            act = act.unsqueeze(1)
        return self._current_dist.log_prob(act.long())
    
    # to satisfy the format of base
    @property
    def std(self) -> float:
        # pass
        return torch.zeros(1, device=self._device)
    
    @std.setter
    def std(self, std) -> None:
        pass
    
    