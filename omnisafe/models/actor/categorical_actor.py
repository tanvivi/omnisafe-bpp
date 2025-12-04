from omnisafe.models.base import Actor
import torch
from torch.distributions import Categorical, Distribution
from omnisafe.utils.model import build_mlp_network

class CategoricalActor(Actor):
    def __init__(self, obs_space, act_space, hidden_sizes, activation = 'relu', weight_initialization_mode = 'kaiming_uniform',
                item_dim=3, bin_state_dim=5):
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)
        self.num_bins = self._act_dim
        self.item_dim = item_dim
        self.bin_state_dim = bin_state_dim
        input_dim = self.bin_state_dim + self.item_dim
        self.logits_scale = 0.2
        self.net: torch.nn.Module = build_mlp_network(
            sizes=[input_dim, *self._hidden_sizes, 1],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )
        self.ln = torch.nn.LayerNorm(self.bin_state_dim + self.item_dim)
        self._device = next(self.net.parameters()).device
        # line 17-21为了应对过大KL
        last_layer = None
        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                last_layer = m
        
        if last_layer is not None:
            # 使用很小的 gain 初始化权重，Bias 设为 0
            torch.nn.init.uniform_(last_layer.weight, -0.001, 0.001)
            torch.nn.init.constant_(last_layer.bias, 0.0)
    
    def _distribution(self, obs: torch.Tensor) -> Categorical:
        """
        Args: obs
        Return: the categorical distribution
        """
        # print(obs.shape)
        self._device = self.ln.weight.device
        # print(f"DEBUG: obs device: {obs.device}, model device: {self._device}")
        if obs.device != self._device:
            obs = obs.to(self._device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        batch_size = obs.shape[0]
        mask = obs[..., :self._act_dim] # extract mask
        # print(f"DEBUG: obs: {obs}, mask : {mask}")
        bin_part_end = self.num_bins + (self.num_bins * self.bin_state_dim)
        bin_flat = obs[:, self.num_bins:bin_part_end] # extract bin states
        item_features = obs[:, bin_part_end:] # extract item features
        bin_features = bin_flat.view(batch_size, self.num_bins, self.bin_state_dim) # reshape to (batch_size, num_bins, bin_state_dim)
        item_expanded = item_features.unsqueeze(1).expand(-1, self.num_bins, -1) # (batch_size, num_bins, item_dim)
        features = torch.cat([bin_features, item_expanded], dim=2) # (batch_size, num_bins, bin_state_dim + item_dim)
        # batch_size * num_bins * input_dim -> batch_size x num_bins x 1
        features = self.ln(features) # only use feature to compute logits
        logits = self.net(features).squeeze(-1) # (batch_size, num_bins)
        # logits = 5.0 * torch.tanh(logits / 5.0)
        bool_mask = (mask < 0.5)
        all_masked = bool_mask.all(dim=-1, keepdim=True)
        if all_masked.any():
            bool_mask = torch.where(all_masked, torch.tensor(False, device=logits.device), bool_mask)
        masked_logits = logits.masked_fill(bool_mask, -1e2)
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
    
    