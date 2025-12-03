"""Implementation of BSCritic."""

from __future__ import annotations

import torch
import torch.nn as nn

from omnisafe.models.base import Critic
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network

class BinNetwork(nn.Module):
    def __init__(self, 
                input_dim: int,
                hidden_sizes: list[int],
                activation: Activation = 'relu',
                weight_initialization_mode: InitFunction = 'kaiming_uniform',
                num_bins: int = 2,
                bin_state_dim: int = 4,
                item_dim: int = 3):
        print(f"DEBUG: Initializing BinNetwork with bins={num_bins}")
        super().__init__()
        self.num_bins = num_bins
        self.bin_state_dim = bin_state_dim
        self.item_dim = item_dim
        input_dim = self.bin_state_dim + self.item_dim
        
        self.ln = nn.LayerNorm(input_dim)
        self.encoder = build_mlp_network(
            sizes=[input_dim, *hidden_sizes],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )
        self.value_head = nn.Linear(hidden_sizes[-1], 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0) 
        nn.init.constant_(self.value_head.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        batch_size = obs.shape[0]
        
        bin_part_end = self.num_bins + (self.num_bins * self.bin_state_dim)
        bin_flat = obs[:, self.num_bins:bin_part_end] # extract bin states
        item_features = obs[:, bin_part_end:] # extract item features
        
        bin_features = bin_flat.view(batch_size, self.num_bins, self.bin_state_dim) # reshape to (batch_size, num_bins, bin_state_dim)
        item_expanded = item_features.unsqueeze(1).expand(-1, self.num_bins, -1) # (batch_size, num_bins, item_dim)
        
        features = torch.cat([bin_features, item_expanded], dim=2) # (batch_size, num_bins, bin_state_dim + item_dim)
        features = self.ln(features)
        
        bin_embeddings = self.encoder(features)
        
        global_values, _ = torch.max(bin_embeddings, dim=1)
        
        values = self.value_head(global_values)
        
        return values

class BSCritic(Critic):
    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        num_critics: int = 1,
        item_dim: int = 3,
        bin_state_dim: int = 4,
        num_bins: int = 2
    ) -> None:
        super().__init__(
            obs_space,
            act_space,
            hidden_sizes,
            activation,
            weight_initialization_mode,
            num_critics,
            use_obs_encoder=False,
        )
        self.item_dim = item_dim
        self.bin_state_dim = bin_state_dim
        self.num_bins = act_space.n if num_bins is None else num_bins
        self.net_lst: list[nn.Module]
        self.net_lst = []

        for idx in range(self._num_critics):
            net = BinNetwork(
                input_dim=self.bin_state_dim + self.item_dim,
                hidden_sizes=self._hidden_sizes,
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
                num_bins=self.num_bins,
                bin_state_dim=self.bin_state_dim,
                item_dim=self.item_dim
            )
            
            self.net_lst.append(net)
            self.add_module(f'critic_{idx}', net)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward function
        Args:
            obs (torch.Tensor): _description_

        Returns:
            list of V critic values (1 for PPO)
        """
        res = []
        for critic in self.net_lst:
            # res.append(torch.squeeze(critic(obs), -1))
            res.append(critic(obs).view(-1))
        return res