"""Implementation of BSCritic."""

from __future__ import annotations

import torch
import torch.nn as nn

from omnisafe.models.base import Critic
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network
from omnisafe.models.actor.categorical_actor import BinItemMatcher

class BinNetwork(nn.Module):
    def __init__(self, 
                input_dim: int,
                hidden_sizes: list[int],
                activation: Activation = 'relu',
                weight_initialization_mode: InitFunction = 'kaiming_uniform',
                num_bins: int = 2,
                bin_state_dim: int = 5,
                item_dim: int = 3):
        print(f"DEBUG: Initializing BinNetwork with bins={num_bins}, bin_state_dim={bin_state_dim}, item_dim={item_dim}")
        super().__init__()
        self.num_bins = num_bins
        self.bin_state_dim = bin_state_dim
        self.item_dim = item_dim
        input_dim = self.bin_state_dim + self.item_dim
        self.matcher = BinItemMatcher(bin_state_dim, item_dim, hidden_sizes, activation)
        
        self.ln_bin = nn.Identity()
        self.ln_item = nn.Identity()
        
        self.value_head = nn.Linear(hidden_sizes[-1], 1)
        nn.init.orthogonal_(self.value_head.weight, gain=0.1) 
        nn.init.constant_(self.value_head.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        device = self.value_head.weight.device
        if obs.device != device:
            obs = obs.to(device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        batch_size = obs.shape[0]
        
        bin_part_end = self.num_bins + (self.num_bins * self.bin_state_dim)
        bin_flat = obs[:, self.num_bins:bin_part_end] # extract bin states
        item_features = obs[:, bin_part_end:] # extract item features
        
        bin_features = bin_flat.view(batch_size, self.num_bins, self.bin_state_dim) # reshape to (batch_size, num_bins, bin_state_dim)
        
        bin_features = self.ln_bin(bin_features)
        item_features = self.ln_item(item_features)     

        bin_embeddings = self.matcher(bin_features, item_features) # (B, N, H)
        
        global_values, _ = torch.max(bin_embeddings, dim=1)
        
        values = self.value_head(global_values)
        
        return values

# class BSCritic(Critic):
#     def __init__(
#         self,
#         obs_space: OmnisafeSpace,
#         act_space: OmnisafeSpace,
#         hidden_sizes: list[int],
#         activation: Activation = 'relu',
#         weight_initialization_mode: InitFunction = 'kaiming_uniform',
#         num_critics: int = 1,
#         item_dim: int = 3,
#         bin_state_dim: int = 5,
#         num_bins: int = 2
#     ) -> None:
#         super().__init__(
#             obs_space,
#             act_space,
#             hidden_sizes,
#             activation,
#             weight_initialization_mode,
#             num_critics,
#             use_obs_encoder=False,
#         )
#         self.item_dim = item_dim
#         self.bin_state_dim = bin_state_dim
#         self.num_bins = act_space.n if num_bins is None else num_bins
#         # self.net_lst: list[nn.Module]
#         # self.net_lst = []
#         self.net_list = nn.ModuleList()

#         for idx in range(self._num_critics):
#             net = BinNetwork(
#                 input_dim=self.bin_state_dim + self.item_dim,
#                 hidden_sizes=self._hidden_sizes,
#                 activation=self._activation,
#                 weight_initialization_mode=self._weight_initialization_mode,
#                 num_bins=self.num_bins,
#                 bin_state_dim=self.bin_state_dim,
#                 item_dim=self.item_dim
#             )
            
#             self.net_list.append(net)
#             self.add_module(f'critic_{idx}', net)
    
#     def forward(self, obs: torch.Tensor) -> torch.Tensor:
#         """
#         Forward function
#         Args:
#             obs (torch.Tensor): _description_

#         Returns:
#             list of V critic values (1 for PPO)
#         """
#         res = []
#         for critic in self.net_list:
#             # res.append(torch.squeeze(critic(obs), -1))
#             res.append(critic(obs).view(-1))
#         return res

class BSCritic(Critic):
    """简化版Critic - 直接MLP处理完整obs"""
    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        num_critics: int = 1,
        **kwargs  # 忽略多余参数
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
        
        obs_dim = obs_space.shape[0]
        
        self.net_list = nn.ModuleList()
        for idx in range(num_critics):
            net = nn.Sequential(
                nn.Linear(obs_dim, hidden_sizes[0]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[0], hidden_sizes[0]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[0], 1)
            )
            
            # Xavier初始化
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
            
            # 最后一层特殊初始化 - 输出接近期望的value
            nn.init.orthogonal_(net[-1].weight, gain=0.01)
            nn.init.constant_(net[-1].bias, 0.0)  # 初始化为reward中点
            
            self.net_list.append(net)
            self.add_module(f'critic_{idx}', net)
    
    def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
        device = next(self.parameters()).device
        if obs.device != device:
            obs = obs.to(device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        res = []
        for net in self.net_list:
            value = net(obs).view(-1)
            res.append(value)
        return res