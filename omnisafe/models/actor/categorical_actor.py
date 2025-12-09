from omnisafe.models.base import Actor
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical, Distribution
from omnisafe.utils.model import build_mlp_network

class BinItemMatcher(nn.Module):
    def __init__(self, bin_dim, item_dim, hidden_sizes, activation='relu'):
        super().__init__()
        
        # 我们取 hidden_sizes 的第一层作为 Embedding 维度
        embed_dim = hidden_sizes[0] 
        
        # 1. 箱子特征编码器 (Bin Tower)
        # Input: bin_dim -> Output: embed_dim
        self.bin_encoder = nn.Sequential(
            nn.Linear(bin_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
        
        # 2. 物品特征编码器 (Item Tower)
        # Input: item_dim -> Output: embed_dim
        self.item_encoder = nn.Sequential(
            nn.Linear(item_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
        
        # 3. 融合后的处理层 (Interaction MLP)
        # Input: embed_dim * 2 (拼接后) -> Output: embed_dim
        # 这一层负责学习 "匹配关系"
        self.interact_net = build_mlp_network(
            sizes=[embed_dim * 2, *hidden_sizes], 
            activation=activation
        )
        
        # 初始化
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                torch.nn.init.constant_(m.bias, 0.0)

    def forward(self, bin_feats, item_feat):
        """
        bin_feats: (Batch, Num_Bins, Bin_Dim)
        item_feat: (Batch, Item_Dim) 
        """
        batch_size, num_bins, _ = bin_feats.shape
        
        # --- A. 分别编码 ---
        bin_emb = self.bin_encoder(bin_feats) # (Batch, N, Hidden)
        
        # Item 需要扩充维度以匹配箱子数量
        # (Batch, Hidden) -> (Batch, 1, Hidden) -> (Batch, N, Hidden)
        item_emb = self.item_encoder(item_feat).unsqueeze(1).expand(-1, num_bins, -1)
        
        # --- B. 特征融合 (Concatenate) ---
        # 将两者拼在一起，让后面的 MLP 去计算它们的非线性关系
        combined = torch.cat([bin_emb, item_emb], dim=-1) # (Batch, N, 2*Hidden)
        
        # --- C. 深度交互 ---
        features = self.interact_net(combined) # (Batch, N, Hidden)
        
        return features

# class CategoricalActor(Actor):
#     def __init__(self, obs_space, act_space, hidden_sizes, activation = 'relu', weight_initialization_mode = 'kaiming_uniform',
#                 item_dim=3, bin_state_dim=5):
#         super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)
#         self.num_bins = self._act_dim
#         self.item_dim = item_dim
#         print(f"DEBUG: CategoricalActor bin_state_dim: {bin_state_dim}")
#         self.bin_state_dim = bin_state_dim
#         self.matcher = BinItemMatcher(bin_state_dim, item_dim, hidden_sizes, activation)
#         self.head = nn.Linear(hidden_sizes[-1], 1)
#         torch.nn.init.orthogonal_(self.head.weight, gain=0.01)
#         torch.nn.init.constant_(self.head.bias, 0.0)
        
#         self.ln_bin = nn.Identity()
#         self.ln_item = nn.Identity()
    
#     def _distribution(self, obs: torch.Tensor) -> Categorical:
#         """
#         Args: obs
#         Return: the categorical distribution
#         """
#         # print(obs.shape)
#         self._device = self.head.weight.device
#         # print(f"DEBUG: obs device: {obs.device}, model device: {self._device}")
#         if obs.device != self._device:
#             obs = obs.to(self._device)
#         if obs.dim() == 1:
#             obs = obs.unsqueeze(0)
#         batch_size = obs.shape[0]
#         mask = obs[..., :self._act_dim] # extract mask
#         # print(f"DEBUG: obs: {obs}, mask : {mask}")
#         bin_part_end = self.num_bins + (self.num_bins * self.bin_state_dim)
#         bin_flat = obs[:, self.num_bins:bin_part_end] # extract bin states
#         item_features = obs[:, bin_part_end:] # extract item features
        
#         bin_features = bin_flat.view(batch_size, self.num_bins, self.bin_state_dim)
        
#         bin_features = self.ln_bin(bin_features)
#         item_features = self.ln_item(item_features)
        
#         features = self.matcher(bin_features, item_features) # (B, N, H)
#         logits = self.head(features).squeeze(-1) # (B, N)
#         # logits = 5.0 * torch.tanh(logits / 5.0)
#         bool_mask = (mask < 0.5)
#         all_masked = bool_mask.all(dim=-1, keepdim=True)
#         if all_masked.any():
#             bool_mask = torch.where(all_masked, torch.tensor(False, device=logits.device), bool_mask)
#         masked_logits = logits.masked_fill(bool_mask, -1e2)
#         return Categorical(logits=masked_logits) # important, used to avoid being considered as probs
    
#     def forward(self, obs: torch.Tensor)-> Distribution:
#         """
#         Args: obs (torch.Tensor): Observation from environments.
#         Returns: The current distribution.
#         """
#         self._current_dist = self._distribution(obs)
#         self._after_inference = True
#         return self._current_dist

#     def predict(self, obs, deterministic = False)-> torch.Tensor:
#         """Predict the action given observation
#         Args:
#             obs (torch.Tensor): Observation from environments.
#             deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

#         Returns: The mean of the distribution if deterministic is True, otherwise the sampled action.
#         """
#         self._current_dist = self._distribution(obs)
#         self._after_inference = True
#         if deterministic:
#             # return self._current_dist.probs.argmax(dim=-1)
#             action = torch.argmax(self._current_dist.probs, dim=-1)
#         else:
#             action = self._current_dist.sample()
#         return action.squeeze(-1) if action.dim() > 1 else action
    
#     def log_prob(self, act) ->torch.Tensor:
#         """compute the log prob of action

#         Args:
#             act (_type_): _description_

#         Returns:
#             torch.Tensor: _description_
#         """
#         assert self._after_inference, 'log_prob() should be called after predict() or forward()'
#         self._after_inference = False
#         if act.dim() == 1:
#             act = act.unsqueeze(1)
#         return self._current_dist.log_prob(act.long())
    
#     # to satisfy the format of base
#     @property
#     def std(self) -> float:
#         # pass
#         return torch.zeros(1, device=self._device)
    
#     @std.setter
#     def std(self, std) -> None:
#         pass

class CategoricalActor(Actor):
    """简化版Actor - 直接MLP，与SimpleBSCritic匹配"""
    def __init__(self, obs_space, act_space, hidden_sizes, 
                 activation='relu', weight_initialization_mode='kaiming_uniform',
                 **kwargs):  # 接受但忽略item_dim, bin_state_dim等
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)
        self.num_bins = self._act_dim
        
        # 直接处理完整obs
        obs_dim = obs_space.shape[0]
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], self.num_bins)
        )
        
        # Xavier初始化
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # 最后一层小初始化
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.constant_(self.net[-1].bias, 0.0)
    
    def _distribution(self, obs: torch.Tensor) -> Categorical:
        device = next(self.parameters()).device
        if obs.device != device:
            obs = obs.to(device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        mask = obs[..., :self.num_bins]
        if not hasattr(self, '_printed_actor_obs'):
            self._printed_actor_obs = True
            print(f"\n=== Actor Input ===")
            print(f"Obs[0]: {obs[0]}")
            print(f"Obs range: [{obs.min():.4f}, {obs.max():.4f}]")
        
        # 直接过MLP
        logits = self.net(obs)
        
        if not hasattr(self, '_printed_logits'):
            self._printed_logits = True
            print(f"Raw logits[0]: {logits[0]}")
            print(f"Logits range: [{logits.min():.4f}, {logits.max():.4f}, {logits.mean():.4f}, {logits.std():.4f}]")
        # Mask处理
        bool_mask = (mask < 0.5)
        all_masked = bool_mask.all(dim=-1, keepdim=True)
        if all_masked.any():
            bool_mask = torch.where(all_masked, torch.tensor(False, device=logits.device), bool_mask)
        
        masked_logits = logits.masked_fill(bool_mask, -20.0) # solve logits std explosion
        return Categorical(logits=masked_logits)
    
    def forward(self, obs):
        self._current_dist = self._distribution(obs)
        self._after_inference = True
        return self._current_dist
    
    def predict(self, obs, deterministic=False):
        self._current_dist = self._distribution(obs)
        self._after_inference = True
        action = self._current_dist.probs.argmax(-1) if deterministic else self._current_dist.sample()
        return action.squeeze(-1) if action.dim() > 1 else action
    
    def log_prob(self, act):
        assert self._after_inference
        self._after_inference = False
        if act.dim() == 1:
            act = act.unsqueeze(1)
        return self._current_dist.log_prob(act.long())

    @property
    def std(self):
        return torch.zeros(1)
    
    @std.setter
    def std(self, std):
        pass
    
    