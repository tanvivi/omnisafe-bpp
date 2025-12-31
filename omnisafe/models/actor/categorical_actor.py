import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from typing import Any, Tuple, Union, Sequence

def obs_processor(obs, num_bins):
    """convert flat observation to structured format
    current obs is [mask, bin_features, item_features, global_features]
    mask: (num_bins,)
    bin_features: (num_bins, 5)"""
    mask = obs[:, :num_bins]
    bin_start = num_bins
    bin_end = num_bins + num_bins * 5
    bin_features = obs[:, bin_start:bin_end].reshape(-1, num_bins, 5)
    item_start = bin_end
    item_end = bin_end + 3
    item_features = obs[:, item_start:item_end]
    global_features = obs[:, item_end:]
    return mask, bin_features, item_features, global_features


class CategoricalActor(nn.Module):
    """OmniSafe-compatible Actor with shared Transformer backbone"""
    def __init__(
        self,
        obs_space,
        act_space,
        hidden_sizes: list = None,
        activation: str = 'relu',
        weight_initialization_mode: str = 'kaiming_uniform',
        num_bins: int = 5,
        bin_feature_dim: int = 5,
        bin_size: list = [10, 10, 10],
        device: Union[str, int, torch.device] = "cuda:0",
        **kwargs
    ) -> None:
        super().__init__()
        
        self._act_dim = num_bins
        self.num_bins = num_bins
        self.device = device
        input_dim = 4 + 5 + 3 + 5 # global features + bin features + item features + bin context features
        self.score_nn = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)
        )
        
        self._current_dist = None
        self._after_inference = False
        
    def _distribution(self, obs: torch.Tensor):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        
        device = next(self.parameters()).device
        if obs.device != device:
            obs = obs.to(device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        batch_size = obs.shape[0]
        
        # Parse mask
        mask, bin_features, item_features, global_features = obs_processor(obs, self.num_bins)
        mask = mask.bool()
        mask_f= mask.float()
        
        item_rep = item_features.unsqueeze(1).expand(batch_size, self.num_bins, -1)
        global_rep = global_features.unsqueeze(1).expand(batch_size, self.num_bins, -1)
        bin_context = (bin_features * mask_f.unsqueeze(-1)).sum(dim=1) / mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        bin_context_rep = bin_context.unsqueeze(1).expand(batch_size, self.num_bins, -1)
        cat_features = torch.cat([bin_features, item_rep, global_rep, bin_context_rep], dim=-1)
        # Compute logits
        raw_score = self.score_nn(cat_features).squeeze(-1)  # (batch_size, num_bins)
        # Apply mask
        if mask is not None:
            bool_mask = ~mask
            all_masked = bool_mask.all(dim=-1, keepdim=True)
            if all_masked.any():
                bool_mask = torch.where(all_masked, torch.tensor(False, device=device), bool_mask)

            logits = raw_score.masked_fill(bool_mask, -20.0)
        
        return Categorical(logits=logits)
    
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
        assert self._after_inference, "Must call forward() or predict() before log_prob()"
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
class BSCritic(nn.Module):
    """OmniSafe-compatible Critic with shared Transformer backbone"""
    def __init__(
        self,
        obs_space,
        act_space,
        hidden_sizes: list = None,
        activation: str = 'relu',
        weight_initialization_mode: str = 'kaiming_uniform',
        num_bins: int = 5,
        bin_feature_dim: int = 5,
        device: Union[str, int, torch.device] = "cuda:0",
        **kwargs
    ) -> None:
        super().__init__()
        
        self.device = device
        
        # Critic-specific layers
        self.num_bins = num_bins
        self.input_dim = 12
        
        # 1. Shared Encoder (特征提取器)
        # 作用：把每个 Bin 的原始数据映射为高维语义向量
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU()
        )
        
        # 2. Value Head (评分器)
        # 输入：Pooling 后的特征 (hidden_dim) + 显式的 Global 特征 (4)
        # 建议把 global features 再次拼接触入，强化 Critic 对 Load Balance 的感知
        self.value_head = nn.Sequential(
            nn.Linear(hidden_sizes[1] + 4, hidden_sizes[1]), 
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1) # 输出 V(s)
        )

        
    def forward(self, obs: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        
        device = next(self.parameters()).device
        if obs.device != device:
            obs = obs.to(device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        mask, bin_feats, item_feats, global_feats = obs_processor(obs, self.num_bins)
        batch_size = obs.shape[0]
        N = self.num_bins
        
        # 2. 特征扩展与拼接 (与 Actor 一样)
        item_rep = item_feats.unsqueeze(1).expand(batch_size, N, -1)
        global_rep = global_feats.unsqueeze(1).expand(batch_size, N, -1)
        cat_feats = torch.cat([bin_feats, item_rep, global_rep], dim=-1) # [Batch, N, 12]
        
        # 3. Parameter Sharing Encoding
        # [Batch, N, 12] -> [Batch, N, Hidden]
        bin_embeddings = self.encoder(cat_feats)
        
        # 4. Masked Pooling (泛化能力的核心！)
        # 我们需要把 N 个 Bin 的向量合并成 1 个，且要忽略 mask=0 的 Bin
        
        # 扩展 mask 维度: [Batch, N] -> [Batch, N, 1]
        mask_expanded = mask.unsqueeze(-1)
        
        # --- Option A: Sum/Mean Pooling (推荐用于 Load Balancing) ---
        # 先把无效 Bin 的 embedding 置 0
        masked_embeddings = bin_embeddings * mask_expanded
        
        # 求和
        sum_embeddings = masked_embeddings.sum(dim=1) # [Batch, Hidden]
        # 计算有效的 Bin 数量 (防止除以 0)
        valid_counts = mask_expanded.sum(dim=1).clamp(min=1.0)
        # 求平均
        pooled_embedding = sum_embeddings / valid_counts
        
        # --- Option B: Max Pooling (如果只关心能否装得下) ---
        # fill_value = -1e9
        # masked_embeddings = bin_embeddings.masked_fill(mask_expanded == 0, fill_value)
        # pooled_embedding = masked_embeddings.max(dim=1)[0]
        
        # 5. 再次拼接 Global Features (强化整体感知)
        # Critic 的核心任务是评估“局面好不好”，Load Balance 指标(std/mean)是决定性因素
        # 所以我们将 pooled_embedding 和 global_feats 拼在一起
        final_input = torch.cat([pooled_embedding, global_feats], dim=-1)
        
        # 6. 计算 Value
        state_value = self.value_head(final_input) # [Batch, 1]
        
        return state_value