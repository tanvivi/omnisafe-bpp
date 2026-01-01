import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from typing import Any, Tuple, Union, Sequence

def obs_processor(obs, num_bins, bin_feature_dim=5):
    """convert flat observation to structured format
    current obs is [mask, bin_features, item_features, global_features]
    mask: (num_bins,)
    bin_features: (num_bins, 5)"""
    mask = obs[:, :num_bins]
    bin_start = num_bins
    bin_end = num_bins + num_bins * bin_feature_dim
    bin_features = obs[:, bin_start:bin_end].reshape(-1, num_bins, bin_feature_dim)
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
        bin_state_dim: int = 6,
        bin_size: list = [10, 10, 10],
        device: Union[str, int, torch.device] = "cuda:0",
        **kwargs
    ) -> None:
        super().__init__()
        
        self._act_dim = num_bins
        self.num_bins = num_bins
        self.device = device
        self.bin_state_dim = bin_state_dim
        self.item_feature_dim = 3
        self.global_feature_dim = 4
        input_dim =  self.bin_state_dim  + self.item_feature_dim  # global features + bin features + item features + bin context features
        self.score_nn = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.LayerNorm(hidden_sizes[0]),  # 归一化帮助学习 / Normalization helps learning
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.LayerNorm(hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)
        )
        
        self._current_dist = None
        self._after_inference = False
        # 初始化 Actor 的最后一层
        last_layer = self.score_nn[-1]
        nn.init.constant_(last_layer.bias, 0.0)
        nn.init.orthogonal_(last_layer.weight, gain=0.01) # 权重极小

                
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
        mask, bin_features, item_features, global_features = obs_processor(obs, self.num_bins, self.bin_state_dim)
        mask = mask.bool()
        mask_f= mask.float()
        
        item_rep = item_features.unsqueeze(1).expand(batch_size, self.num_bins, -1)
        global_rep = global_features.unsqueeze(1).expand(batch_size, self.num_bins, -1)
        bin_context = (bin_features * mask_f.unsqueeze(-1)).sum(dim=1) / mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        bin_context_rep = bin_context.unsqueeze(1).expand(batch_size, self.num_bins, -1)
        cat_features = torch.cat([bin_features, item_rep], dim=-1)
        # Compute logits
        raw_score = self.score_nn(cat_features).squeeze(-1)  # (batch_size, num_bins)
        import random
        if random.random() < 0.001:
            print("\n"+"="*20)
            delta_feat = bin_features[0, :, -1] 
            print(f"input obs :{cat_features}")
            print(f"📊 Input Delta Feature (Raw): {delta_feat.detach().cpu().numpy()}")
            if torch.all(delta_feat == 0):
                print("⚠️ 警告: Delta 特征全为 0！特征提取可能失效！")
            current_logits = raw_score[0].detach().cpu().numpy()
            print(f"🧠 Output Logits: {current_logits}")
            print(f"   -> Max - Min Diff: {current_logits.max() - current_logits.min():.4f}")
            print("="*20 + "\n")    
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
        bin_state_dim: int = 5,
        device: Union[str, int, torch.device] = "cuda:0",
        **kwargs
    ) -> None:
        super().__init__()
        
        self.device = device
        
        # Critic-specific layers
        self.num_bins = num_bins
        self.bin_state_dim = bin_state_dim
        self.item_feature_dim = 3 # L,W,H
        self.global_feature_dim = 4 # util_std, util_mean, itemcnt_std, itemcnt_mean
        self.input_dim = self.bin_state_dim + self.item_feature_dim  # bin features + item features + global features
        
        # 1. Shared Encoder (特征提取器)
        # 作用：把每个 Bin 的原始数据映射为高维语义向量
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_sizes[0]),
            nn.LayerNorm(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.LayerNorm(hidden_sizes[1]),
            nn.ReLU()
        )
        
        # 2. Value Head (评分器)
        # 输入：Pooling 后的特征 (hidden_dim) + 显式的 Global 特征 (4)
        # 建议把 global features 再次拼接触入，强化 Critic 对 Load Balance 的感知
        pooled_dim = hidden_sizes[1] * 3
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(pooled_dim + self.global_feature_dim, hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)
        )
        nn.init.uniform_(self.value_head[-1].weight, -0.01, 0.01)
        nn.init.constant_(self.value_head[-1].bias, 0.0)
        
    def forward(self, obs: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        
        device = next(self.parameters()).device
        if obs.device != device:
            obs = obs.to(device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        mask, bin_feats, item_feats, global_feats = obs_processor(obs, self.num_bins, self.bin_state_dim)
        batch_size = obs.shape[0]
        N = self.num_bins
        
        # 2. 特征扩展与拼接 (与 Actor 一样)
        item_rep = item_feats.unsqueeze(1).expand(batch_size, N, -1)
        global_rep = global_feats.unsqueeze(1).expand(batch_size, N, -1)
        cat_feats = torch.cat([bin_feats, item_rep], dim=-1) # [Batch, N, 12]
        
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
        raw_counts = mask_expanded.sum(dim=1)
        valid_counts = raw_counts.clamp(min=1.0)
        # 求平均
        pooled_embedding = sum_embeddings / valid_counts
        
        # --- Option B: Max Pooling (如果只关心能否装得下) ---
        # fill_value = -1e9
        # masked_embeddings = bin_embeddings.masked_fill(mask_expanded == 0, fill_value)
        # pooled_embedding = masked_embeddings.max(dim=1)[0]
        masked_embeddings = bin_embeddings.clone()
        masked_embeddings[~mask.bool()] = -20.0  # 将无效 bin 设为极小值 / Set invalid bins to very small
        max_pool = masked_embeddings.max(dim=1)[0]  # [B, hidden]
        is_all_empty = (raw_counts == 0) # [Batch, 1]

        max_pool = torch.where(is_all_empty, torch.zeros_like(max_pool), max_pool)
        squared_diff = (bin_embeddings - pooled_embedding.unsqueeze(1)) ** 2
        masked_squared_diff = squared_diff * mask_expanded
        variance = masked_squared_diff.sum(dim=1) / valid_counts
        std_pool = torch.sqrt(variance + 1e-8)  # [B, hidden]
        
        # 拼接三种池化结果 / Concatenate three pooling results
        pooled = torch.cat([pooled_embedding, max_pool, std_pool], dim=-1)  # [B, 3*hidden]
        
        # 加入全局特征 / Add global features
        final_input = torch.cat([pooled, global_feats], dim=-1)
        
        # 输出 Value / Output Value
        state_value = self.value_head(final_input)  # [B, 1]
        
        return state_value