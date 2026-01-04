import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
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
        self.bin_encoder = nn.Sequential(
            nn.Linear(bin_state_dim, hidden_sizes[0]),
            nn.LayerNorm(hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1])
        )
        self.item_encoder = nn.Sequential(
            nn.Linear(self.item_feature_dim, hidden_sizes[0]),
            nn.LayerNorm(hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1])
        )
        self.score_nn = nn.Sequential(
            nn.Linear(hidden_sizes[1] * 2, hidden_sizes[1]),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(hidden_sizes[1], hidden_sizes[1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1] // 2, 1)
        ) # bilinear 
        # TODO change to 0.0 later DONE
        self.log_temp = nn.Parameter(torch.tensor(0.0)) # use smaller temp for initial exploration try -1, 0.3
        self._init_weights()
        for module in [self.bin_encoder[-1], self.item_encoder[-1]]:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
        self._current_dist = None
        self._after_inference = False

    def _init_weights(self):
        """
        专门针对 Cosine Similarity 架构的初始化策略
        """
        # 1. 第一轮遍历：通用初始化 (针对所有中间层)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 正交初始化有助于保持特征的独立性
                # gain=sqrt(2) 是为了配合 ReLU
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            
            elif isinstance(m, nn.LayerNorm):
                # LayerNorm 标准初始化
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)

        # 2. 第二轮遍历：修正输出层 (Embedding Heads)
        # 这一步至关重要！覆盖掉上面的通用初始化
        for encoder in [self.bin_encoder, self.item_encoder]:
            # 获取该 encoder 的最后一层 (假设是 Sequential)
            # 注意：如果你的 Encoder 结尾是 LayerNorm，请往前回溯找到最后一个 Linear
            last_module = list(encoder.modules())[-1] 
            
            # 如果结尾是 Sequential，我们要找里面的最后一个 Linear
            if not isinstance(last_module, nn.Linear):
                for sub_m in reversed(list(encoder.modules())):
                    if isinstance(sub_m, nn.Linear):
                        last_module = sub_m
                        break
            
            # 对 Embedding 输出层进行重置
            if isinstance(last_module, nn.Linear):
                print(f"Re-initializing output layer: {last_module}")
                # Gain = 1.0 (因为后面没有 ReLU，直接进 Normalize)
                nn.init.orthogonal_(last_module.weight, gain=1.0)
                # Bias = 0.0 (极其重要：确保向量中心在原点)
                if last_module.bias is not None:
                    nn.init.constant_(last_module.bias, 0.0)

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
        
        bin_embeddings = self.bin_encoder(bin_features)  # (batch_size, num_bins, hidden_size)
        item_embeddings = self.item_encoder(item_features).unsqueeze(1)  # (batch_size, hidden_size)
        item_embeddings = item_embeddings.expand(-1, self.num_bins, -1)  # (batch_size, num_bins, hidden_size)
        combined = torch.cat([bin_embeddings, item_embeddings], dim=-1)
        # raw_score = self.score_nn(combined).squeeze(-1)  # (batch_size, num_bins)
        bin_embeddings = F.normalize(bin_embeddings, dim=-1)
        item_embeddings = F.normalize(item_embeddings, dim=-1)
        raw_score = torch.einsum("bnd,bnd->bn", bin_embeddings, item_embeddings)
        
        temp = torch.clamp(self.log_temp, max=2.5).exp()
        scaled_score = raw_score * temp
        
        import random
        if random.random() < 0.001:
            print("="*20)
            print(f"input obs :{bin_features}")
            # print(f"embeddings { bin_embeddings}, {item_embeddings.squeeze(-1)}")
            current_logits = raw_score[0].detach().cpu().numpy()
            print(f"🧠 Output Logits: {current_logits}")
            print(f"   -> Max - Min Diff: {current_logits.max() - current_logits.min():.4f}")
            print(f"scaled score:{scaled_score[0].detach().cpu().numpy()}")
            print("="*20)    
        # Apply mask
        if mask is not None:
            bool_mask = ~mask
            all_masked = bool_mask.all(dim=-1, keepdim=True)
            if all_masked.any():
                bool_mask = torch.where(all_masked, torch.tensor(False, device=device), bool_mask)

            scaled_score = scaled_score.masked_fill(bool_mask, -20.0)
        
        return Categorical(logits=scaled_score)
    
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
        self.bin_encoder = nn.Sequential(
            nn.Linear(bin_state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1])
        )
        self.item_encoder = nn.Sequential(
            nn.Linear(self.item_feature_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1])
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_feature_dim, hidden_sizes[0]//2),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0]//2, hidden_sizes[1]//2)
        )
        self.similarity_nn = nn.Sequential(
            nn.Linear(hidden_sizes[1] * 2, hidden_sizes[1]), 
            nn.Tanh(),                  # 第一道防线：非线性激活
            nn.Linear(hidden_sizes[1], 1) # 映射到 1 个分数
        )
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_sizes[1] + hidden_sizes[1]//2, hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[1]//2),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1]//2, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        # 4. 关键修改：不要用 zeros_，改用小的 gain
        # 这样 Critic 会输出一个随机的小数值（如 0.01, -0.02），而不是死锁在 0
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        nn.init.constant_(self.value_head[-1].bias, 0.0)
        # nn.init.zeros_(self.value_head[-1].weight)
        # nn.init.constant_(self.value_head[-1].bias, 0.0) # TODO change to reward? not sure whether to change the initialization for weight
        
    def forward(self, obs: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        
        device = next(self.parameters()).device
        if obs.device != device:
            obs = obs.to(device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        mask, bin_feats, item_feats, global_feats = obs_processor(obs, self.num_bins, self.bin_state_dim)
        N = self.num_bins
        
        bin_embeddings = self.bin_encoder(bin_feats)  # (batch_size, num_bins, hidden_size)
        
        item_embeddings = self.item_encoder(item_feats).unsqueeze(1)  # (batch_size, 1, hidden_size)
        item_exp = item_embeddings.expand(-1, N, -1)  # (batch_size, num_bins, hidden_size)
        
        global_embeddings = self.global_encoder(global_feats)  # (batch_size
        
        # similarity_scores = self.similarity_nn(bin_embeddings, item_exp).squeeze(-1)  # (batch_size, num_bins)
        cat_features = torch.cat([bin_embeddings, item_exp], dim=-1) # (B, N, 2H)
        
        # 2. 通过 MLP 计算分数 (代替 Bilinear)
        # 形状变化: (B, N, 2H) -> (B, N, H) -> (B, N, 1) -> (B, N)
        similarity_scores = self.similarity_nn(cat_features).squeeze(-1)
        if mask is not None:
            mask = mask.bool()
            all_masked = (~mask).all(dim=-1, keepdim=True)
            if all_masked.any():
                mask = mask.masked_fill(all_masked, False) # if all bins are masked, unmask all to avoid NaN
            similarity_scores = similarity_scores.masked_fill(~mask, -1e8)  # (batch_size, num_bins)
        attn_weights = F.softmax(similarity_scores, dim=-1)
        weighted_bin_emb = (attn_weights.unsqueeze(-1) * bin_embeddings).sum(dim=1)  # (batch_size, hidden_size)
        if all_masked is not None and all_masked.any(): # only consider global feature when all bins are masked
            weighted_bin_emb = torch.where(
                all_masked.expand(-1, weighted_bin_emb.size(-1)),
                torch.zeros_like(weighted_bin_emb),
                weighted_bin_emb
            )
            weighted_bin_emb = weighted_bin_emb.masked_fill(all_masked, 0.0)
        combined_features = torch.cat([weighted_bin_emb, global_embeddings], dim=-1)  # (batch_size, num_bins + hidden_size//2)
        
        state_value = self.value_head(combined_features)  # (batch_size,)
        return state_value.view(-1, 1) # expect (batch_size, 1) in omni safe