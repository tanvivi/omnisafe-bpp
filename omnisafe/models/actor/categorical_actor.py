import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Any, Tuple, Union, Sequence
import time

def obs_processor(obs, num_bins, bin_feature_dim=100):
    """convert flat observation to structured format
    current obs is [mask, bin_features, item_features, global_features]
    mask: (num_bins,)
    bin_features: (num_bins, 5)"""
    mask = obs[:, :num_bins]
    bin_start = num_bins
    bin_end = num_bins + num_bins * bin_feature_dim
    hmap_L = int(np.sqrt(bin_feature_dim))
    hmap_W = hmap_L
    bin_features = obs[:, bin_start:bin_end].reshape(-1, num_bins, hmap_L, hmap_W)
    item_start = bin_end
    item_end = bin_end + 3
    item_features = obs[:, item_start:item_end]
    global_features = obs[:, item_end:]
    return mask, bin_features, item_features, global_features

class binCNN(nn.Module):
    def __init__(self, hmap_size=(10,10), embedding_dim=64):
        super().__init__()
        self.hmap_size = hmap_size
        self.embedding_dim = embedding_dim
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        conv_output_size = self._get_conv_output_size()
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, embedding_dim)
        
        self.dropout = nn.Dropout(0.2)
    
    def _get_conv_output_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, *self.hmap_size)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            return x.view(1, -1).size(1)

    def forward(self, hmaps):
        batch_size, num_bins, L, W = hmaps.shape
        x = hmaps.reshape(-1, 1, L, W)  # (batch_size * num_bins, 1, L, W
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        
        x = x.view(batch_size * num_bins, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        embeddings = x.view(batch_size, num_bins, -1)
        
        return embeddings

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
        bin_state_dim: int = 5,
        bin_size: list = [10, 10, 10],
        device: Union[str, int, torch.device] = "cuda:0",
        bin_embedding_dim=64,
        **kwargs
    ) -> None:
        super().__init__()
        
        self._act_dim = num_bins
        self.num_bins = num_bins
        self.device = device
        self.bin_state_dim = bin_state_dim
        self.item_feature_dim = 3
        self.global_feature_dim = 4
        self.bin_embedding_dim = hidden_sizes[1]
        # input_dim =  self.bin_state_dim  + self.item_feature_dim  # global features + bin features + item features + bin context features
        self.bin_encoder = binCNN(hmap_size=(bin_size[1], bin_size[2]), embedding_dim=hidden_sizes[1])
        self.item_encoder = nn.Sequential(
            nn.Linear(self.item_feature_dim, hidden_sizes[0]),
            nn.LayerNorm(hidden_sizes[0]),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1])
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_feature_dim, hidden_sizes[0]//2),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[0]//2, hidden_sizes[1]//2),
        )
        context_dim = hidden_sizes[1] + hidden_sizes[1]//2 # item embedding dim + global embedding dim
        self.query_proj = nn.Linear(context_dim, self.bin_embedding_dim)
        self.key_proj = nn.Linear(self.bin_embedding_dim, self.bin_embedding_dim)
        self.value_proj = nn.Linear(self.bin_embedding_dim, self.bin_embedding_dim)
        
        self.policy_head = nn.Sequential(
            nn.Linear(bin_embedding_dim + context_dim, hidden_sizes[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[1], 1)    
            )

        self._current_dist = None
        self._after_inference = False
        self._enable_diagnostics = False  # Disable by default for performance
        
    def _distribution(self, obs: torch.Tensor):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        
        device = next(self.parameters()).device
        if obs.device != device:
            obs = obs.to(device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if torch.isnan(obs).any():
            print("💀 [Fatal] Input Observation contains NaN!")
            # 打印出有问题的部分，帮助定位是 Env 哪里算错了
            print(f"NaN indices: {torch.nonzero(torch.isnan(obs))}")
            raise ValueError("Input NaN detected")
        batch_size = obs.shape[0]
        
        # Parse mask
        mask, bin_features, item_features, global_features = obs_processor(obs, self.num_bins, self.bin_state_dim)
        mask = mask.bool()
        # mask_f= mask.float()
        
        bin_embeddings = self.bin_encoder(bin_features)  # (batch_size, num_bins, hidden_size)
        if torch.isnan(bin_embeddings).any():
            print("💀 [Fatal] Bin Encoder output contains NaN! (Weights might be broken)")
            # 检查权重是否已坏
            for name, param in self.bin_encoder.named_parameters():
                if torch.isnan(param).any():
                    print(f" -> Found NaN in weights: {name}")
            raise ValueError("Encoder NaN detected")
        item_embeddings = self.item_encoder(item_features).unsqueeze(1)  # (batch_size, 1, hidden_size)
        global_embeddings = self.global_encoder(global_features).unsqueeze(1)  # (batch_size, hidden_size//2)
        context = torch.cat([item_embeddings, global_embeddings], dim=-1)
        query = self.query_proj(context)  # (batch_size, num_bins, bin_embedding_dim)
        keys = self.key_proj(bin_embeddings)      # (batch_size, num_bins, bin_embedding_dim)
        
        attn_scores = torch.matmul(query, keys.transpose(-2, -1)) / np.sqrt(self.bin_embedding_dim)  # (batch_size, num_bins, num_bins)
        if torch.isnan(attn_scores).any():
             print("💀 [Fatal] Attention Scores contain NaN (Before Masking)!")
             print(f"Query max: {query.max()}, Key max: {keys.max()}")
             raise ValueError("Attention NaN detected")
        # Systematized diagnostics (can be enabled/disabled)
        if hasattr(self, '_enable_diagnostics') and self._enable_diagnostics:
            self._log_diagnostics(attn_scores, mask)
        # Apply mask
        if mask is not None:
            bool_mask = ~mask
            all_masked = bool_mask.all(dim=-1, keepdim=True)
            if all_masked.any():
                print("⚠️ Warning: All-masked state detected in Actor. Unmasking to avoid crash.")
                bool_mask = torch.where(all_masked, torch.tensor(False, device=device), bool_mask)

            attn_scores = attn_scores.masked_fill(bool_mask, -1e4)
        
        return Categorical(logits=attn_scores)
    
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

    def _log_diagnostics(self, logits, mask):
        """
        logits: (Batch, N) - 未经过 Softmax 的分数
        mask: (Batch, N) - 有效掩码
        """
        import random
        if random.random() < 0.01:  # Sample 1% of calls
        
            with torch.no_grad():
                # 1. 基础统计量 (Logits Variance)
                # 只统计 mask 为 True 的部分，避免被 -1e8 干扰
                if mask is not None:
                    # 这种写法稍微粗糙，但能看个大概
                    valid_logits = logits[mask.bool()] 
                    logit_std = valid_logits.std().item()
                    logit_range = (valid_logits.max() - valid_logits.min()).item()
                else:
                    logit_std = logits.std().item()
                    logit_range = (logits.max() - logits.min()).item()

                # 2. 概率分布统计 (Entropy & Confidence)
                probs = F.softmax(logits, dim=-1) # (B, N)
                
                # 计算熵: -sum(p * log(p))
                # 加上 1e-8 防止 log(0)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
                
                # 计算最大置信度 (Max Probability)
                max_prob = probs.max(dim=-1)[0].mean().item()
                
                # 3. 打印或 Log
                print(f"🔍 [Actor Check] Logit Std: {logit_std:.4f} | Range: {logit_range:.4f} | "
                    f"Entropy: {entropy:.4f} | MaxProb: {max_prob:.4f}")
                
                # 4. 警报
                if logit_std < 0.01:
                    print("⚠️ Warning: Attention Collapse (Logits are identical)")
                if max_prob < (1.0 / logits.size(-1)) + 0.1:
                    print("⚠️ Warning: Actor is guessing randomly")
    
    def enable_diagnostics(self, enable=True):
        """Enable/disable diagnostic logging"""
        self._enable_diagnostics = enable
    
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
        num_critics: int = 1,
        device: Union[str, int, torch.device] = "cuda:0",
        **kwargs
    ) -> None:
        super().__init__()
        
        self.device = device
        
        self._num_critics = num_critics
        
        # Critic-specific layers
        self.num_bins = num_bins
        self.bin_state_dim = bin_state_dim
        self.item_feature_dim = 3 # L,W,H
        self.global_feature_dim = 4 # util_std, util_mean, itemcnt_std, itemcnt_mean
        self.input_dim = self.bin_state_dim + self.item_feature_dim  # bin features + item features + global features
        
        # 1. Shared Encoder (特征提取器)
        # 作用：把每个 Bin 的原始数据映射为高维语义向量
        # self.bin_proj = nn.Linear(hidden_sizes[1], hidden_sizes[1])
        # self.item_proj = nn.Linear(hidden_sizes[1], hidden_sizes[1])
        self.bin_encoder = binCNN(hmap_size=(10, 10), embedding_dim=hidden_sizes[1])
        self.item_encoder = nn.Sequential(
            nn.Linear(self.item_feature_dim, hidden_sizes[0]),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]//2)
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_feature_dim, hidden_sizes[0]//2),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[0]//2, hidden_sizes[1]//2),
            nn.LayerNorm(hidden_sizes[1]//2)
        )
        
        # self.bin_aggregator = nn.Sequential(
        #     nn.Linear(hidden_sizes[1], hidden_sizes[1]),
        #     nn.ReLU()
        # ) #TODO：first don't use aggregator
        
        value_input_dim = hidden_sizes[1] + hidden_sizes[1]//2 + hidden_sizes[1]//2
        self.net_list: list[nn.Module] = []
        for idx in range(self._num_critics):
            value_head = nn.Sequential(
            nn.Linear(value_input_dim, hidden_sizes[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[1]//2),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[1]//2, 1)
        )
            self.net_list.append(value_head)
            self.add_module(f'critic_{idx}', value_head)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        for critic in self.net_list:
            nn.init.orthogonal_(critic[-1].weight, gain=1.0)
            nn.init.constant_(critic[-1].bias, 0.0)
    
    def forward(self, obs: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        # #region agent log
        import json
        import os
        log_path = '/home/qxt0570/code/GOPT/.cursor/debug3.log'
        should_log = hasattr(self, '_log_counter') and self._log_counter % 50 == 0
        if not hasattr(self, '_log_counter'):
            self._log_counter = 0
        self._log_counter += 1
        # #endregion
        
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
        
        # #region agent log
        if should_log:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'C',
                    'location': 'categorical_actor.py:277',
                    'message': 'bin_embeddings_stats',
                    'data': {
                        'mean': float(bin_embeddings.mean().item()),
                        'std': float(bin_embeddings.std().item()),
                        'min': float(bin_embeddings.min().item()),
                        'max': float(bin_embeddings.max().item()),
                        'std_across_bins': float(bin_embeddings.std(dim=1).mean().item())
                    },
                    'timestamp': int(time.time() * 1000)
                }) + '\n')
        # #endregion
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float() # (B, N, 1)
            bin_embeddings = bin_embeddings * mask_expanded # 把无效 Bin 置 0
            
            # Sum Pooling
            bin_sum = bin_embeddings.sum(dim=1) # (B, H)
            
            # Count valid bins (clamp to avoid div by zero)
            mask_count = mask_expanded.sum(dim=1).clamp(min=1.0)
            
            # Mean Pooling
            bin_pooled = bin_sum / mask_count # (B, H)
        else:
            bin_pooled = bin_embeddings.mean(dim=1) # (B, H)
        
        
        item_embeddings = self.item_encoder(item_feats)  # (batch_size, 1, hidden_size)
        # item_exp = item_embeddings.expand(-1, N, -1)  # (batch_size, num_bins, hidden_size)
        
        global_embeddings = self.global_encoder(global_feats)  # (batch_size, hidden_size//2)
        
        
        combined_features = torch.cat([bin_pooled, item_embeddings, global_embeddings], dim=-1)  # (batch_size, 2*H + H//2)
        res = []
        for critic in self.net_list:
            value = critic(combined_features)
            res.append(torch.squeeze(value, -1))
        
        return res # expect (batch_size, 1) in omni safe