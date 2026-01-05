import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Any, Tuple, Union, Sequence
import time

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
        # Scoring methods: Choose one that explicitly models bin-item relationships
        # Option 1: Bilinear transformation - explicit relationship modeling with larger range
        # Score = bin_emb^T @ W @ item_emb, where W is learnable matrix
        self.bilinear_weight = nn.Parameter(torch.randn(hidden_sizes[1], hidden_sizes[1]))
        
        # Option 2: Element-wise interaction + MLP - explicit interaction then processing
        # First compute element-wise product (explicit interaction), then MLP
        self.interaction_nn = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[1]),
            nn.LayerNorm(hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1] // 2, 1)
        )
        
        # Option 3: Learnable scale for cosine similarity
        # Instead of fixed 5.0, use learnable scale parameter
        self.log_scale = nn.Parameter(torch.tensor(np.log(5.0)))  # Initial scale = 5.0
        
        # Scoring method selection: 'bilinear', 'interaction', 'cosine', or 'cosine_scaled'
        # FIXED: Default to 'interaction' for better stability (bilinear can cause large KL)
        # 'interaction' provides explicit relationship modeling with better initialization control
        self.scoring_method = kwargs.get('scoring_method', 'interaction')  # Default: interaction (more stable)
        
        # Temperature parameter with better initialization and constraints
        # log_temp=0.0 gives temp=1.0, which is a good starting point
        # We clamp it to prevent extreme values during training
        self.log_temp = nn.Parameter(torch.tensor(0.0))  # temp=1.0 initially
        self._init_weights()
        for module in [self.bin_encoder[-1], self.item_encoder[-1]]:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
        self._current_dist = None
        self._after_inference = False
        self._enable_diagnostics = False  # Disable by default for performance

    def _init_weights(self):
        """
        Improved weight initialization for stable training
        """
        # 1. First pass: General initialization for all layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Orthogonal initialization helps maintain feature independence
                # gain=sqrt(2) is for ReLU activation
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            
            elif isinstance(m, nn.LayerNorm):
                # Standard LayerNorm initialization
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)

        # 2. Second pass: Special initialization for output layers
        # For encoder output layers (if using cosine similarity)
        for encoder in [self.bin_encoder, self.item_encoder]:
            last_module = list(encoder.modules())[-1]
            if not isinstance(last_module, nn.Linear):
                for sub_m in reversed(list(encoder.modules())):
                    if isinstance(sub_m, nn.Linear):
                        last_module = sub_m
                        break
            
            if isinstance(last_module, nn.Linear):
                # Smaller gain for embedding layers to prevent extreme values
                nn.init.orthogonal_(last_module.weight, gain=1.0)
                if last_module.bias is not None:
                    nn.init.constant_(last_module.bias, 0.0)
        
        # 3. Special initialization for scoring layers
        # Bilinear weight: Use smaller gain to prevent initial logits from being too large
        # Smaller gain (0.1-0.2) helps prevent KL divergence explosion at the start
        nn.init.xavier_uniform_(self.bilinear_weight, gain=0.1)  # Reduced from 1.0 to 0.1
        
        # Interaction MLP output layer: smaller gain for stability
        if hasattr(self.interaction_nn, '__iter__'):
            for module in reversed(list(self.interaction_nn.modules())):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
                    break

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
        item_embeddings = self.item_encoder(item_features).unsqueeze(1)  # (batch_size, 1, hidden_size)
        item_embeddings = item_embeddings.expand(-1, self.num_bins, -1)  # (batch_size, num_bins, hidden_size)
        
        # Explicit relationship modeling with larger range
        if self.scoring_method == 'bilinear':
            # Bilinear transformation: bin^T @ W @ item
            # Explicitly models relationship, no range restriction
            # For each (bin, item) pair: compute bin^T @ W @ item
            # Efficient einsum: (B, N, H) @ (H, H) with (B, N, H) -> (B, N)
            raw_score = torch.einsum('bnh,ho,bnh->bn', bin_embeddings, self.bilinear_weight, item_embeddings)
            
        elif self.scoring_method == 'interaction':
            # Element-wise product (explicit interaction) + MLP
            # First compute interaction, then process with MLP
            interaction = bin_embeddings * item_embeddings  # (B, N, H) - explicit element-wise interaction
            raw_score = self.interaction_nn(interaction).squeeze(-1)  # (B, N)
            
        elif self.scoring_method == 'cosine_scaled':
            # Cosine similarity with learnable scale (instead of fixed 5.0)
            bin_emb_norm = F.normalize(bin_embeddings, dim=-1)
            item_emb_norm = F.normalize(item_embeddings, dim=-1)
            cosine_sim = torch.einsum("bnd,bnd->bn", bin_emb_norm, item_emb_norm)  # [-1, 1]
            scale = self.log_scale.exp()  # Learnable scale, initialized to 5.0
            raw_score = cosine_sim * scale  # Range: [-scale, scale]
            
        else:  # 'cosine' (default fallback)
            # Standard cosine similarity with fixed scale
            bin_emb_norm = F.normalize(bin_embeddings, dim=-1)
            item_emb_norm = F.normalize(item_embeddings, dim=-1)
            raw_score = torch.einsum("bnd,bnd->bn", bin_emb_norm, item_emb_norm) * 5.0
        
        # Temperature scaling with better constraints
        # Clamp log_temp to prevent extreme temperatures that can destabilize training
        # Higher temperature = more exploration, lower temperature = more exploitation
        temp = torch.clamp(self.log_temp, min=-2.0, max=2.0).exp()  # temp in [0.135, 7.39]
        scaled_score = raw_score / temp  # Divide by temp: higher temp makes distribution more uniform
        
        # Systematized diagnostics (can be enabled/disabled)
        if hasattr(self, '_enable_diagnostics') and self._enable_diagnostics:
            self._log_diagnostics(raw_score, scaled_score, temp, mask)    
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

    def _log_diagnostics(self, raw_score, scaled_score, temp, mask):
        """Systematized diagnostic logging for debugging"""
        import random
        if random.random() < 0.01:  # Sample 1% of calls
            batch_idx = 0
            raw_score_np = raw_score[batch_idx].detach().cpu().numpy()
            scaled_score_np = scaled_score[batch_idx].detach().cpu().numpy()
            
            # Compute policy statistics
            probs = F.softmax(scaled_score[batch_idx:batch_idx+1], dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            max_prob = probs.max().item()
            
            # Compute valid scores (unmasked)
            if mask is not None:
                valid_mask = ~mask[batch_idx].bool()
                if valid_mask.any():
                    valid_raw = raw_score_np[valid_mask.cpu().numpy()]
                    valid_scaled = scaled_score_np[valid_mask.cpu().numpy()]
                else:
                    valid_raw = raw_score_np
                    valid_scaled = scaled_score_np
            else:
                valid_raw = raw_score_np
                valid_scaled = scaled_score_np
            
            print("="*50)
            print(f"🔍 Actor Diagnostics (Batch {batch_idx}):")
            print(f"  Scoring method: {self.scoring_method}")
            if self.scoring_method == 'cosine_scaled':
                scale = self.log_scale.exp().item()
                print(f"  Learnable scale: {scale:.4f} (log_scale={self.log_scale.item():.4f})")
            print(f"  Raw score: min={valid_raw.min():.4f}, max={valid_raw.max():.4f}, "
                  f"mean={valid_raw.mean():.4f}, std={valid_raw.std():.4f}")
            print(f"  Temperature: {temp.item():.4f} (log_temp={self.log_temp.item():.4f})")
            print(f"  Scaled score: min={valid_scaled.min():.4f}, max={valid_scaled.max():.4f}, "
                  f"mean={valid_scaled.mean():.4f}, std={valid_scaled.std():.4f}")
            print(f"  Policy entropy: {entropy.item():.4f}, Max prob: {max_prob:.4f}")
            if mask is not None:
                num_valid = valid_mask.sum().item() if valid_mask.any() else len(valid_raw)
                print(f"  Valid bins: {num_valid}/{self.num_bins}")
            print("="*50)
    
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
        # Value head - Enhanced with multiple aggregation strategies
        # Use both attention-weighted and max-pooled bin embeddings for better discrimination
        # Input: [weighted_bin_emb (H), max_pooled_bin_emb (H), global_emb (H//2)]
        value_input_dim = hidden_sizes[1] * 2 + hidden_sizes[1]//2
        self.net_list: list[nn.Module] = []
        for idx in range(self._num_critics):
            value_head = nn.Sequential(
            nn.Linear(value_input_dim, hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[1]//2),
            nn.ReLU(),
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
        log_path = '/home/qxt0570/code/GOPT/.cursor/debug1.log'
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
        
        item_embeddings = self.item_encoder(item_feats).unsqueeze(1)  # (batch_size, 1, hidden_size)
        item_exp = item_embeddings.expand(-1, N, -1)  # (batch_size, num_bins, hidden_size)
        
        # #region agent log
        if should_log:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'C',
                    'location': 'categorical_actor.py:282',
                    'message': 'item_embeddings_stats',
                    'data': {
                        'mean': float(item_embeddings.mean().item()),
                        'std': float(item_embeddings.std().item())
                    },
                    'timestamp': int(time.time() * 1000)
                }) + '\n')
        # #endregion
        
        global_embeddings = self.global_encoder(global_feats)  # (batch_size
        
        # #region agent log
        if should_log:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'C',
                    'location': 'categorical_actor.py:285',
                    'message': 'global_embeddings_stats',
                    'data': {
                        'mean': float(global_embeddings.mean().item()),
                        'std': float(global_embeddings.std().item())
                    },
                    'timestamp': int(time.time() * 1000)
                }) + '\n')
        # #endregion
        
        # similarity_scores = self.similarity_nn(bin_embeddings, item_exp).squeeze(-1)  # (batch_size, num_bins)
        cat_features = torch.cat([bin_embeddings, item_exp], dim=-1) # (B, N, 2H)
        
        # 2. 通过 MLP 计算分数 (代替 Bilinear)
        # 形状变化: (B, N, 2H) -> (B, N, H) -> (B, N, 1) -> (B, N)
        similarity_scores = self.similarity_nn(cat_features).squeeze(-1)
        
        # #region agent log
        if should_log:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'F',
                    'location': 'categorical_actor.py:289',
                    'message': 'similarity_scores_stats',
                    'data': {
                        'mean': float(similarity_scores.mean().item()),
                        'std': float(similarity_scores.std().item()),
                        'min': float(similarity_scores.min().item()),
                        'max': float(similarity_scores.max().item()),
                        'std_per_sample': float(similarity_scores.std(dim=1).mean().item())
                    },
                    'timestamp': int(time.time() * 1000)
                }) + '\n')
        # #endregion
        
        if mask is not None:
            mask = mask.bool()
            all_masked = (~mask).all(dim=-1, keepdim=True)
            if all_masked.any():
                mask = mask.masked_fill(all_masked, False) # if all bins are masked, unmask all to avoid NaN
            similarity_scores = similarity_scores.masked_fill(~mask, -1e8)  # (batch_size, num_bins)
        attn_weights = F.softmax(similarity_scores, dim=-1)
        
        # #region agent log
        if should_log:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'B',
                    'location': 'categorical_actor.py:296',
                    'message': 'attn_weights_stats',
                    'data': {
                        'mean': float(attn_weights.mean().item()),
                        'std': float(attn_weights.std().item()),
                        'entropy': float((-attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=1).mean().item()),
                        'max_weight_per_sample': float(attn_weights.max(dim=1)[0].mean().item())
                    },
                    'timestamp': int(time.time() * 1000)
                }) + '\n')
        # #endregion
        
        weighted_bin_emb = (attn_weights.unsqueeze(-1) * bin_embeddings).sum(dim=1)  # (batch_size, hidden_size)
        
        # Add max pooling as an alternative aggregation to capture different information
        # This helps when attention is uniform - max pooling captures the "best" bin
        max_pooled_bin_emb = bin_embeddings.max(dim=1)[0]  # (batch_size, hidden_size)
        
        # #region agent log
        if should_log:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'B',
                    'location': 'categorical_actor.py:297',
                    'message': 'weighted_bin_emb_stats',
                    'data': {
                        'mean': float(weighted_bin_emb.mean().item()),
                        'std': float(weighted_bin_emb.std().item()),
                        'std_across_batch': float(weighted_bin_emb.std(dim=0).mean().item()) if weighted_bin_emb.size(0) > 1 else 0.0
                    },
                    'timestamp': int(time.time() * 1000)
                }) + '\n')
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'B',
                    'location': 'categorical_actor.py:298',
                    'message': 'max_pooled_bin_emb_stats',
                    'data': {
                        'mean': float(max_pooled_bin_emb.mean().item()),
                        'std': float(max_pooled_bin_emb.std().item()),
                        'std_across_batch': float(max_pooled_bin_emb.std(dim=0).mean().item()) if max_pooled_bin_emb.size(0) > 1 else 0.0
                    },
                    'timestamp': int(time.time() * 1000)
                }) + '\n')
        # #endregion
        
        if all_masked is not None and all_masked.any(): # only consider global feature when all bins are masked
            weighted_bin_emb = torch.where(
                all_masked.expand(-1, weighted_bin_emb.size(-1)),
                torch.zeros_like(weighted_bin_emb),
                weighted_bin_emb
            )
            weighted_bin_emb = weighted_bin_emb.masked_fill(all_masked, 0.0)
            max_pooled_bin_emb = torch.where(
                all_masked.expand(-1, max_pooled_bin_emb.size(-1)),
                torch.zeros_like(max_pooled_bin_emb),
                max_pooled_bin_emb
            )
            max_pooled_bin_emb = max_pooled_bin_emb.masked_fill(all_masked, 0.0)
        
        # Combine: attention-weighted, max-pooled, and global features
        # This gives the value_head more diverse information to distinguish states
        combined_features = torch.cat([weighted_bin_emb, max_pooled_bin_emb, global_embeddings], dim=-1)  # (batch_size, 2*H + H//2)
        res = []
        for critic in self.net_list:
            value = critic(combined_features)
            res.append(torch.squeeze(value, -1))
        
        return res # expect (batch_size, 1) in omni safe