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
        # self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)
        conv_output_size = self._get_conv_output_size()
        self.fc1 = nn.Linear(conv_output_size, 128)
        # self.bn_fc = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, embedding_dim)
        
        self.dropout = nn.Dropout(0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _get_conv_output_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, *self.hmap_size)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            return x.view(1, -1).size(1)

    def forward(self, hmaps):
        batch_size, num_bins, L, W = hmaps.shape
        x = hmaps.reshape(-1, 1, L, W)
        x = self.pool(F.relu((self.conv1(x))))
        x = self.pool(F.relu((self.conv2(x))))
        x = F.relu((self.conv3(x)))
        
        x = x.view(batch_size * num_bins, -1)
        x = F.relu((self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        embeddings = x.view(batch_size, num_bins, -1)
        
        return embeddings


class BinContextAggregator(nn.Module):
    """Simple aggregator for bin context using weighted averaging"""
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.importance_net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.Tanh(),
            nn.Linear(embedding_dim // 2, 1)
        )
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, bin_embeddings, mask=None):
        """
        Args:
            bin_embeddings: (batch_size, num_bins, embedding_dim)
            mask: (batch_size, num_bins) - True for valid bins
        Returns:
            context: (batch_size, embedding_dim)
        """
        importance_scores = self.importance_net(bin_embeddings).squeeze(-1)
        
        if mask is not None:
            importance_scores = importance_scores.masked_fill(~mask, -1e9)
        
        attention_weights = F.softmax(importance_scores, dim=-1).unsqueeze(-1)
        context = (bin_embeddings * attention_weights).sum(dim=1)
        context = self.layer_norm(context)
        
        return context

class CategoricalActor(nn.Module):

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
        
        # Feature encoders with normalization
        self.bin_encoder = binCNN(hmap_size=(bin_size[1], bin_size[2]), embedding_dim=hidden_sizes[1])
        self.bin_norm = nn.LayerNorm(hidden_sizes[1])
        
        self.item_encoder = nn.Sequential(
            nn.Linear(self.item_feature_dim, hidden_sizes[0]),
            nn.LayerNorm(hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.LayerNorm(hidden_sizes[1])
        )
        
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_feature_dim, hidden_sizes[0]//2),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0]//2, hidden_sizes[1]//2),
            nn.LayerNorm(hidden_sizes[1]//2)
        )
        
        # Bin context aggregator
        self.bin_context_aggregator = BinContextAggregator(self.bin_embedding_dim)
        
        # Attention mechanism
        context_dim = self.bin_embedding_dim + hidden_sizes[1] + hidden_sizes[1]//2
        
        # Query and key projections (linear transformations, following standard attention)
        # Removing LeakyReLU to improve gradient flow and stability
        self.query_net = nn.Sequential(
            nn.Linear(context_dim, self.bin_embedding_dim),
            nn.LayerNorm(self.bin_embedding_dim)
        )

        self.key_net = nn.Sequential(
            nn.Linear(self.bin_embedding_dim, self.bin_embedding_dim),
            nn.LayerNorm(self.bin_embedding_dim)
        )
        
        # Learnable bias for each bin
        self.bin_bias = nn.Parameter(torch.zeros(num_bins))

        # Fixed temperature for attention scaling (following standard Transformer practice)
        # Using sqrt(d_k) as per "Attention is All You Need" paper
        self.temperature = float(np.sqrt(self.bin_embedding_dim))  # Fixed at 8.0 for embedding_dim=64
        
        # State management for OmniSafe's call pattern
        self._current_dist = None
        self._after_inference = False
        
        # Diagnostics flag
        self._enable_diagnostics = False
        
    def _distribution(self, obs: torch.Tensor) -> Categorical:
        """Compute action distribution from observations.
        
        This method encodes observations through the attention mechanism
        and produces logits for bin selection. It is called by forward()
        and predict() to establish the distribution state.
        
        Args:
            obs: Observation tensor from environment
            
        Returns:
            Categorical distribution over bins
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        
        device = next(self.parameters()).device
        if obs.device != device:
            obs = obs.to(device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # Parse observation into components
        mask, bin_features, item_features, global_features = obs_processor(
            obs, self.num_bins, self.bin_state_dim
        )
        mask = mask.bool()
        
        # Encode features through neural networks
        bin_embeddings = self.bin_encoder(bin_features)
        bin_embeddings = self.bin_norm(bin_embeddings)
        
        item_embeddings = self.item_encoder(item_features)
        global_embeddings = self.global_encoder(global_features)
        
        # Aggregate bin context using attention pooling
        bin_context = self.bin_context_aggregator(bin_embeddings, mask)
        
        # Construct attention query and keys
        context = torch.cat([bin_context, item_embeddings, global_embeddings], dim=-1)
        query = self.query_net(context).unsqueeze(1)
        keys = self.key_net(bin_embeddings)
        
        # Compute scaled dot-product attention scores
        # Using fixed temperature for stable training
        attn_scores = torch.matmul(query, keys.transpose(-2, -1)).squeeze(1) / self.temperature
        
        # Add learnable per-bin bias
        attn_scores = attn_scores + self.bin_bias.unsqueeze(0)

        # CRITICAL: Apply mask BEFORE clipping to avoid extreme value disparities
        # This prevents KL divergence explosion caused by masked values being far outside valid range
        if mask is not None:
            bool_mask = ~mask
            # Handle edge case where all bins are masked
            all_masked = bool_mask.all(dim=-1, keepdim=True)
            if all_masked.any():
                print("⚠️ Warning: All bins masked - this should not happen in valid states")
                bool_mask = torch.where(
                    all_masked,
                    torch.tensor(False, device=device),
                    bool_mask
                )
            # Use -1e9 instead of -1e8 for better numerical stability
            attn_scores = attn_scores.masked_fill(bool_mask, -1e9)

        # Clip for numerical stability AFTER masking
        attn_scores = torch.clamp(attn_scores, min=-100, max=100.0)

        # Diagnostics logging
        if self._enable_diagnostics:
            self._log_diagnostics(attn_scores, mask)
        
        return Categorical(logits=attn_scores)
    
    def forward(self, obs: torch.Tensor) -> Categorical:
        self._current_dist = self._distribution(obs)
        self._after_inference = True
        return self._current_dist
    
    def predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        self._current_dist = self._distribution(obs)
        self._after_inference = True
        
        if deterministic:
            action = self._current_dist.probs.argmax(-1)
        else:
            action = self._current_dist.sample()
        
        return action.squeeze(-1) if action.dim() > 1 else action
    
    def log_prob(self, act):
        """Compute log probability of actions.

        CRITICAL: This method uses the distribution from the most recent forward() or predict() call.
        In PPO, you MUST call forward() immediately before log_prob() to ensure the distribution
        corresponds to the current policy, not a stale cached version.

        Args:
            act: Action tensor to compute log probability for

        Returns:
            Log probability of the actions under the current distribution
        """
        assert self._after_inference, "Must call forward() or predict() before log_prob()"
        assert self._current_dist is not None, "Distribution not initialized"

        # Reset state after computing log_prob
        self._after_inference = False

        # Ensure action has correct shape (squeeze if needed)
        if act.dim() > 1:
            act = act.squeeze(-1)

        # Verify shapes match to prevent broadcasting errors
        if act.shape != self._current_dist.batch_shape:
            raise ValueError(
                f"Action shape {act.shape} does not match distribution batch shape "
                f"{self._current_dist.batch_shape}. This would cause incorrect broadcasting "
                f"and lead to KL divergence explosion."
            )

        return self._current_dist.log_prob(act.long())
    
    def _log_diagnostics(self, logits: torch.Tensor, mask: torch.Tensor):
        """Log diagnostic information about logits and distribution properties.
        
        This method provides insights into the policy's behavior during training,
        including logit statistics, entropy, and learned parameters. It samples
        calls to avoid excessive logging overhead.
        """
        import random
        if random.random() < 0.05:  # Sample 5% of calls
            with torch.no_grad():
                if mask is not None:
                    valid_logits = logits[mask.bool()]
                    if valid_logits.numel() > 0:
                        logit_std = valid_logits.std().item()
                        logit_range = (valid_logits.max() - valid_logits.min()).item()
                        logit_mean = valid_logits.mean().item()
                    else:
                        logit_std = logit_range = logit_mean = 0.0
                else:
                    logit_std = logits.std().item()
                    logit_range = (logits.max() - logits.min()).item()
                    logit_mean = logits.mean().item()

                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
                max_prob = probs.max(dim=-1)[0].mean().item()

                print(f"🔍 [Actor Diagnostics]")
                print(f"    Logits - Mean: {logit_mean:.3f} | Std: {logit_std:.3f} | Range: {logit_range:.3f}")
                print(f"    Distribution - Entropy: {entropy:.3f} | Max Prob: {max_prob:.3f}")
                print(f"    Temperature (fixed): {self.temperature:.3f}")
                print(f"    Bin Bias: [{', '.join([f'{b:.3f}' for b in self.bin_bias.data.cpu().numpy()])}]")
    
    def enable_diagnostics(self, enable: bool = True):
        self._enable_diagnostics = enable
    
    @property
    def std(self):
        return torch.zeros(1)
    
    @std.setter
    def std(self, std):
        pass
class BSCritic(nn.Module):
    """Simplified Critic with proper initialization"""
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
        self.num_bins = num_bins
        self.bin_state_dim = bin_state_dim
        self.item_feature_dim = 3
        self.global_feature_dim = 4
        
        # Encoders with normalization
        self.bin_encoder = binCNN(hmap_size=(10, 10), embedding_dim=hidden_sizes[1])
        self.bin_norm = nn.LayerNorm(hidden_sizes[1])
        
        self.item_encoder = nn.Sequential(
            nn.Linear(self.item_feature_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]//2),
            nn.LayerNorm(hidden_sizes[1]//2)
        )
        
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_feature_dim, hidden_sizes[0]//2),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0]//2, hidden_sizes[1]//2),
            nn.LayerNorm(hidden_sizes[1]//2)
        )
        
        # Statistical aggregation: mean, std, max, min
        stat_features_dim = hidden_sizes[1] * 4
        value_input_dim = stat_features_dim + hidden_sizes[1]//2 + hidden_sizes[1]//2
        
        # Simple value networks
        self.net_list: list[nn.Module] = []
        for idx in range(self._num_critics):
            value_head = nn.Sequential(
                nn.Linear(value_input_dim, hidden_sizes[1]),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_sizes[1], hidden_sizes[1]//2),
                nn.ReLU(),
                nn.Linear(hidden_sizes[1]//2, 1)
            )
            self.net_list.append(value_head)
            self.add_module(f'critic_{idx}', value_head)
        
        self._init_weights()
        self._log_counter = 0

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        # Standard initialization for final layer (gain=1.0, not 0.01)
        for critic in self.net_list:
            nn.init.orthogonal_(critic[-1].weight, gain=1.0)
            nn.init.constant_(critic[-1].bias, 0.0)
    
    def forward(self, obs: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        import json
        log_path = '/home/qxt0570/code/MultiGOPT/.cursor/debug3.log'
        import pathlib
        log_dir = pathlib.Path(log_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        should_log = self._log_counter % 50 == 0
        self._log_counter += 1
        
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        
        device = next(self.parameters()).device
        if obs.device != device:
            obs = obs.to(device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        mask, bin_feats, item_feats, global_feats = obs_processor(obs, self.num_bins, self.bin_state_dim)
        
        # Encode features
        bin_embeddings = self.bin_encoder(bin_feats)
        bin_embeddings = self.bin_norm(bin_embeddings)
        
        if should_log:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'C',
                    'location': 'simplified_critic:forward',
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
        
        # Statistical aggregation
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            masked_embeddings = bin_embeddings * mask_expanded
            
            valid_count = mask_expanded.sum(dim=1).clamp(min=1.0)
            bin_mean = masked_embeddings.sum(dim=1) / valid_count
            
            bin_centered = masked_embeddings - bin_mean.unsqueeze(1) * mask_expanded
            bin_var = (bin_centered ** 2).sum(dim=1) / valid_count
            bin_std = torch.sqrt(bin_var + 1e-8)
            
            bin_max = masked_embeddings.max(dim=1)[0]
            
            large_value = 1e6
            mask = mask.bool()
            bin_min_input = masked_embeddings + (~mask.unsqueeze(-1)) * large_value
            bin_min = bin_min_input.min(dim=1)[0]
            all_masked = (mask_expanded.sum(dim=1) == 0).float()
            bin_min = bin_min * (1 - all_masked) + bin_mean * all_masked
        else:
            bin_mean = bin_embeddings.mean(dim=1)
            bin_std = bin_embeddings.std(dim=1)
            bin_max = bin_embeddings.max(dim=1)[0]
            bin_min = bin_embeddings.min(dim=1)[0]
        
        bin_stat_features = torch.cat([bin_mean, bin_std, bin_max, bin_min], dim=-1)
        
        item_embeddings = self.item_encoder(item_feats)
        global_embeddings = self.global_encoder(global_feats)
        
        combined_features = torch.cat([bin_stat_features, item_embeddings, global_embeddings], dim=-1)
        
        res = []
        for critic in self.net_list:
            value = critic(combined_features)
            res.append(torch.squeeze(value, -1))
        
        return res