from omnisafe.models.base import Actor
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical, Distribution
from omnisafe.utils.model import build_mlp_network
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Union, Sequence

def init_(m, gain=1.0):
    """Weight initialization helper"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    return m


class EncoderBlock(nn.Module):
    """Transformer encoder block for bin-item interaction"""
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, item_embed, bin_embed, mask=None):
        # Self-attention for bins
        attn_out, _ = self.attention(bin_embed, bin_embed, bin_embed, key_padding_mask=~mask if mask is not None else None)
        bin_embed = self.norm1(bin_embed + self.dropout(attn_out))
        ff_out = self.feed_forward(bin_embed)
        bin_embed = self.norm2(bin_embed + self.dropout(ff_out))
        
        # Cross-attention: item attends to bins
        item_attn, _ = self.attention(item_embed, bin_embed, bin_embed, key_padding_mask=~mask if mask is not None else None)
        item_embed = self.norm1(item_embed + self.dropout(item_attn))
        item_ff = self.feed_forward(item_embed)
        item_embed = self.norm2(item_embed + self.dropout(item_ff))
        
        return item_embed, bin_embed


class ShareNet(nn.Module):
    """Shared feature extractor for bins and items"""
    def __init__(
        self,
        num_bins: int = 5,
        bin_size: Sequence[int] = [10, 10, 10],
        embed_size: int = 128,
        num_layers: int = 3,
        forward_expansion: int = 4,
        heads: int = 8,
        dropout: float = 0.1,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        
        self.device = device
        self.num_bins = num_bins
        self.embed_size = embed_size
        self.hmap_size = bin_size[0] * bin_size[1]
        
        # Observation structure
        self.mask_size = num_bins
        self.bin_feature_size = num_bins * 5
        self.hmap_total_size = num_bins * self.hmap_size
        self.item_size = 3
        
        # Item encoder
        self.item_encoder = nn.Sequential(
            init_(nn.Linear(3, 64)),
            nn.LeakyReLU(),
            init_(nn.Linear(64, embed_size)),
        )
        
        # Heightmap encoder
        self.heightmap_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, embed_size // 2)
        )
        
        # Bin feature encoder
        self.bin_feature_encoder = nn.Sequential(
            init_(nn.Linear(5, embed_size // 2)),
            nn.LeakyReLU(),
        )
        
        self.bin_proj = init_(nn.Linear(embed_size, embed_size))
        
        # Transformer backbone
        self.backbone = nn.ModuleList([
            EncoderBlock(
                embed_size=embed_size,
                heads=heads,
                dropout=dropout,
                forward_expansion=forward_expansion,
            )
            for _ in range(num_layers)
        ])
        
    def parse_observation(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parse flat observation into structured components"""
        idx = 0
        mask = obs[:, idx:idx+self.mask_size]
        idx += self.mask_size
        
        bin_features_flat = obs[:, idx:idx+self.bin_feature_size]
        bin_features = bin_features_flat.reshape(-1, self.num_bins, 5)
        idx += self.bin_feature_size
        
        hmaps_flat = obs[:, idx:idx+self.hmap_total_size]
        hmaps = hmaps_flat.reshape(-1, self.num_bins, int(np.sqrt(self.hmap_size)), int(np.sqrt(self.hmap_size)))
        idx += self.hmap_total_size
        
        item_features = obs[:, idx:idx+self.item_size]
        
        return mask, bin_features, hmaps, item_features
        
    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        mask_input: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        
        mask, bin_features, hmaps, item_features = self.parse_observation(obs)
        
        if mask_input is not None:
            mask = mask_input
        
        mask = mask.bool()
        if (~mask).all(dim=1).any():
            mask[(~mask).all(dim=1), 0] = True
        batch_size = obs.shape[0]
        
        # Encode item
        item_embedding = self.item_encoder(item_features).unsqueeze(1)
        
        # Encode heightmaps
        hmaps_input = hmaps.unsqueeze(2)
        hmaps_flat = hmaps_input.reshape(-1, 1, hmaps.shape[2], hmaps.shape[3])
        hm_encoded = self.heightmap_encoder(hmaps_flat)
        hm_encoded = hm_encoded.view(batch_size, self.num_bins, -1)
        
        # Encode bin features
        bin_encoded = self.bin_feature_encoder(bin_features)
        
        # Combine
        bin_embedding = torch.cat([hm_encoded, bin_encoded], dim=-1)
        bin_embedding = self.bin_proj(bin_embedding)
        
        # Transformer
        for layer in self.backbone:
            item_embedding, bin_embedding = layer(item_embedding, bin_embedding, mask)
        
        return item_embedding, bin_embedding, state


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
        bin_size: list = [10, 10, 10],
        embed_size: int = 128,
        num_layers: int = 3,
        heads: int = 8,
        dropout: float = 0.1,
        padding_mask: bool = True,
        share_net: ShareNet = None,  # Accept shared network
        device: Union[str, int, torch.device] = "cuda:0",
        **kwargs
    ) -> None:
        super().__init__()
        
        self._act_dim = num_bins
        self.num_bins = num_bins
        self.padding_mask = padding_mask
        self.device = device
        
        # Use provided shared network or create new one
        if share_net is not None:
            self.preprocess = share_net
        else:
            self.preprocess = ShareNet(
                num_bins=num_bins,
                bin_size=bin_size,
                embed_size=embed_size,
                num_layers=num_layers,
                heads=heads,
                dropout=dropout,
                device=device
            )
        
        # Actor-specific layers
        self.layer_1 = nn.Sequential(
            init_(nn.Linear(embed_size, embed_size)),
            nn.LeakyReLU(),
        )
        self.layer_2 = nn.Sequential(
            init_(nn.Linear(embed_size, embed_size)),
            nn.LeakyReLU(),
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
        
        # Parse mask
        mask, _, _, _ = self.preprocess.parse_observation(obs)
        mask = mask.bool() if self.padding_mask else None
        
        # Extract features
        item_embedding, bin_embedding, _ = self.preprocess(obs, None, mask)
        
        # Process
        item_embedding = self.layer_1(item_embedding)
        bin_embedding = self.layer_2(bin_embedding).permute(0, 2, 1)
        
        # Compute logits
        logits = torch.bmm(item_embedding, bin_embedding).squeeze(1)
        
        # Apply mask
        if mask is not None:
            bool_mask = ~mask
            all_masked = bool_mask.all(dim=-1, keepdim=True)
            if all_masked.any():
                bool_mask = torch.where(all_masked, torch.tensor(False, device=device), bool_mask)

            logits = logits.masked_fill(bool_mask, -20.0)
        
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
        bin_size: list = [10, 10, 10],
        embed_size: int = 128,
        num_layers: int = 3,
        heads: int = 8,
        dropout: float = 0.1,
        padding_mask: bool = True,
        share_net: ShareNet = None,  # Accept shared network
        device: Union[str, int, torch.device] = "cuda:0",
        **kwargs
    ) -> None:
        super().__init__()
        
        self.num_bins = num_bins
        self.padding_mask = padding_mask
        self.device = device
        
        # Use provided shared network or create new one
        if share_net is not None:
            self.preprocess = share_net
        else:
            self.preprocess = ShareNet(
                num_bins=num_bins,
                bin_size=bin_size,
                embed_size=embed_size,
                num_layers=num_layers,
                heads=heads,
                dropout=dropout,
                device=device
            )
        
        # Critic-specific layers
        self.layer_1 = nn.Sequential(
            init_(nn.Linear(embed_size, embed_size)),
            nn.LeakyReLU(),
        )
        self.layer_2 = nn.Sequential(
            init_(nn.Linear(embed_size, embed_size)),
            nn.LeakyReLU(),
        )
        self.layer_3 = nn.Sequential(
            init_(nn.Linear(2 * embed_size, embed_size)),
            nn.LeakyReLU(),
            init_(nn.Linear(embed_size, embed_size)),
            nn.LeakyReLU(),
            init_(nn.Linear(embed_size, 1))
        )
        
    def forward(self, obs: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        
        device = next(self.parameters()).device
        if obs.device != device:
            obs = obs.to(device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # Parse mask
        mask, _, _, _ = self.preprocess.parse_observation(obs)
        mask = mask.bool() if self.padding_mask else None
        
        # Extract features
        item_embedding, bin_embedding, _ = self.preprocess(obs, None, mask)
        
        # Process
        item_embedding = self.layer_1(item_embedding)
        bin_embedding = self.layer_2(bin_embedding)
        
        # Aggregate
        item_embedding = item_embedding.squeeze(1)
        if mask is not None:
            bin_embedding = torch.sum(bin_embedding * mask.unsqueeze(-1), dim=1)
        else:
            bin_embedding = torch.sum(bin_embedding, dim=1)
        
        # Predict value
        joint_embedding = torch.cat([item_embedding, bin_embedding], dim=-1)
        state_value = self.layer_3(joint_embedding)
        return state_value


# Example usage with shared parameters
# if __name__ == "__main__":
    # from gym import spaces
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # num_bins = 5
    # bin_size = [10, 10, 10]
    # embed_size = 128
    
    # obs_dim = num_bins + num_bins * 5 + num_bins * 100 + 3
    # obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    # act_space = spaces.Discrete(num_bins)
    
    # # Create shared feature extractor
    # shared_net = ShareNet(
    #     num_bins=num_bins,
    #     bin_size=bin_size,
    #     embed_size=embed_size,
    #     num_layers=3,
    #     heads=8,
    #     device=device
    # )
    
    # # Initialize actor and critic with shared network
    # actor = CategoricalActor(
    #     obs_space=obs_space,
    #     act_space=act_space,
    #     num_bins=num_bins,
    #     bin_size=bin_size,
    #     embed_size=embed_size,
    #     share_net=shared_net,  # Share parameters
    #     device=device
    # )
    
    # critic = BSCritic(
    #     obs_space=obs_space,
    #     act_space=act_space,
    #     num_bins=num_bins,
    #     bin_size=bin_size,
    #     embed_size=embed_size,
    #     share_net=shared_net,  # Share parameters
    #     device=device
    # )
    
    # # Verify parameter sharing
    # print(f"✓ Shared parameters: {actor.preprocess is critic.preprocess}")
    
    # # Test
    # batch_size = 4
    # obs_flat = torch.randn(batch_size, obs_dim).to(device)
    # obs_flat[:, :num_bins] = torch.tensor([
    #     [1, 1, 1, 0, 0],
    #     [1, 1, 0, 0, 0],
    #     [1, 0, 0, 0, 0],
    #     [1, 1, 1, 1, 0]
    # ], dtype=torch.float32).to(device)
    
    # action = actor.predict(obs_flat)
    # dist = actor.forward(obs_flat)
    # log_prob = actor.log_prob(action)
    # value = critic(obs_flat)
    
    # print(f"\nActions: {action}")
    # print(f"Log probs: {log_prob}")
    # print(f"Values: {value}")