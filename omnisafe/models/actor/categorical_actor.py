import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from typing import Any, Tuple, Union, Sequence


class Attention(nn.Module):
    def __init__(self, embed_size, heads):
        super(Attention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        # linear layers for query, key, value and final output
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, query, keys, values, pad_mask=None):
        # A.P.: Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        #A.P.: Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # A.P.: (N, value_len, heads, head_dim)
        keys = self.keys(keys)        # A.P.: (N, key_len, heads, head_dim)
        queries = self.queries(query) # A.P.: (N, query_len, heads, heads_dim)

        # A.P.: Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm
        # calculate the energy scores
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # A.P.: queries shape: (N, query_len, heads, heads_dim),
        # A.P.: keys shape: (N, key_len, heads, heads_dim)
        # A.P.: energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if pad_mask is not None:
            # pad_mask = pad_mask.unsqueeze(-1).expand(N, query_len, key_len)
            # pad_mask = pad_mask.unsqueeze(1).repeat(1, self.heads, 1, 1)
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(1)  # (N, 1, 1, key_len)
            energy = energy.masked_fill(pad_mask==0, -1e18)
            # energy = energy.masked_fill(pad_mask==0, float("-inf"))

        # A.P.: Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        # 得到每个query的输出
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # A.P.: attention shape: (N, heads, query_len, key_len)
        # 拼接多个head的输出
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # A.P.: attention shape: (N, heads, query_len, key_len)
        # A.P.: values shape: (N, value_len, heads, heads_dim)
        # A.P.: out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.
        # 映射回原始维度
        out = self.fc_out(out)
        # A.P.: Linear layer doesn't modify the shape, final shape will be (N, query_len, embed_size)

        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__() # ？残差连接 + layernorm
        # general transformer block with multi-head attention and feed forward network
        self.attention = Attention(embed_size, heads) # multi-head attention for query-key-value
        self.norm1 = nn.LayerNorm(embed_size) # used after attention 
        self.norm2 = nn.LayerNorm(embed_size) # used after feed forward network

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        ) # two-layer feed forward network

        self.dropout = nn.Dropout(dropout) # regularization, avoid overfitting

    def forward(self, query, key, value, pad_mask=None):
        attention = self.attention(query, key, value, pad_mask)

        # A.P.: Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
class BinSelectionEncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(BinSelectionEncoderBlock, self).__init__()
        
        self.item_self_attn = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.bin_self_attn = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.bin_on_item = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.item_on_bin = TransformerBlock(embed_size, heads, dropout, forward_expansion)

    def forward(self, item_feature, bin_feature, mask=None):
        item_embedding = self.item_self_attn(item_feature, item_feature, item_feature)
        bin_embedding = self.bin_self_attn(bin_feature, bin_feature, bin_feature, mask)
        bin_on_item = self.bin_on_item(bin_embedding, item_embedding, item_embedding)
        item_on_bin = self.item_on_bin(item_embedding, bin_embedding, bin_embedding, mask)
        return item_on_bin, bin_on_item

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

init_relu_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('leaky_relu'))
init_val_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), 1.0)
init_pol_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), 0.01) # 关键！




class ShareNet(nn.Module):
    """Shared feature extractor for bins and items"""
    def __init__(
        self,
        num_bins: int = 5,
        bin_size: Sequence[int] = [10, 10, 10],
        bin_feature_dim: int = 5,
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
        self.bin_feature_dim = bin_feature_dim
        self.hmap_size = bin_size[0] * bin_size[1]
        
        # Observation structure
        self.mask_size = num_bins
        self.bin_feature_size = num_bins * bin_feature_dim
        self.hmap_total_size = num_bins * self.hmap_size
        self.item_size = 3
        
        self.item_encoder = nn.Sequential(
            init_relu_(nn.Linear(3, 64)),
            # nn.LayerNorm(64),
            nn.LeakyReLU(),
            init_relu_(nn.Linear(64, embed_size)),
            # nn.LayerNorm(embed_size),
        )
        
        # Heightmap encoder
        self.heightmap_encoder = nn.Sequential(
                init_relu_(nn.Linear(self.hmap_size, 128)),
                # nn.LayerNorm(128),
                nn.LeakyReLU(),
                init_relu_(nn.Linear(128, embed_size)),
                # nn.LayerNorm(embed_size),
            )
        
        # Bin feature encoder
        self.bin_feature_encoder = nn.Sequential(
            init_relu_(nn.Linear(bin_feature_dim, 64)),
            # nn.LayerNorm(64),
            nn.LeakyReLU(),
            init_relu_(nn.Linear(64, embed_size)),
            # nn.LayerNorm(embed_size),
        )
        
        # self.bin_fusion = nn.Sequential(
        #     init_relu_(nn.Linear(64 + 32, embed_size)),
        #     nn.LeakyReLU()
        # )
        
        # Transformer backbone
        self.backbone = nn.ModuleList([
            BinSelectionEncoderBlock(
                embed_size=embed_size,
                heads=heads,
                dropout=dropout,
                forward_expansion=forward_expansion,
            )
            for _ in range(num_layers)
        ])
        
    def parse_observation(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parse flat observation into structured components"""
        batch_size = obs.shape[0]
        idx = 0
        mask = obs[:, idx:idx+self.mask_size]
        idx += self.mask_size
        
        bin_features_flat = obs[:, idx:idx+self.bin_feature_size]
        bin_features = bin_features_flat.reshape(-1, self.num_bins, self.bin_feature_dim)
        idx += self.bin_feature_size
        
        hmaps_flat = obs[:, idx:idx+self.hmap_total_size]
        # hmaps = hmaps_flat.reshape(-1, self.num_bins, int(np.sqrt(self.hmap_size)), int(np.sqrt(self.hmap_size)))
        hmaps = hmaps_flat.reshape(batch_size, self.num_bins, self.hmap_size)
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
            mask[(~mask).all(dim=1), 0] = True # fool the model if all bins are masked
        
        # Encode item
        item_embedding = self.item_encoder(item_features).unsqueeze(1)
        
        # Encode heightmaps
        hmap_embedding = self.heightmap_encoder(hmaps)
        
        # Encode bin features
        bin_encoded = self.bin_feature_encoder(bin_features)
        
        # Combine
        # bin_embedding = torch.cat([hmap_embedding, bin_encoded], dim=-1)
        # bin_embedding = self.bin_proj(bin_embedding)
        bin_embedding = bin_encoded + hmap_embedding  # Residual connection
        
        
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
        bin_feature_dim: int = 5,
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
                bin_feature_dim=bin_feature_dim,
                embed_size=embed_size,
                num_layers=num_layers,
                heads=heads,
                dropout=dropout,
                device=device
            )
        
        # Actor-specific layers
        self.layer_1 = nn.Sequential(
            init_pol_(nn.Linear(embed_size, embed_size)),
            # nn.LayerNorm(embed_size),
            nn.LeakyReLU(),
        )
        self.layer_2 = nn.Sequential(
            init_pol_(nn.Linear(embed_size, embed_size)),
            # nn.LayerNorm(embed_size),
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
        bin_feature_dim: int = 5,
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
                bin_feature_dim=bin_feature_dim,
                bin_size=bin_size,
                embed_size=embed_size,
                num_layers=num_layers,
                heads=heads,
                dropout=dropout,
                device=device
            )
        
        # Critic-specific layers
        self.layer_1 = nn.Sequential(
            init_val_(nn.Linear(embed_size, embed_size)),
            nn.LeakyReLU(),
        )
        self.layer_2 = nn.Sequential(
            init_val_(nn.Linear(embed_size, embed_size)),
            nn.LeakyReLU(),
        )
        self.layer_3 = nn.Sequential(
            init_val_(nn.Linear(2*embed_size, embed_size)),
            nn.LeakyReLU(),
            init_val_(nn.Linear(embed_size, embed_size)),
            nn.LeakyReLU(),
            init_val_(nn.Linear(embed_size, 1))
        )
        self.ln_backbone = nn.LayerNorm(embed_size)
        
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
        # joint_embedding = self.ln_backbone(item_embedding)
        state_value = self.layer_3(joint_embedding)
        return state_value


# Example usage with shared parameters
def test_model_shapes():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 1. 定义超参数（模拟你的环境）
    num_bins = 5
    bs = 2 # Batch size
    
    # 2. 实例化网络
    actor = CategoricalActor(
        obs_space=None, act_space=None, 
        num_bins=num_bins, 
        embed_size=32, heads=4, # 用小一点的参数方便看
        device=device
    ).to(device)
    
    critic = BSCritic(
        obs_space=None, act_space=None,
        num_bins=num_bins,
        embed_size=32, heads=4,
        share_net=actor.preprocess, # 共享特征提取层
        device=device
    ).to(device)

    # 3. 构造假数据 (Batch_size, Obs_dim)
    # 假设 obs 结构: [mask(5) | bin_feat(5*5) | hmap(5*100) | item(3)]
    # 计算总维度
    obs_dim = num_bins + (num_bins * 5) + (num_bins * 10 * 10) + 3
    fake_obs = torch.randn(bs, obs_dim).to(device)
    
    # 模拟 Mask: 假设第2个样本的最后两个 bin 是空的 (0)
    # Mask 在 obs 的前 num_bins 位
    fake_obs[1, 3:5] = 0 
    # 确保 mask 是 0/1 (bool)
    fake_obs[:, :num_bins] = (fake_obs[:, :num_bins] > 0).float()

    print(f"Testing with Input Shape: {fake_obs.shape}")

    # 4. 前向传播测试 (Forward Check)
    try:
        dist = actor(fake_obs)
        action = dist.sample()
        value = critic(fake_obs)
        print("✅ Forward pass successful")
        print(f"Action shape: {action.shape} (Expect: [{bs}])")
        print(f"Value shape: {value.shape} (Expect: [{bs}, 1])")
    except RuntimeError as e:
        print(f"❌ Forward pass failed: {e}")
        return

    # 5. Mask 逻辑验证 (关键！)
    # 检查被 Mask 掉的 bin 对应的 logit 是否被设为了极小值 (-1e18 或 -20)
    logits = dist.logits
    print("\nChecking Mask Logic:")
    print(f"Mask used in obs:\n{fake_obs[:, :num_bins]}")
    print(f"Logits output:\n{logits}")
    
    # 如果 mask 为 0 的位置，logits 不是非常小的负数，说明 Mask 没生效，Transformer 会注意到不存在的物体
    if logits[1, 4] > -10: 
        print("⚠️ WARNING: Masking might be failing! Invalid bin has high logit.")

    # 6. 反向传播测试 (Backward Check)
    # 很多手动写的 Transformer 会不小心 detach 掉梯度
    loss = value.mean() + dist.log_prob(action).mean()
    loss.backward()
    
    has_grad = False
    for name, param in actor.named_parameters():
        if param.grad is not None:
            has_grad = True
            # print(f"{name} has grad")
            break
            
    if has_grad:
        print("\n✅ Gradient flow successful")
    else:
        print("\n❌ Gradient flow failed (No grads found)")

if __name__ == "__main__":
    test_model_shapes()