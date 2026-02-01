import torch
import torch.nn as nn

from model.config import ModelConfig
from model.attention import CachedCausalSelfAttention
from model.layers import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CachedCausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor, kv_cache) -> torch.Tensor:
        """
        x: (B, 1, D)
        kv_cache: KVCache for this layer
        """
        x = x + self.attn(self.ln1(x), kv_cache)
        x = x + self.mlp(self.ln2(x))
        return x
