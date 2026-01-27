import torch
import torch.nn as nn

from model.config import ModelConfig


class TokenEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (B, T)
        returns:   (B, T, D)
        """
        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.block_size,
            embedding_dim=config.d_model
        )

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        position_ids: (T) or (B, T)
        returns:      (B, T, D)
        """
        return self.embedding(position_ids)
