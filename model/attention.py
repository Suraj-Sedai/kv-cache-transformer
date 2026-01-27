import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        q, k, v: (B, T, D)
        mask:    (T, T) or (B, T, T)
        return:  (B, T, D)
        """

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)

        return output
