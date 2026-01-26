from dataclasses import dataclass

@dataclass(frozen=True)#prevents accidental mutation
class ModelConfig:
    vocab_size: int
    n_layers: int
    n_heads: int
    d_model: int
    block_size: int
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
    
    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"