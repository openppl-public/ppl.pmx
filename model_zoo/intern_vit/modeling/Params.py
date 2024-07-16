from dataclasses import dataclass
from typing import Optional

@dataclass
class ViTParams:
    hidden_dim: int = 512
    intermediate_dim: int = 2048

    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: Optional[int] = None

    norm_eps: float = 1e-5

    image_size: int = 224
    patch_size: int = 32
    projection_dim: int = 512
    
    llm_hidden_dim: int = 512
    downsample_ratio: int = 2