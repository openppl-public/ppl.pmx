from dataclasses import dataclass
from typing import Optional

@dataclass
class BertParams:
    hidden_dim: int = 512
    intermediate_dim: int = 2048

    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: Optional[int] = None

    norm_eps: float = 1e-5
    vocab_size: int = -1

    position_embedding_type: str = 'absolute'
    type_vocab_size: int = 2
    max_position_embeddings: int=2048
