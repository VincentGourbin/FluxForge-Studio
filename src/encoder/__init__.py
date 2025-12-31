"""
Encoder Module

This module provides text encoding functionality for FLUX.2 generation.
Supports multiple encoding backends:
- Local pipeline (default)
- Remote HuggingFace endpoint
- Local Mistral (PyTorch/Transformers)

FLUX.2-dev uses Mistral hidden states from layers 10, 20, 30 combined
into a (batch, seq, 15360) tensor.

Reference:
- https://huggingface.co/spaces/multimodalart/mistral-text-encoder
"""

# Primary encoder: Transformers-based (works on all platforms)
from .mistral_encoder import (
    is_mistral_available,
    encode_prompt_mistral,
    get_mistral_encoder,
    unload_mistral_encoder,
    get_encoder_status,
    MISTRAL_MODEL_ID,
    MISTRAL_FP8_MODEL_ID,
    HIDDEN_STATE_LAYERS
)

# Legacy MLX encoder (Apple Silicon only, experimental)
try:
    from .mistral_mlx_encoder import (
        is_mlx_available,
        encode_prompt_mlx,
        get_mlx_encoder,
        unload_mlx_encoder as unload_mlx,
        MLX_MODEL_ID
    )
except ImportError:
    is_mlx_available = lambda: False
    encode_prompt_mlx = None
    get_mlx_encoder = None
    unload_mlx = None
    MLX_MODEL_ID = None

__all__ = [
    # Primary Mistral encoder (Transformers)
    'is_mistral_available',
    'encode_prompt_mistral',
    'get_mistral_encoder',
    'unload_mistral_encoder',
    'get_encoder_status',
    'MISTRAL_MODEL_ID',
    'MISTRAL_FP8_MODEL_ID',
    'HIDDEN_STATE_LAYERS',
    # Legacy MLX encoder
    'is_mlx_available',
    'encode_prompt_mlx',
    'get_mlx_encoder',
    'unload_mlx',
    'MLX_MODEL_ID'
]
