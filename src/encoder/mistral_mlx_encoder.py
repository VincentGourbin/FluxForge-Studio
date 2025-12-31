"""Mistral MLX Local Text Encoder for FLUX.2.

This module provides local text encoding using Mistral via MLX framework,
enabling offline operation and potential memory savings by separating
the text encoding step from the diffusion process.

FLUX.2-dev was trained to use Mistral embeddings from hidden states layers
(10, 20, 30), not raw token embeddings. This creates a (batch, seq, 15360)
tensor that FLUX.2 expects.

Benefits:
- Offline operation (no internet required)
- Lower latency (local inference)
- Memory optimization (encoder loaded separately)
- Apple Silicon optimized (Metal acceleration)

Requirements:
- pip install mlx-lm
- Apple Silicon Mac (M1/M2/M3/M4)

Model: mlx-community/Mistral-Small-3.2-24B-Instruct-2506-8bit (~24GB)
License: Apache 2.0

Reference:
- https://huggingface.co/spaces/multimodalart/mistral-text-encoder
"""

import gc
from typing import Optional, Tuple, Any, List, Union

# Lazy-loaded encoder components
_mlx_model = None
_mlx_tokenizer = None
_mlx_available = None

# Model configuration
MLX_MODEL_ID = "mlx-community/Mistral-Small-3.2-24B-Instruct-2506-8bit"

# Hidden state layers used by FLUX.2 for Mistral embeddings
HIDDEN_STATE_LAYERS = (10, 20, 30)
MAX_SEQUENCE_LENGTH = 512


def is_mlx_available() -> bool:
    """Check if MLX framework is available (Apple Silicon only).

    Returns:
        bool: True if MLX is installed and available, False otherwise
    """
    global _mlx_available

    if _mlx_available is not None:
        return _mlx_available

    try:
        import mlx
        import mlx.core as mx
        import mlx_lm

        # Check if we're on Apple Silicon
        _mlx_available = True
        print("✅ MLX framework available")
        return True

    except ImportError as e:
        _mlx_available = False
        print(f"⚠️ MLX not available: {e}")
        print("   Install with: pip install mlx-lm")
        return False


def get_mlx_encoder(model_id: Optional[str] = None) -> Tuple[Any, Any]:
    """Get or create MLX Mistral encoder (singleton pattern).

    Args:
        model_id: Optional model ID to load (defaults to MLX_MODEL_ID)

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        RuntimeError: If MLX is not available
    """
    global _mlx_model, _mlx_tokenizer

    if not is_mlx_available():
        raise RuntimeError(
            "MLX is not available. This feature requires:\n"
            "1. Apple Silicon Mac (M1/M2/M3/M4)\n"
            "2. pip install mlx-lm"
        )

    if _mlx_model is None:
        from mlx_lm import load

        model_to_load = model_id or MLX_MODEL_ID
        print(f"🔄 Loading MLX Mistral encoder: {model_to_load}")
        print("   This may take a few minutes on first run...")

        _mlx_model, _mlx_tokenizer = load(model_to_load)
        print("✅ MLX Mistral encoder loaded successfully")

    return _mlx_model, _mlx_tokenizer


def format_prompt_for_mistral(prompt: str, system_message: str = None) -> List[dict]:
    """Format prompt in Mistral chat format.

    Args:
        prompt: User prompt to format
        system_message: Optional system message

    Returns:
        List of message dicts in Mistral format
    """
    if system_message is None:
        system_message = (
            "You are an AI that reasons about image descriptions. "
            "You give structured responses focusing on object relationships, "
            "object attribution and actions without speculation."
        )

    # Remove any [IMG] tokens that could cause issues
    cleaned_prompt = prompt.replace("[IMG]", "")

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": cleaned_prompt}
    ]


def encode_prompt_mlx(
    prompt: str,
    device: str = "mps",
    max_length: int = MAX_SEQUENCE_LENGTH
) -> Optional[Any]:
    """Encode prompt using MLX Mistral model.

    This function extracts hidden states from layers 10, 20, 30 of the
    Mistral model and combines them into prompt_embeds compatible with FLUX.2.

    Note: MLX models loaded via mlx_lm don't natively support output_hidden_states.
    This implementation uses the available embedding layer as a fallback,
    but for full compatibility, you may need to use the Transformers-based
    remote encoder or a custom MLX implementation.

    Args:
        prompt: Text prompt to encode
        device: Target device for output tensor ("mps" or "cpu")
        max_length: Maximum token length

    Returns:
        Tensor of prompt embeddings, or None if encoding fails

    Raises:
        RuntimeError: If MLX is not available
    """
    import torch
    import numpy as np

    try:
        import mlx.core as mx

        model, tokenizer = get_mlx_encoder()

        # Format prompt for Mistral instruction format
        # Check if chat template is available
        formatted_prompt = None

        try:
            if hasattr(tokenizer, 'apply_chat_template') and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                messages = format_prompt_for_mistral(prompt)
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    tokenize=False
                )
        except Exception as e:
            print(f"⚠️ Chat template not available: {e}")

        # Fallback: use Mistral instruction format directly
        if formatted_prompt is None:
            system_msg = (
                "You are an AI that reasons about image descriptions. "
                "You give structured responses focusing on object relationships, "
                "object attribution and actions without speculation."
            )
            formatted_prompt = f"[INST] {system_msg}\n\n{prompt} [/INST]"

        # Tokenize
        tokens = tokenizer.encode(formatted_prompt)

        # Truncate if needed
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
            print(f"⚠️ Prompt truncated to {max_length} tokens")

        # Convert to MLX array
        input_ids = mx.array([tokens])

        print(f"📊 Token count: {len(tokens)}")

        # Try to get hidden states from the model
        # MLX models structure: model['language_model']['model']
        lm_model = None
        if 'language_model' in model:
            lm = model['language_model']
            if 'model' in lm:
                lm_model = lm['model']

        if lm_model is None:
            print("⚠️ Could not access Mistral language model structure")
            return None

        # Get embeddings from embedding layer
        if not hasattr(lm_model, 'embed_tokens'):
            print("⚠️ Could not find embed_tokens layer")
            return None

        # Get initial embeddings
        embeddings = lm_model.embed_tokens(input_ids)
        mx.eval(embeddings)

        embed_dim = embeddings.shape[-1]
        print(f"📊 Embedding dimension: {embed_dim}")

        # Note: Full hidden state extraction requires running the full model
        # For now, we use the embedding layer output
        # FLUX.2 expects hidden states from layers 10, 20, 30 stacked together

        # Check if we can run the full model to get hidden states
        if hasattr(lm_model, 'layers') and len(lm_model.layers) >= max(HIDDEN_STATE_LAYERS):
            print(f"🔄 Running forward pass through {len(lm_model.layers)} layers...")
            print(f"   Extracting hidden states from layers: {HIDDEN_STATE_LAYERS}")

            try:
                # Run through transformer layers
                hidden = embeddings

                hidden_states_to_collect = []
                for i, layer in enumerate(lm_model.layers):
                    hidden = layer(hidden)
                    if isinstance(hidden, tuple):
                        hidden = hidden[0]  # Get just the hidden states

                    # Collect hidden states from target layers
                    if i in HIDDEN_STATE_LAYERS:
                        mx.eval(hidden)
                        hidden_states_to_collect.append(hidden)
                        print(f"   Collected layer {i}: shape={hidden.shape}")

                if len(hidden_states_to_collect) == len(HIDDEN_STATE_LAYERS):
                    # Stack hidden states: (batch, num_layers, seq, hidden)
                    stacked = mx.stack(hidden_states_to_collect, axis=1)
                    mx.eval(stacked)

                    # Reshape to (batch, seq, num_layers * hidden)
                    batch_size, num_channels, seq_len, hidden_dim = stacked.shape
                    # Transpose and reshape
                    stacked = mx.transpose(stacked, (0, 2, 1, 3))
                    prompt_embeds = mx.reshape(stacked, (batch_size, seq_len, num_channels * hidden_dim))

                    # Convert to float32 for numpy compatibility (bfloat16 not supported)
                    prompt_embeds = prompt_embeds.astype(mx.float32)
                    mx.eval(prompt_embeds)

                    print(f"✅ Combined hidden states: shape={prompt_embeds.shape}")

                    # Convert to PyTorch via numpy
                    embeds_np = np.array(prompt_embeds)
                    embeds_torch = torch.from_numpy(embeds_np)

                    # Convert to bfloat16 and move to target device
                    embeds_torch = embeds_torch.to(torch.bfloat16)

                    if device == "mps" and torch.backends.mps.is_available():
                        embeds_torch = embeds_torch.to("mps")
                    elif device == "cuda" and torch.cuda.is_available():
                        embeds_torch = embeds_torch.to("cuda")

                    print(f"✅ MLX encoding complete: shape={embeds_torch.shape}, dtype={embeds_torch.dtype}")
                    return embeds_torch

            except Exception as e:
                print(f"⚠️ Full model forward pass failed: {e}")
                import traceback
                traceback.print_exc()

        # Fallback: model doesn't support full forward pass extraction
        print("⚠️ MLX model doesn't support hidden state extraction")
        print("   Falling back to pipeline encoder")
        return None

    except Exception as e:
        print(f"❌ MLX encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_embedding_dim() -> int:
    """Get the embedding dimension of the loaded model.

    For FLUX.2 with Mistral, the combined dimension is 3 * 5120 = 15360
    (3 layers of 5120-dim hidden states).

    Returns:
        int: Embedding dimension
    """
    if _mlx_model is None:
        return 15360  # 3 layers * 5120 hidden dim

    try:
        # Get hidden dim from model
        if 'language_model' in _mlx_model:
            lm = _mlx_model['language_model']
            if 'model' in lm and hasattr(lm['model'], 'embed_tokens'):
                import mlx.core as mx
                sample_tokens = mx.array([[1]])
                sample_embed = lm['model'].embed_tokens(sample_tokens)
                mx.eval(sample_embed)
                hidden_dim = sample_embed.shape[-1]
                return len(HIDDEN_STATE_LAYERS) * hidden_dim

        return 15360
    except Exception:
        return 15360


def unload_mlx_encoder():
    """Unload MLX encoder to free memory.

    Call this when switching to pipeline encoder or when memory is needed.
    """
    global _mlx_model, _mlx_tokenizer

    if _mlx_model is not None:
        print("🧹 Unloading MLX encoder...")

        del _mlx_model
        del _mlx_tokenizer
        _mlx_model = None
        _mlx_tokenizer = None

        # Force garbage collection
        gc.collect()

        # Try to clear MLX cache if available
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass

        print("✅ MLX encoder unloaded")


def get_encoder_status() -> dict:
    """Get current status of the MLX encoder.

    Returns:
        dict: Status information including availability, loaded state, model ID
    """
    return {
        "mlx_available": is_mlx_available(),
        "encoder_loaded": _mlx_model is not None,
        "model_id": MLX_MODEL_ID,
        "hidden_state_layers": HIDDEN_STATE_LAYERS,
        "embedding_dim": get_embedding_dim() if _mlx_model else None
    }
