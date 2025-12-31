"""Mistral Local Text Encoder for FLUX.2.

This module provides local text encoding using Mistral via PyTorch/Transformers,
based on the official HuggingFace implementation for FLUX.2-dev.

FLUX.2-dev uses Mistral embeddings from hidden states layers (10, 20, 30),
creating a (batch, seq, 15360) tensor.

Reference:
- https://huggingface.co/spaces/multimodalart/mistral-text-encoder
- https://huggingface.co/spaces/black-forest-labs/FLUX.2-dev

Requirements:
- transformers >= 4.45.0
- torch

Model: mistralai/Mistral-Small-3.2-24B-Instruct-2506 (~48GB full, ~24GB 8-bit)
License: Apache 2.0
"""

import gc
import torch
from typing import Optional, Tuple, List, Union

# Lazy-loaded encoder components
_mistral_model = None
_mistral_tokenizer = None

# Model configuration
MISTRAL_MODEL_ID = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
MISTRAL_TOKENIZER_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
MISTRAL_FP8_MODEL_ID = "unsloth/Mistral-Small-3.2-24B-Instruct-2506-FP8"

# Hidden state layers used by FLUX.2 for Mistral embeddings
HIDDEN_STATE_LAYERS = (10, 20, 30)
MAX_SEQUENCE_LENGTH = 512

# System message for text encoding (same as HuggingFace)
SYSTEM_MESSAGE = """You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object attribution and actions without speculation."""


def is_mistral_available() -> bool:
    """Check if Mistral can be loaded via Transformers.

    Returns:
        bool: True if transformers is available
    """
    try:
        from transformers import Mistral3ForConditionalGeneration, AutoProcessor
        return True
    except ImportError as e:
        print(f"⚠️ Mistral encoder not available: {e}")
        return False


def get_mistral_encoder(
    model_id: Optional[str] = None,
    use_fp8: bool = False
) -> Tuple:
    """Get or create Mistral encoder (singleton pattern).

    Args:
        model_id: Optional model ID to load (overrides default)
        use_fp8: Whether to use FP8 quantized model from unsloth (saves memory)
                 Default: False (uses full precision model)

    Returns:
        Tuple of (model, tokenizer)
    """
    global _mistral_model, _mistral_tokenizer

    if _mistral_model is None:
        from transformers import Mistral3ForConditionalGeneration, AutoProcessor

        # Select model based on FP8 flag
        if use_fp8:
            model_to_load = MISTRAL_FP8_MODEL_ID
            print(f"🔄 Loading Mistral text encoder (FP8 quantized): {model_to_load}")
        else:
            model_to_load = model_id or MISTRAL_MODEL_ID
            print(f"🔄 Loading Mistral text encoder: {model_to_load}")

        tokenizer_id = MISTRAL_TOKENIZER_ID
        print("   This may take a few minutes on first run...")

        # Determine device and dtype
        if torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.bfloat16
        elif torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16
        else:
            device = "cpu"
            dtype = torch.float32

        print(f"   Device: {device}, dtype: {dtype}")

        # Load model
        load_kwargs = {
            "torch_dtype": dtype,
            "device_map": device,
        }

        _mistral_model = Mistral3ForConditionalGeneration.from_pretrained(
            model_to_load,
            **load_kwargs
        )

        # Load tokenizer
        _mistral_tokenizer = AutoProcessor.from_pretrained(tokenizer_id)

        print(f"✅ Mistral encoder loaded on {device}")

    return _mistral_model, _mistral_tokenizer


def format_text_input(prompts: List[str], system_message: str = None) -> List:
    """Format prompts in Mistral chat format.

    Args:
        prompts: List of text prompts
        system_message: System message for the model

    Returns:
        List of formatted message batches
    """
    if system_message is None:
        system_message = SYSTEM_MESSAGE

    # Remove [IMG] tokens to avoid Pixtral validation issues
    cleaned_txt = [prompt.replace("[IMG]", "") for prompt in prompts]

    return [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        for prompt in cleaned_txt
    ]


def get_mistral_prompt_embeds(
    text_encoder,
    tokenizer,
    prompt: Union[str, List[str]],
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
    hidden_states_layers: Tuple[int, ...] = HIDDEN_STATE_LAYERS,
) -> torch.Tensor:
    """Generate prompt embeddings from Mistral hidden states.

    This is the core encoding function that extracts hidden states from
    layers 10, 20, 30 and combines them into FLUX.2-compatible embeddings.

    Args:
        text_encoder: Mistral model
        tokenizer: Mistral tokenizer/processor
        prompt: Text prompt or list of prompts
        max_sequence_length: Maximum token length
        hidden_states_layers: Which layers to extract (default: 10, 20, 30)

    Returns:
        torch.Tensor: Prompt embeddings of shape (batch, seq_len, 15360)
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt

    # Format input messages
    messages_batch = format_text_input(prompts=prompt)

    # Process all messages at once
    inputs = tokenizer.apply_chat_template(
        messages_batch,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length,
    )

    # Move to device
    device = next(text_encoder.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    print(f"📊 Input tokens: {input_ids.shape[1]}")

    # Forward pass through the model with hidden states
    with torch.inference_mode():
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    # Extract and stack hidden states from specified layers
    # output.hidden_states is a tuple of (num_layers + 1) tensors
    # Index 0 is embeddings, indices 1-N are layer outputs
    print(f"📊 Total hidden state layers available: {len(output.hidden_states)}")
    print(f"📊 Extracting from layers: {hidden_states_layers}")

    out = torch.stack(
        [output.hidden_states[k] for k in hidden_states_layers],
        dim=1
    )
    out = out.to(dtype=text_encoder.dtype, device=device)

    # Reshape: (batch, num_layers, seq, hidden) -> (batch, seq, num_layers * hidden)
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(
        batch_size, seq_len, num_channels * hidden_dim
    )

    print(f"✅ Mistral encoding complete: shape={prompt_embeds.shape}")

    return prompt_embeds


def encode_prompt_mistral(
    prompt: str,
    device: str = "mps",
    max_length: int = MAX_SEQUENCE_LENGTH,
    use_fp8: bool = False
) -> Optional[torch.Tensor]:
    """Encode prompt using Mistral model.

    High-level function that handles model loading and encoding.

    Args:
        prompt: Text prompt to encode
        device: Target device for output tensor
        max_length: Maximum token length
        use_fp8: Whether to use FP8 quantized model (saves memory)

    Returns:
        torch.Tensor: Prompt embeddings, or None if encoding fails
    """
    try:
        model, tokenizer = get_mistral_encoder(use_fp8=use_fp8)

        prompt_embeds = get_mistral_prompt_embeds(
            text_encoder=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_sequence_length=max_length,
        )

        # Ensure correct dtype
        prompt_embeds = prompt_embeds.to(torch.bfloat16)

        return prompt_embeds

    except Exception as e:
        print(f"❌ Mistral encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def unload_mistral_encoder():
    """Unload Mistral encoder to free memory.

    Call this after encoding to free GPU memory before loading FLUX.2.
    """
    global _mistral_model, _mistral_tokenizer

    if _mistral_model is not None:
        print("🧹 Unloading Mistral encoder...")

        # Move to CPU first to free GPU memory
        try:
            _mistral_model.to("cpu")
        except:
            pass

        del _mistral_model
        del _mistral_tokenizer
        _mistral_model = None
        _mistral_tokenizer = None

        # Force garbage collection
        gc.collect()

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except:
                pass

        print("✅ Mistral encoder unloaded")


def get_encoder_status() -> dict:
    """Get current status of the Mistral encoder.

    Returns:
        dict: Status information
    """
    return {
        "mistral_available": is_mistral_available(),
        "encoder_loaded": _mistral_model is not None,
        "model_id": MISTRAL_MODEL_ID,
        "hidden_state_layers": HIDDEN_STATE_LAYERS,
        "embedding_dim": len(HIDDEN_STATE_LAYERS) * 5120  # 3 * 5120 = 15360
    }
