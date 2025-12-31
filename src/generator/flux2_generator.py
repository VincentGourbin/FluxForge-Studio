"""FLUX.2-dev Unified Image Generation Module

This module provides a unified interface for FLUX.2-dev generation,
replacing 8 separate FLUX.1/Qwen model pipelines with a single architecture.

Key Features:
- Unified pipeline supporting 3 generation modes
- Dynamic LoRA loading (up to 3 simultaneous adapters)
- Two-step inference with standalone Mistral encoder for memory optimization
- Progress tracking and database integration

Two-Step Inference (Memory Optimization):
- Step 1: Load Mistral text encoder (~48GB) → encode prompt → unload
- Step 2: Load transformer (+ VAE) → generate image with prompt_embeds → unload
- Mistral and transformer are never in memory together

Modes Supported:
- text-to-image: Standard prompt-based generation
- image-to-image: Image variation and transformation
- multi-reference: Combine multiple reference images (Multi-Angles)

Quantization Options:
- qint8: Quantized transformer (~30GB) + VAE (~3GB)
- full: Full precision transformer (~60GB) + VAE (~3GB)
Text encoder: Mistral-Small-3.2-24B (full precision, ~48GB bfloat16)

Author: FluxForge Team
License: MIT
"""

import os
import datetime
import random
import time
import torch
import gc
import warnings

# Suppress harmless dtype mismatch warning from quanto quantization
# This occurs because quanto stores weights in float32 but computations use bfloat16
warnings.filterwarnings("ignore", message="Mismatch dtype between input and weight")
from pathlib import Path
from typing import List, Optional, Callable, Tuple, Dict, Any
from PIL import Image

# Hugging Face libraries
from diffusers import Flux2Pipeline
from diffusers.models import Flux2Transformer2DModel
from diffusers.utils import load_image
from huggingface_hub import snapshot_download, list_repo_files
from optimum.quanto import QuantizedDiffusersModel

# Local imports
from core import config
from core.database import save_flux2_generation
from utils.progress_tracker import global_progress_tracker
from utils.image_processing import ensure_rgb_format, cleanup_memory, save_image_with_metadata


# Custom quantized transformer class for loading pre-quantized models
class QuantizedFlux2Transformer2DModel(QuantizedDiffusersModel):
    """Quantized wrapper for Flux2Transformer2DModel.

    Used to load pre-quantized transformer weights from HuggingFace Hub.
    Compatible with qint8, qint4, and qint2 quantization types.
    """
    base_class = Flux2Transformer2DModel


# Pre-quantized model repository
QUANTO_REPO_ID = "VincentGOURBIN/flux_qint_8bit"  # Quantized transformer

# Model path mapping for the quantized repo
# Structure: {model_slug}/transformer/qint8
FLUX2_MODEL_SLUG = "flux-2-dev"


class Flux2Generator:
    """Unified FLUX.2-dev generator supporting 3 generation modes.

    Modes: text-to-image, image-to-image, multi-reference (Multi-Angles)

    Uses two-step inference for memory optimization:
    1. Mistral text encoder (~48GB) encodes prompt, then unloads
    2. Transformer + VAE generates image with prompt_embeds

    Attributes:
        device (str): Target device ('mps', 'cuda', or 'cpu')
        dtype (torch.dtype): Model precision (bfloat16 for GPU/MPS, float32 for CPU)
        model_id (str): HuggingFace model identifier
        pipeline (Flux2Pipeline): Cached diffusion pipeline instance
        current_lora_paths (list): Currently loaded LoRA file paths
        current_lora_scales (list): Current LoRA influence scales
        current_quantization (str): Current quantization setting ('qint8' or 'full')
        lora_directory (str): Directory containing LoRA files
        lora_data (list): Available LoRA models metadata
    """

    def __init__(self):
        """Initialize FLUX.2 generator with device configuration and defaults."""

        # Device and precision configuration
        self.device = config.device

        # FLUX.2 requires bfloat16 for optimal performance
        if self.device in ['mps', 'cuda']:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        # Model selection
        # Full precision: black-forest-labs/FLUX.2-dev (complete pipeline)
        # Quantized: transformer from VincentGOURBIN + text encoder from unsloth
        self.model_id = "black-forest-labs/FLUX.2-dev"

        # Pipeline caching
        self.pipeline = None
        self.current_lora_paths = []
        self.current_lora_scales = []
        self.current_quantization = None

        # Two-step inference: Mistral text encoder loaded separately from transformer
        # This saves memory by never having both in VRAM at the same time
        # Text encoding handled by src/encoder/mistral_encoder.py (singleton pattern)

        # LoRA configuration from database
        self.lora_directory = config.lora_directory
        self.lora_data = self._get_lora_data()

        print(f"✅ FLUX.2 Generator initialized on {self.device} with {self.dtype}")

    def _get_lora_data(self):
        """Load LoRA metadata from database.

        Returns:
            list: LoRA models metadata including file names, activation keywords, etc.
        """
        try:
            from core.database import get_lora_for_image_generator
            return get_lora_for_image_generator()
        except Exception as e:
            print(f"⚠️  Failed to load LoRA data: {e}")
            return []

    def _encode_prompt_separately(self, prompt: str) -> torch.Tensor:
        """Encode prompt using standalone Mistral encoder, then unload to save memory.

        Two-step inference optimization:
        1. Load Mistral text encoder (~48GB)
        2. Encode prompt → get prompt_embeds (batch, 512, 15360)
        3. Unload Mistral to free memory
        4. Return embeddings for later use with transformer

        Uses the standalone Mistral encoder from src/encoder/mistral_encoder.py
        which extracts hidden states from layers 10, 20, 30.

        Args:
            prompt: Text prompt to encode

        Returns:
            torch.Tensor: Prompt embeddings of shape (1, 512, 15360)
        """
        from encoder.mistral_encoder import (
            encode_prompt_mistral,
            unload_mistral_encoder
        )

        print("🧠 Step 1/2: Text encoding with Mistral...")
        print(f"  📝 Encoding: '{prompt[:50]}...'")

        # Encode with standalone Mistral
        prompt_embeds = encode_prompt_mistral(
            prompt=prompt,
            device=self.device,
            use_fp8=False  # Full precision only (FP8 doesn't work reliably)
        )

        if prompt_embeds is None:
            raise RuntimeError("Failed to encode prompt with Mistral")

        print(f"  ✅ Encoded: shape={prompt_embeds.shape}, dtype={prompt_embeds.dtype}")

        # Unload Mistral to free memory for transformer
        print("  🧹 Unloading text encoder...")
        unload_mistral_encoder()
        cleanup_memory()

        return prompt_embeds

    def generate(
        self,
        prompt: str,
        mode: str = "text-to-image",
        reference_images: Optional[List[Image.Image]] = None,
        steps: int = 28,
        guidance_scale: float = 4.0,
        width: int = 1024,
        height: int = 1024,
        seed: int = 0,
        lora_paths: Optional[List[str]] = None,
        lora_scales: Optional[List[float]] = None,
        quantization: str = "qint8",
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Image.Image, str, Dict[str, Any]]:
        """Unified generation method supporting FLUX.2 modes.

        Supported modes:
        - text-to-image: Standard prompt-based generation
        - image-to-image: Image variation/transformation
        - multi-reference: Multi-image composition (Multi-Angles)

        Args:
            prompt: Text description of desired image
            mode: Generation mode (text-to-image, image-to-image, multi-reference)
            reference_images: Input images for image-based modes
            steps: Number of diffusion steps (28-50 recommended, 28 is cost-effective)
            guidance_scale: Classifier-free guidance strength (4.0 default)
            width: Output image width
            height: Output image height
            seed: Random seed (0 = random)
            lora_paths: List of LoRA file paths to apply
            lora_scales: Corresponding LoRA influence scales
            quantization: Quantization mode ("qint8" or "full")
            progress_callback: Optional callback for progress updates

        Returns:
            tuple: (generated_image, status_message, timing_info)
        """

        # Timing measurements
        start_time = time.time()

        # Random seed generation
        if seed == 0:
            seed = random.randint(0, 2**32 - 1)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        try:
            # ===== TWO-STEP INFERENCE FOR MEMORY OPTIMIZATION =====
            # Step 1: Encode prompt (load text encoder, encode, unload)
            # Step 2: Generate image (load transformer only, use prompt_embeds)
            # This way text encoder and transformer are never in memory together

            # Step 1: Encode prompt
            prompt_embeds = self._encode_prompt_separately(prompt)

            # Step 2: Load transformer-only pipeline
            print("🎨 Step 2/2: Image generation...")
            self._ensure_pipeline_loaded(quantization)

            # Setup LoRA adapters
            if lora_paths != self.current_lora_paths or lora_scales != self.current_lora_scales:
                self._update_loras(lora_paths or [], lora_scales or [])

            # Prepare pipeline inputs with pre-computed embeddings
            pipeline_inputs = self._prepare_inputs(
                mode=mode,
                prompt=None,  # Don't pass prompt, use embeddings
                prompt_embeds=prompt_embeds,
                reference_images=reference_images,
            )

            # Progress tracking integration
            global_progress_tracker.reset()
            global_progress_tracker.apply_tqdm_patches()

            try:
                # Start model execution timing
                model_start_time = time.time()

                # Generate image
                result = self.pipeline(
                    **pipeline_inputs,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator
                )

                model_end_time = time.time()

                # Extract generated image
                generated_image = result.images[0]

            finally:
                global_progress_tracker.remove_tqdm_patches()

            # Ensure RGB format
            generated_image = ensure_rgb_format(generated_image)

            # Calculate timing
            end_time = time.time()
            total_time = end_time - start_time
            model_time = model_end_time - model_start_time

            # Save to database
            now = datetime.datetime.now()
            timestamp_db = now.strftime("%Y-%m-%d %H:%M:%S")  # For database (consistent with other generators)
            timestamp_file = now.strftime("%Y%m%d_%H%M%S")    # For filename
            output_filename = f"flux2_{mode.replace('-', '_')}_{timestamp_file}_{seed}.png"
            output_path = os.path.join("outputimage", output_filename)

            # Save image file
            os.makedirs("outputimage", exist_ok=True)
            generated_image.save(output_path)

            # Save to database with metadata (use output_path which includes directory)
            save_flux2_generation(
                timestamp=timestamp_db,  # Use proper timestamp format for database sorting
                seed=seed,
                prompt=prompt,
                flux2_mode=mode,
                steps=steps,
                guidance=guidance_scale,
                height=height,
                width=width,
                lora_paths=lora_paths or [],
                lora_scales=lora_scales or [],
                output_filename=output_path,  # Full path for load_history to find the file
                quantization=quantization,
                control_type=None,  # No longer used, kept for DB compatibility
                total_generation_time=total_time,
                model_generation_time=model_time
            )

            # Prepare return info
            timing_info = {
                'total_time': total_time,
                'model_time': model_time,
                'overhead_time': total_time - model_time
            }

            status_message = (
                f"✅ Generated successfully in {total_time:.2f}s "
                f"(model: {model_time:.2f}s) - Mode: {mode}"
            )

            return generated_image, status_message, timing_info

        except Exception as e:
            error_message = f"❌ Generation failed: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()

            # Return None with error info
            return None, error_message, {'total_time': time.time() - start_time}

    def _check_repo_has_component(self, repo_id: str, model_slug: str, component: str, quant_type: str) -> bool:
        """Check if the HuggingFace repo has a specific quantized component.

        Args:
            repo_id: HuggingFace repository ID
            model_slug: Model slug (e.g., "flux-2-dev", "flux-1-schnell")
            component: Component name ("transformer" or "text_encoder")
            quant_type: Quantization type ("qint8", etc.)

        Returns:
            bool: True if the component exists in the repo
        """
        try:
            files = list_repo_files(repo_id)
            # New structure: flux-2-dev/transformer/qint8/
            pattern = f"{model_slug}/{component}/{quant_type}/"
            return any(f.startswith(pattern) for f in files)
        except Exception:
            return False

    def _ensure_pipeline_loaded(self, quantization: str):
        """Load FLUX.2 pipeline (transformer only) with appropriate quantization.

        Two-step inference: text encoder is loaded separately in _encode_prompt_separately().
        This method only loads the transformer and VAE for diffusion.

        Args:
            quantization: Quantization mode ("qint8" or "full")

        Loading strategy:
        - qint8: Quantized transformer from VincentGOURBIN/flux_qint_8bit
        - full: Full precision transformer from black-forest-labs/FLUX.2-dev
        Both modes: NO text encoder (uses prompt_embeds from step 1)
        """

        # Check if pipeline needs reloading
        if self.pipeline is not None and self.current_quantization == quantization:
            return  # Already loaded with correct configuration

        print(f"  📦 Loading transformer with {quantization} quantization...")

        # Clean up existing pipeline
        if self.pipeline is not None:
            self.pipeline = None
            cleanup_memory()

        try:
            if quantization == "qint8":
                # Load quantized transformer from VincentGOURBIN repo
                has_transformer = self._check_repo_has_component(
                    QUANTO_REPO_ID, FLUX2_MODEL_SLUG, "transformer", "qint8"
                )

                if not has_transformer:
                    raise RuntimeError(f"No quantized transformer found in {QUANTO_REPO_ID}/{FLUX2_MODEL_SLUG}")

                transformer_subpath = f"{FLUX2_MODEL_SLUG}/transformer/qint8"
                download_patterns = [f"{FLUX2_MODEL_SLUG}/transformer/qint8/*"]

                print(f"  📦 Transformer: {transformer_subpath} (qint8)")

                # Download quantized transformer
                quant_path = snapshot_download(
                    QUANTO_REPO_ID,
                    allow_patterns=download_patterns
                )
                transformer_path = os.path.join(quant_path, transformer_subpath)

                # Load base pipeline WITHOUT transformer AND text encoder
                # (transformer will be replaced, text encoding done separately)
                try:
                    self.pipeline = Flux2Pipeline.from_pretrained(
                        self.model_id,
                        transformer=None,
                        text_encoder=None,  # No text encoder needed
                        torch_dtype=self.dtype,
                        use_safetensors=True,
                        local_files_only=True
                    )
                    print("  ✅ VAE loaded from local cache")
                except Exception:
                    print("  ℹ️  Not in cache, downloading...")
                    self.pipeline = Flux2Pipeline.from_pretrained(
                        self.model_id,
                        transformer=None,
                        text_encoder=None,
                        torch_dtype=self.dtype,
                        use_safetensors=True
                    )

                self.pipeline = self.pipeline.to(self.device)

                # Load and set quantized transformer
                quantized_transformer = QuantizedFlux2Transformer2DModel.from_pretrained(
                    transformer_path
                )
                quantized_transformer.to(self.device)
                self.pipeline.transformer = quantized_transformer
                print(f"  ✅ Transformer loaded (qint8)")

            elif quantization == "full":
                # Full precision transformer (no text encoder)
                print("  📦 Transformer: full precision")

                try:
                    self.pipeline = Flux2Pipeline.from_pretrained(
                        self.model_id,
                        text_encoder=None,  # No text encoder needed
                        torch_dtype=self.dtype,
                        use_safetensors=True,
                        local_files_only=True
                    ).to(self.device)
                    print("  ✅ Loaded from local cache")
                except Exception:
                    print("  ℹ️  Not in cache, downloading...")
                    self.pipeline = Flux2Pipeline.from_pretrained(
                        self.model_id,
                        text_encoder=None,
                        torch_dtype=self.dtype,
                        use_safetensors=True
                    ).to(self.device)

            else:
                # Unknown quantization, default to qint8
                if quantization == "None":
                    print("⚠️  'None' is deprecated, use 'full' or 'qint8'")
                    return self._ensure_pipeline_loaded("full")
                print(f"⚠️  Unknown quantization '{quantization}', defaulting to qint8...")
                return self._ensure_pipeline_loaded("qint8")

            # Enable memory optimizations
            if hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing()

            self.current_quantization = quantization

        except Exception as e:
            error_msg = f"❌ Failed to load FLUX.2 pipeline: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            raise RuntimeError(error_msg)

    def _prepare_inputs(
        self,
        mode: str,
        prompt: Optional[str],
        prompt_embeds: Optional[torch.Tensor],
        reference_images: Optional[List[Image.Image]],
    ) -> Dict[str, Any]:
        """Prepare pipeline inputs based on generation mode.

        Supported modes:
        - text-to-image: Standard prompt-based generation
        - image-to-image: Image variation/transformation
        - multi-reference: Multi-image composition (Multi-Angles)

        Args:
            mode: Generation mode
            prompt: Text prompt (None if using prompt_embeds)
            prompt_embeds: Pre-computed prompt embeddings
            reference_images: Input images (if applicable)

        Returns:
            dict: Pipeline input parameters
        """
        # Use either prompt or prompt_embeds (prompt_embeds takes precedence)
        if prompt_embeds is not None:
            base_inputs = {'prompt_embeds': prompt_embeds}
        else:
            base_inputs = {'prompt': prompt}

        if mode == "text-to-image":
            return base_inputs

        elif mode == "image-to-image":
            if reference_images and len(reference_images) > 0:
                base_inputs['image'] = reference_images[0]
            return base_inputs

        elif mode == "multi-reference":
            if reference_images and len(reference_images) > 0:
                base_inputs['image'] = reference_images  # List of images
            return base_inputs

        else:
            print(f"⚠️  Unknown mode '{mode}', defaulting to text-to-image")
            return base_inputs

    def _check_lora_compatibility(self, lora_filename: str) -> bool:
        """Check if a LoRA is compatible with FLUX.2-dev.

        Args:
            lora_filename: Name of the LoRA file

        Returns:
            bool: True if compatible, False otherwise
        """
        # Find LoRA in loaded data
        for lora_info in self.lora_data:
            if lora_info['file_name'] == lora_filename:
                compatible_models = lora_info.get('compatible_models', [])
                # Check if flux2-dev is in compatible models
                if 'flux2-dev' in compatible_models:
                    return True
                # If no compatibility info, assume compatible (legacy behavior)
                if not compatible_models:
                    print(f"  ⚠️  LoRA '{lora_filename}' has no compatibility info, assuming compatible")
                    return True
                return False
        # LoRA not found in database, assume compatible
        return True

    def _update_loras(self, lora_paths: List[str], lora_scales: List[float]):
        """Update loaded LoRA adapters.

        Args:
            lora_paths: List of LoRA file paths
            lora_scales: Corresponding influence scales
        """

        if not self.pipeline:
            print("⚠️  Pipeline not loaded, skipping LoRA update")
            return

        try:
            # Unload existing LoRAs
            if self.current_lora_paths:
                print("🔄 Unloading previous LoRA adapters...")
                if hasattr(self.pipeline, 'disable_lora'):
                    self.pipeline.disable_lora()

            # Load new LoRAs if provided
            if lora_paths:
                print(f"📦 Loading {len(lora_paths)} LoRA adapter(s)...")

                adapter_names = []
                adapter_weights = []

                for i, (lora_path, scale) in enumerate(zip(lora_paths, lora_scales)):
                    try:
                        # Generate adapter name from filename
                        lora_filename = os.path.basename(lora_path)
                        adapter_name = lora_filename.replace('.safetensors', '').replace('.', '_')

                        # Check LoRA compatibility with FLUX.2
                        if not self._check_lora_compatibility(lora_filename):
                            print(f"  ⚠️  Skipping LoRA '{lora_filename}': not compatible with FLUX.2-dev")
                            print(f"      This LoRA was trained for a different model architecture.")
                            continue

                        # Load LoRA weights
                        self.pipeline.load_lora_weights(
                            self.lora_directory,
                            weight_name=lora_filename,
                            adapter_name=adapter_name
                        )

                        adapter_names.append(adapter_name)
                        adapter_weights.append(scale)

                        print(f"  ✅ Loaded LoRA {i+1}/{len(lora_paths)}: {lora_filename} (scale: {scale})")

                    except Exception as e:
                        print(f"  ⚠️  Failed to load LoRA '{lora_path}': {e}")
                        continue

                # Set adapter weights if any loaded successfully
                if adapter_names:
                    # Use transformer.set_adapters directly (pipeline.set_adapters doesn't sync properly)
                    self.pipeline.transformer.set_adapters(adapter_names, adapter_weights)
                    print(f"✅ Applied {len(adapter_names)} LoRA adapter(s)")
                else:
                    print("⚠️  No LoRA adapters loaded successfully")

            # Update current state
            self.current_lora_paths = lora_paths
            self.current_lora_scales = lora_scales

        except Exception as e:
            print(f"❌ LoRA update failed: {e}")
            import traceback
            traceback.print_exc()

    def unload_pipeline(self):
        """Unload FLUX.2 pipeline to free memory.

        Call this after generation is complete if you need to free GPU memory
        for other tasks. The pipeline will be reloaded on next generate() call.

        Note: Text encoder (Mistral) is managed separately via encoder.mistral_encoder
        """
        if self.pipeline is not None:
            print("🧹 Unloading FLUX.2 pipeline...")

            # Reset state
            self.pipeline = None
            self.current_lora_paths = []
            self.current_lora_scales = []
            self.current_quantization = None

            # Force memory cleanup
            cleanup_memory()

            print("✅ FLUX.2 pipeline unloaded")
        else:
            print("ℹ️  No pipeline loaded to unload")


# Singleton instance
_flux2_generator_instance = None


def get_flux2_generator() -> Flux2Generator:
    """Get or create the singleton FLUX.2 generator instance.

    Returns:
        Flux2Generator: Shared generator instance
    """
    global _flux2_generator_instance

    if _flux2_generator_instance is None:
        _flux2_generator_instance = Flux2Generator()

    return _flux2_generator_instance
