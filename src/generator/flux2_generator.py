"""FLUX.2-dev Unified Image Generation Module

This module provides a unified interface for FLUX.2-dev multi-modal generation,
replacing 8 separate FLUX.1/Qwen model pipelines with a single architecture.

Key Features:
- Unified pipeline supporting 7 generation modes
- Text-to-Image, Image-to-Image, Inpainting, Outpainting
- Depth-guided, Canny-guided, Multi-reference composition
- Dynamic LoRA loading (up to 3 simultaneous adapters)
- Full qint8 quantization (Transformer + Text Encoder)
- Progress tracking and database integration

Modes Supported:
- ✨ text-to-image: Standard prompt-based generation
- 🔄 image-to-image: Image variation and transformation
- 🎨 inpainting: Fill masked areas with AI-generated content
- 📐 outpainting: Extend image boundaries
- 🌊 depth-guided: Structure-preserving generation using depth maps
- 🖋️ canny-guided: Edge-preserving generation using Canny edges
- 🔀 multi-reference: Combine multiple reference images (FLUX.2 feature)

Quantization Options (from VincentGOURBIN/flux_qint_8bit):
- qint8: Pre-quantized 8-bit models (~52GB total)
  - Transformer: qint8 (~30GB)
  - Text Encoder (Mistral): qint8 (~22GB)
  - VAE: bfloat16 (not quantized, ~3GB)
- full: Full precision from black-forest-labs/FLUX.2-dev (~115GB)

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
import io
import requests

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
from huggingface_hub import get_token, snapshot_download, list_repo_files
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

    Note: Text encoder quantization is NOT used due to dtype issues causing
    NaN values and black images. Only transformer is quantized.
    """
    base_class = Flux2Transformer2DModel


# Pre-quantized model repository (contains FLUX.2-dev and FLUX.1-schnell)
QUANTO_REPO_ID = "VincentGOURBIN/flux_qint_8bit"

# Model path mapping for the quantized repo
# Structure: {model_slug}/transformer/qint8, {model_slug}/text_encoder/qint8
FLUX2_MODEL_SLUG = "flux-2-dev"


class Flux2Generator:
    """Unified FLUX.2-dev generator supporting all generation modes via single pipeline.

    This class consolidates 8 separate model pipelines (FLUX.1-dev, FLUX.1-Krea-dev,
    Qwen-Image, FLUX Fill, Kontext, Depth, Canny, Redux) into one unified architecture.

    Attributes:
        device (str): Target device ('mps', 'cuda', or 'cpu')
        dtype (torch.dtype): Model precision (bfloat16 for GPU/MPS, float32 for CPU)
        model_id (str): HuggingFace model identifier
        use_remote_text_encoder (bool): Whether to use remote text encoding
        pipeline (Flux2Pipeline): Cached pipeline instance
        current_lora_paths (list): Currently loaded LoRA file paths
        current_lora_scales (list): Current LoRA influence scales
        current_quantization (str): Current quantization setting
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

        # Model selection - default to standard (will quantize if requested)
        # Pre-quantized BNB 4-bit: "diffusers/FLUX.2-dev-bnb-4bit"
        # Standard: "black-forest-labs/FLUX.2-dev"
        self.model_id = "black-forest-labs/FLUX.2-dev"

        # Memory optimization toggle
        self.use_remote_text_encoder = False  # Disabled by default, can be enabled via UI

        # Pipeline caching (same pattern as ImageGenerator)
        self.pipeline = None
        self.current_lora_paths = []
        self.current_lora_scales = []
        self.current_quantization = None

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

    def generate(
        self,
        prompt: str,
        mode: str = "text-to-image",
        reference_images: Optional[List[Image.Image]] = None,
        mask: Optional[Image.Image] = None,
        control_image: Optional[Image.Image] = None,
        control_type: Optional[str] = None,
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
        """Unified generation method supporting all FLUX.2 modes.

        Args:
            prompt: Text description of desired image
            mode: Generation mode (text-to-image, image-to-image, inpainting, etc.)
            reference_images: Input images for image-based modes
            mask: Mask image for inpainting/outpainting
            control_image: Preprocessed control image (depth map or canny edges)
            control_type: Type of control ("depth" or "canny")
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
            # Ensure pipeline is loaded with correct quantization
            self._ensure_pipeline_loaded(quantization)

            # Update LoRA adapters if changed
            if lora_paths != self.current_lora_paths or lora_scales != self.current_lora_scales:
                self._update_loras(lora_paths or [], lora_scales or [])

            # Prepare pipeline inputs based on mode
            pipeline_inputs = self._prepare_inputs(
                mode=mode,
                prompt=prompt,
                reference_images=reference_images,
                mask=mask,
                control_image=control_image,
                control_type=control_type
            )

            # Remote text encoder (optional memory optimization)
            if self.use_remote_text_encoder:
                try:
                    pipeline_inputs['prompt_embeds'] = self._get_remote_embeds(prompt)
                    pipeline_inputs.pop('prompt', None)  # Use embeds instead
                except Exception as e:
                    print(f"⚠️  Remote text encoder failed, using local: {e}")
                    # Continue with local prompt

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
                control_type=control_type,
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
        """Load FLUX.2 pipeline with appropriate quantization if not already loaded.

        Args:
            quantization: Quantization mode ("qint8" or "full")

        Pre-quantized models are loaded from VincentGOURBIN/flux_qint_8bit.
        Structure: flux-2-dev/transformer/qint8/, flux-2-dev/text_encoder/qint8/
        """

        # Check if pipeline needs reloading (quantization changed)
        if self.pipeline is not None and self.current_quantization == quantization:
            return  # Already loaded with correct quantization

        print(f"🔄 Loading FLUX.2 pipeline with {quantization} quantization...")

        # Clean up existing pipeline
        if self.pipeline is not None:
            self.pipeline = None  # Set to None instead of del to keep attribute
            cleanup_memory()

        try:
            if quantization == "qint8":
                # Check for available transformer in the repo
                has_transformer = self._check_repo_has_component(
                    QUANTO_REPO_ID, FLUX2_MODEL_SLUG, "transformer", "qint8"
                )

                if not has_transformer:
                    raise RuntimeError(f"No quantized transformer found in {QUANTO_REPO_ID}/{FLUX2_MODEL_SLUG}")

                # Only quantize transformer - text encoder quantization causes dtype issues (NaN/black images)
                transformer_subpath = f"{FLUX2_MODEL_SLUG}/transformer/qint8"
                download_patterns = [f"{FLUX2_MODEL_SLUG}/transformer/qint8/*"]

                print(f"📦 Loading pre-quantized qint8 models from {QUANTO_REPO_ID}...")
                print(f"  📦 Transformer: {transformer_subpath}")
                print(f"  📦 Text Encoder: full precision (quantized causes dtype issues)")

                # Download quantized transformer
                quant_path = snapshot_download(
                    QUANTO_REPO_ID,
                    allow_patterns=download_patterns
                )
                transformer_path = os.path.join(quant_path, transformer_subpath)

                # Load base pipeline WITHOUT transformer only (keep text encoder full precision)
                print("📦 Loading base pipeline with full precision text encoder...")
                try:
                    self.pipeline = Flux2Pipeline.from_pretrained(
                        self.model_id,
                        transformer=None,
                        torch_dtype=self.dtype,
                        use_safetensors=True,
                        local_files_only=True
                    )
                    print("  ✅ Loaded from local cache")
                except Exception:
                    print("  ℹ️  Not in cache, downloading...")
                    self.pipeline = Flux2Pipeline.from_pretrained(
                        self.model_id,
                        transformer=None,
                        torch_dtype=self.dtype,
                        use_safetensors=True
                    )

                # Move pipeline components to device
                self.pipeline = self.pipeline.to(self.device)

                # Load quantized transformer
                print(f"📦 Loading quantized transformer...")
                quantized_transformer = QuantizedFlux2Transformer2DModel.from_pretrained(
                    transformer_path
                )
                print(f"  📦 Moving to {self.device}...")
                quantized_transformer.to(self.device)
                self.pipeline.transformer = quantized_transformer
                print(f"  ✅ Transformer loaded (qint8)")

                print(f"✅ Pre-quantized model loaded (transformer qint8, text encoder full)")

            elif quantization == "full":
                # Full precision from official Black Forest Labs repo
                # Try local cache first to avoid re-downloading
                print("📦 Loading full precision model from black-forest-labs/FLUX.2-dev...")
                try:
                    self.pipeline = Flux2Pipeline.from_pretrained(
                        self.model_id,
                        torch_dtype=self.dtype,
                        use_safetensors=True,
                        local_files_only=True  # Try cache first
                    ).to(self.device)
                    print("  ✅ Loaded from local cache")
                except Exception:
                    print("  ℹ️  Not in cache, downloading...")
                    self.pipeline = Flux2Pipeline.from_pretrained(
                        self.model_id,
                        torch_dtype=self.dtype,
                        use_safetensors=True
                    ).to(self.device)

                print("✅ Full precision model loaded successfully")

            else:
                # Unknown quantization or legacy "None", default to qint8
                if quantization == "None":
                    print("⚠️  'None' is deprecated, use 'full' for full precision or 'qint8' for quantized")
                    return self._ensure_pipeline_loaded("full")
                print(f"⚠️  Unknown quantization '{quantization}', defaulting to qint8...")
                return self._ensure_pipeline_loaded("qint8")

            # Enable memory optimizations
            if hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing()

            # Update current quantization state
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
        prompt: str,
        reference_images: Optional[List[Image.Image]],
        mask: Optional[Image.Image],
        control_image: Optional[Image.Image],
        control_type: Optional[str]
    ) -> Dict[str, Any]:
        """Prepare pipeline inputs based on generation mode.

        Args:
            mode: Generation mode
            prompt: Text prompt
            reference_images: Input images (if applicable)
            mask: Mask image (if applicable)
            control_image: Control image (if applicable)
            control_type: Type of control (if applicable)

        Returns:
            dict: Pipeline input parameters
        """

        base_inputs = {'prompt': prompt}

        if mode == "text-to-image":
            # Standard text-to-image generation
            return base_inputs

        elif mode == "image-to-image":
            # Image variation / transformation
            if reference_images and len(reference_images) > 0:
                base_inputs['image'] = reference_images[0]
            return base_inputs

        elif mode == "inpainting":
            # Fill masked areas
            if reference_images and len(reference_images) > 0:
                base_inputs['image'] = reference_images[0]
            if mask:
                base_inputs['mask_image'] = mask
            return base_inputs

        elif mode == "outpainting":
            # Extend image boundaries
            if reference_images and len(reference_images) > 0:
                base_inputs['image'] = reference_images[0]
            if mask:
                base_inputs['mask_image'] = mask
            return base_inputs

        elif mode == "depth-guided":
            # Depth-preserving generation
            if control_image:
                base_inputs['control_image'] = control_image
                base_inputs['controlnet_conditioning_scale'] = 0.7
            return base_inputs

        elif mode == "canny-guided":
            # Edge-preserving generation
            if control_image:
                base_inputs['control_image'] = control_image
                base_inputs['controlnet_conditioning_scale'] = 0.7
            return base_inputs

        elif mode == "multi-reference":
            # Multi-image composition (NEW FLUX.2 feature)
            if reference_images and len(reference_images) > 0:
                base_inputs['image'] = reference_images  # List of images
            return base_inputs

        else:
            # Unknown mode - fallback to text-to-image
            print(f"⚠️  Unknown mode '{mode}', defaulting to text-to-image")
            return base_inputs

    def _get_remote_embeds(self, prompt: str) -> torch.Tensor:
        """Get prompt embeddings from remote text encoder (memory optimization).

        Args:
            prompt: Text prompt to encode

        Returns:
            torch.Tensor: Prompt embeddings
        """

        response = requests.post(
            "https://remote-text-encoder-flux-2.huggingface.co/predict",
            json={"prompt": prompt},
            headers={
                "Authorization": f"Bearer {get_token()}",
                "Content-Type": "application/json"
            }
        )

        prompt_embeds = torch.load(io.BytesIO(response.content))
        return prompt_embeds.to(self.device)

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
