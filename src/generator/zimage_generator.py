"""
Z-Image-Turbo Generation Module

This module provides image generation using the Tongyi-MAI/Z-Image-Turbo model.
Z-Image-Turbo is a 6B parameter model optimized for fast, high-quality text-to-image generation
using only 8 NFEs (Number of Function Evaluations).

Key Features:
- Fast generation with 8 NFEs (9 steps)
- High quality output with 6B parameters
- Supports bfloat16 for memory efficiency
- Must use guidance_scale=0.0 for Turbo models

Author: Vincent
License: MIT
"""

import os
import datetime
import random
import time
import torch
import gc
import warnings

from pathlib import Path
from typing import Optional, Tuple, List
from PIL import Image

from core import config
from diffusers import ZImagePipeline
from utils.progress_tracker import global_progress_tracker
from utils.image_processing import cleanup_memory


# Model ID for Z-Image-Turbo
ZIMAGE_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"


class ZImageGenerator:
    """Generator class for Z-Image-Turbo text-to-image generation.

    This class manages the Z-Image-Turbo pipeline for fast, high-quality
    image generation. It uses only 8 NFEs for generation.

    Attributes:
        pipeline: Cached ZImagePipeline instance
        device: Target device (mps, cuda, cpu)
        dtype: Model precision (bfloat16 or float32)
        _is_loaded: Whether the pipeline is currently loaded
    """

    def __init__(self):
        """Initialize the ZImageGenerator with default state."""
        self.pipeline = None
        self._is_loaded = False

        # Device configuration
        self.device = config.device

        # Z-Image works best with bfloat16
        if self.device == 'mps':
            self.dtype = torch.bfloat16
        elif self.device == 'cuda':
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        # LoRA configuration
        self.current_lora_paths = []
        self.current_lora_scales = []
        self.lora_directory = config.lora_directory
        self.lora_data = self._get_lora_data()

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

    def _check_lora_compatibility(self, lora_filename: str) -> bool:
        """Check if a LoRA is compatible with Z-Image-Turbo.

        Args:
            lora_filename: Name of the LoRA file

        Returns:
            bool: True if compatible, False otherwise
        """
        for lora_info in self.lora_data:
            if lora_info['file_name'] == lora_filename:
                compatible_models = lora_info.get('compatible_models', [])
                if 'zimage-turbo' in compatible_models:
                    return True
                # If no compatibility info, assume NOT compatible (safer for Z-Image)
                if not compatible_models:
                    print(f"  ⚠️  LoRA '{lora_filename}' has no compatibility info, skipping")
                    return False
                return False
        # LoRA not found in database
        return False

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
                if hasattr(self.pipeline, 'unload_lora_weights'):
                    self.pipeline.unload_lora_weights()

            # Load new LoRAs if provided
            if lora_paths:
                print(f"📦 Loading {len(lora_paths)} LoRA adapter(s) for Z-Image-Turbo...")

                adapter_names = []
                adapter_weights = []

                for i, (lora_path, scale) in enumerate(zip(lora_paths, lora_scales)):
                    try:
                        lora_filename = os.path.basename(lora_path)
                        adapter_name = lora_filename.replace('.safetensors', '').replace('.', '_')

                        # Check LoRA compatibility
                        if not self._check_lora_compatibility(lora_filename):
                            print(f"  ⚠️  Skipping LoRA '{lora_filename}': not compatible with Z-Image-Turbo")
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
                    self.pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
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

    def _ensure_pipeline_loaded(self):
        """Load the Z-Image-Turbo pipeline if not already loaded."""
        if self._is_loaded and self.pipeline is not None:
            return

        print(f"Loading Z-Image-Turbo from {ZIMAGE_MODEL_ID}...")

        try:
            # Try loading from local cache first
            self.pipeline = ZImagePipeline.from_pretrained(
                ZIMAGE_MODEL_ID,
                torch_dtype=self.dtype,
                local_files_only=True
            )
            print("  Loaded from local cache")
        except Exception:
            print("  Not in cache, downloading...")
            self.pipeline = ZImagePipeline.from_pretrained(
                ZIMAGE_MODEL_ID,
                torch_dtype=self.dtype
            )

        # Move to device
        self.pipeline = self.pipeline.to(self.device)

        # Enable memory optimizations
        self.pipeline.enable_attention_slicing()

        self._is_loaded = True
        print(f"Z-Image-Turbo loaded on {self.device}")

    def unload(self):
        """Unload the pipeline to free memory."""
        if self.pipeline is not None:
            self.pipeline = None
            self._is_loaded = False
            cleanup_memory()
            print("Z-Image-Turbo pipeline unloaded")

    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 9,
        seed: Optional[int] = None,
        lora_paths: Optional[List[str]] = None,
        lora_scales: Optional[List[float]] = None,
        negative_prompt: Optional[str] = None
    ) -> Tuple[Optional[Image.Image], str, dict]:
        """Generate an image using Z-Image-Turbo.

        Args:
            prompt: Text description of desired image
            width: Output image width (default 1024)
            height: Output image height (default 1024)
            steps: Number of inference steps (default 9, results in 8 NFEs)
            seed: Random seed for reproducibility (None for random)
            lora_paths: List of LoRA file paths to apply
            lora_scales: Corresponding LoRA influence scales
            negative_prompt: Optional negative prompt (may not be supported)

        Returns:
            tuple: (PIL.Image or None, status_message, timing_info)
        """
        # Ensure pipeline is loaded
        self._ensure_pipeline_loaded()

        # Update LoRAs if changed
        lora_paths = lora_paths or []
        lora_scales = lora_scales or []
        if lora_paths != self.current_lora_paths or lora_scales != self.current_lora_scales:
            self._update_loras(lora_paths, lora_scales)

        # Handle seed
        if seed is None or seed == 0:
            seed = random.randint(1, 2**32 - 1)

        # Ensure dimensions are valid
        width = int(width)
        height = int(height)
        steps = max(1, int(steps))

        print(f"Generating Z-Image-Turbo: {prompt[:50]}...")
        print(f"  Size: {width}x{height}, Steps: {steps}, Seed: {seed}")

        # Start timing
        total_start = time.time()

        try:
            # Set up generator for reproducibility
            generator = torch.Generator(device=self.device).manual_seed(seed)

            # Apply progress tracking
            global_progress_tracker.reset()
            global_progress_tracker.apply_tqdm_patches()

            try:
                model_start = time.time()

                # Generate image
                # IMPORTANT: guidance_scale MUST be 0.0 for Turbo models
                result = self.pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=0.0,  # Must be 0.0 for Turbo
                    generator=generator
                )

                model_end = time.time()
                model_time = model_end - model_start

            finally:
                global_progress_tracker.remove_tqdm_patches()

            image = result.images[0]

            # Calculate total time
            total_end = time.time()
            total_time = total_end - total_start

            # Create timing info
            timing_info = {
                'total_time': total_time,
                'model_time': model_time,
                'seed': seed
            }

            # Save image
            output_dir = Path("outputimage")
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.datetime.now()
            output_filename = output_dir / f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{seed}.png"
            image.save(str(output_filename))

            # Save to database
            self._save_to_database(
                prompt=prompt,
                seed=seed,
                width=width,
                height=height,
                steps=steps,
                output_path=str(output_filename),
                total_time=total_time,
                model_time=model_time,
                lora_paths=lora_paths,
                lora_scales=lora_scales
            )

            # Clean up memory
            self._cleanup_memory()

            status = f"Generated in {total_time:.1f}s (model: {model_time:.1f}s)"
            print(f"  {status}")

            return image, status, timing_info

        except Exception as e:
            error_msg = f"Z-Image generation failed: {str(e)}"
            print(f"  {error_msg}")
            return None, error_msg, {}

    def _save_to_database(
        self,
        prompt: str,
        seed: int,
        width: int,
        height: int,
        steps: int,
        output_path: str,
        total_time: float,
        model_time: float,
        lora_paths: Optional[List[str]] = None,
        lora_scales: Optional[List[float]] = None
    ):
        """Save generation info to database."""
        try:
            from core.database import save_zimage_generation
            import os

            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Debug: verify file exists and log path
            print(f"  💾 Saving to database: {output_path}")
            print(f"  📁 File exists: {os.path.exists(output_path)}")

            save_zimage_generation(
                timestamp=timestamp,
                seed=seed,
                prompt=prompt,
                steps=steps,
                width=width,
                height=height,
                output_path=output_path,
                total_generation_time=total_time,
                model_generation_time=model_time,
                lora_paths=lora_paths,
                lora_scales=lora_scales
            )
            print(f"  ✅ Saved to database successfully")

        except Exception as e:
            print(f"  ❌ Failed to save to database: {e}")
            import traceback
            traceback.print_exc()

    def _cleanup_memory(self):
        """Clean up GPU/MPS memory after generation."""
        try:
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif self.device == 'mps':
                torch.mps.empty_cache()
                torch.mps.synchronize()
            gc.collect()
        except Exception as e:
            print(f"  Warning: Memory cleanup failed: {e}")


# Singleton instance
_zimage_generator: Optional[ZImageGenerator] = None


def get_zimage_generator() -> ZImageGenerator:
    """Get or create the singleton ZImageGenerator instance."""
    global _zimage_generator
    if _zimage_generator is None:
        _zimage_generator = ZImageGenerator()
    return _zimage_generator
