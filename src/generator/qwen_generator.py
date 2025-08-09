"""
Qwen-Image Generator

A dedicated generator for Qwen-Image model with support for MPS (Apple Silicon),
CUDA (NVIDIA GPUs), and CPU fallback.

Features:
- Cross-platform device detection and optimization
- Intelligent memory management
- MPS-optimized dtype selection
- Direct generation (no queue system)
- Magic prompt integration

Author: FluxForge Team
License: MIT
"""

import torch
import gc
import os
import time
from datetime import datetime
from pathlib import Path
from diffusers import DiffusionPipeline

# Try to import QwenImagePipeline, fall back to DiffusionPipeline if not available
try:
    from diffusers import QwenImagePipeline
    QWEN_PIPELINE_AVAILABLE = True
except ImportError:
    print("ℹ️  QwenImagePipeline not available in this diffusers version, using DiffusionPipeline")
    QwenImagePipeline = None
    QWEN_PIPELINE_AVAILABLE = False
from PIL import Image
from typing import Optional, Tuple, Any, List


class QwenImageGenerator:
    """
    Qwen-Image generation class with multi-device support.
    """
    
    def __init__(self):
        """Initialize QwenImageGenerator with device-specific optimizations."""
        
        # Device and precision configuration
        self.device = self._detect_device()
        self.dtype = self._get_optimal_dtype()
        
        # Model configuration
        self.model_name = "Qwen/Qwen-Image"
        self.pipeline = None
        
        # Model state tracking for caching (similar to ImageGenerator)
        self.current_lora_paths = []
        self.current_lora_scales = []
        self.current_quantization = None
        
        # LoRA configuration
        self.lora_data = self.get_lora_data()
        self.lora_directory = "lora"
        
        # Removed magic prompts - they were just basic keyword additions
        
        # Output directory
        self.output_dir = Path("outputimage")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🤖 QwenImageGenerator initialized on {self.device} with {self.dtype}")
    
    def _detect_device(self) -> str:
        """Detect optimal device for generation."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _get_optimal_dtype(self) -> torch.dtype:
        """Get optimal dtype for the current device."""
        if self.device == "mps":
            # MPS works best with bfloat16 for Qwen models
            return torch.bfloat16
        elif self.device == "cuda":
            # CUDA supports bfloat16 efficiently
            return torch.bfloat16
        else:
            # CPU fallback to float32
            return torch.float32
    
    def get_lora_data(self):
        """
        Get LoRA data from database (same as ImageGenerator).
        
        Returns:
            list: List of LoRA dictionaries compatible with existing code
        """
        try:
            from core.database import get_lora_for_image_generator
            return get_lora_for_image_generator()
        except Exception as e:
            print(f"⚠️  Error loading LoRA data from database: {e}")
            return []
    
    def refresh_lora_data(self):
        """
        Refresh LoRA data from database.
        This should be called after LoRA management operations.
        """
        try:
            from core.database import get_lora_for_image_generator
            self.lora_data = get_lora_for_image_generator()
            print(f"✅ LoRA data refreshed: {len(self.lora_data)} models loaded")
        except Exception as e:
            print(f"❌ Error refreshing LoRA data: {e}")
            # Keep existing data if refresh fails
    
    def load_pipeline(self, quantization: str = "None") -> bool:
        """
        Load the Qwen-Image pipeline with optional quantization.
        
        Args:
            quantization: Quantization method ("None", "8-bit", "4-bit")
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print("🔄 Loading Qwen-Image pipeline...")
            print(f"   Model: {self.model_name}")
            print(f"   Device: {self.device}")
            print(f"   Dtype: {self.dtype}")
            print(f"   Quantization: {quantization}")
            
            # Load pipeline - Try QwenImagePipeline first if available, otherwise use DiffusionPipeline
            if QWEN_PIPELINE_AVAILABLE:
                print("🔄 Loading Qwen-Image with QwenImagePipeline (dedicated class)")
                try:
                    self.pipeline = QwenImagePipeline.from_pretrained(
                        self.model_name,
                        torch_dtype=self.dtype
                    )
                    print("✅ QwenImagePipeline.from_pretrained() succeeded")
                except Exception as load_error:
                    print(f"❌ QwenImagePipeline.from_pretrained() failed: {load_error}")
                    print(f"   Trying fallback with DiffusionPipeline...")
                    try:
                        self.pipeline = DiffusionPipeline.from_pretrained(
                            self.model_name,
                            torch_dtype=self.dtype
                        )
                        print("✅ DiffusionPipeline fallback succeeded")
                    except Exception as fallback_error:
                        print(f"❌ Both methods failed. Final error: {fallback_error}")
                        raise fallback_error
            else:
                print("🔄 Loading Qwen-Image with DiffusionPipeline (QwenImagePipeline not available)")
                try:
                    self.pipeline = DiffusionPipeline.from_pretrained(
                        self.model_name,
                        torch_dtype=self.dtype
                    )
                    print("✅ DiffusionPipeline.from_pretrained() succeeded")
                except Exception as load_error:
                    print(f"❌ DiffusionPipeline.from_pretrained() failed: {load_error}")
                    raise load_error
            
            # Move to device
            print(f"🔄 Moving pipeline to {self.device}")
            self.pipeline = self.pipeline.to(self.device)
            
            # Store current quantization state
            self.current_quantization = quantization
            
            # Apply quantization if requested
            if quantization and quantization != "None":
                print(f"🔄 Applying {quantization} quantization...")
                try:
                    from utils.quantization import quantize_pipeline_components
                    if quantization in ["8-bit", "Auto"]:
                        success, error_msg = quantize_pipeline_components(
                            self.pipeline, 
                            device=self.device,
                            prefer_4bit=False,
                            verbose=True
                        )
                        if not success:
                            print(f"⚠️  8-bit quantization failed: {error_msg}")
                    elif quantization == "4-bit":
                        success, error_msg = quantize_pipeline_components(
                            self.pipeline, 
                            device=self.device,
                            prefer_4bit=True,
                            verbose=True
                        )
                        if not success:
                            print(f"⚠️  4-bit quantization failed: {error_msg}")
                    print(f"✅ {quantization} quantization applied successfully")
                except Exception as quant_error:
                    print(f"⚠️  Quantization {quantization} failed: {quant_error}")
                    print("   Continuing without quantization...")
            
            # Memory optimization for MPS
            if self.device == "mps":
                # Enable attention slicing for better memory usage
                self.pipeline.enable_attention_slicing()
                
            print(f"✅ Qwen-Image pipeline loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load Qwen-Image pipeline: {e}")
            return False
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: Optional[int] = None,
# Removed magic prompt parameters - they just added generic keywords
        lora_paths: List[str] = None,
        lora_scales: List[float] = None,
        quantization: str = "None",
        progress_callback: Optional[Any] = None
    ) -> Tuple[Optional[Image.Image], str]:
        """
        Generate an image using Qwen-Image model with optional LoRA support.
        
        Args:
            prompt: Text description of the desired image
            negative_prompt: What to avoid in the image
            width: Image width (default: 1024)
            height: Image height (default: 1024)
            num_inference_steps: Number of denoising steps (default: 50)
            guidance_scale: Guidance scale for generation (default: 4.0)
            seed: Random seed for reproducibility (None for random)
# Removed magic prompt documentation
            lora_paths: List of LoRA file paths to apply
            lora_scales: List of corresponding LoRA scales
            quantization: Quantization method ("None", "8-bit", "4-bit")
            progress_callback: Callback function for progress updates
            
        Returns:
            tuple: (PIL Image or None, status message)
        """
        
        # Initialize LoRA parameters
        if lora_paths is None:
            lora_paths = []
        if lora_scales is None:
            lora_scales = []
        
        print(f"🎯 QwenImageGenerator received LoRA paths: {lora_paths}")
        print(f"🎯 QwenImageGenerator received LoRA scales: {lora_scales}")
        
        # Check if we need to reload pipeline (LoRA or quantization changed)
        need_reload = (
            self.pipeline is None or 
            lora_paths != self.current_lora_paths or
            lora_scales != self.current_lora_scales or
            quantization != self.current_quantization
        )
        
        if need_reload:
            print("🔄 Pipeline reload needed due to LoRA/quantization changes")
            if not self.load_pipeline(quantization):
                return None, "❌ Failed to load Qwen-Image pipeline"
        
        # Apply LoRA weights if specified
        if lora_paths:
            try:
                print(f"🔄 Applying {len(lora_paths)} LoRA models...")
                
                # Unload any existing LoRA weights first
                if hasattr(self.pipeline, 'unload_lora_weights'):
                    self.pipeline.unload_lora_weights()
                
                # Load LoRA weights
                valid_paths = []
                valid_scales = []
                
                for i, (lora_path, scale) in enumerate(zip(lora_paths, lora_scales)):
                    if lora_path and Path(lora_path).exists() and scale > 0:
                        valid_paths.append(lora_path)
                        valid_scales.append(scale)
                        print(f"   LoRA {i+1}: {Path(lora_path).name} (scale: {scale})")
                
                if valid_paths:
                    if hasattr(self.pipeline, 'load_lora_weights'):
                        if len(valid_paths) == 1:
                            # Single LoRA
                            self.pipeline.load_lora_weights(valid_paths[0])
                            self.pipeline.set_adapters(["default"], adapter_weights=[valid_scales[0]])
                        else:
                            # Multiple LoRA - use adapter names
                            adapter_names = [f"lora_{i}" for i in range(len(valid_paths))]
                            for i, (path, name) in enumerate(zip(valid_paths, adapter_names)):
                                self.pipeline.load_lora_weights(path, adapter_name=name)
                            self.pipeline.set_adapters(adapter_names, adapter_weights=valid_scales)
                        
                        print(f"✅ {len(valid_paths)} LoRA models loaded successfully")
                    else:
                        print("⚠️  Pipeline does not support LoRA weights")
                        
                # Update current state
                self.current_lora_paths = lora_paths.copy()
                self.current_lora_scales = lora_scales.copy()
                
            except Exception as lora_error:
                print(f"⚠️  LoRA loading failed: {lora_error}")
                print("   Continuing without LoRA...")
                self.current_lora_paths = []
                self.current_lora_scales = []
        
        # Start timing for total generation time
        total_start_time = time.time()
        
        try:
            # Use prompt directly - removed magic prompt enhancement
            enhanced_prompt = prompt
            
            print(f"🎨 Generating image with Qwen-Image...")
            print(f"   Prompt: {enhanced_prompt[:60]}...")
            print(f"   Negative Prompt: {negative_prompt[:60] if negative_prompt else 'None'}...")
            print(f"   Size: {width}x{height}")
            print(f"   Steps: {num_inference_steps}")
            print(f"   True CFG Scale: {guidance_scale}")
            print(f"   Device: {self.device}")
            
            # Set up generator with seed
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
                print(f"   Seed: {seed}")
            
            # Memory cleanup before generation
            if self.device in ["cuda", "mps"]:
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                elif self.device == "mps":
                    torch.mps.empty_cache()
                gc.collect()
            
            # Generate image with Qwen-Image specific parameters
            # Prepare pipeline arguments
            pipeline_args = {
                "prompt": enhanced_prompt,
                "negative_prompt": negative_prompt if negative_prompt else None,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "true_cfg_scale": guidance_scale,  # Qwen-Image uses true_cfg_scale instead of guidance_scale
                "generator": generator
            }
            
            # Intercept tqdm progress if callback is provided
            if progress_callback is not None:
                # Try to intercept tqdm progress
                import sys
                from contextlib import redirect_stderr, redirect_stdout
                from io import StringIO
                
                # Monkey patch tqdm to call our callback
                original_tqdm = None
                try:
                    from tqdm import tqdm
                    original_tqdm = tqdm.__init__
                    
                    def tqdm_with_callback(self, iterable=None, *args, **kwargs):
                        # Call original init
                        result = original_tqdm(self, iterable, *args, **kwargs)
                        
                        # Store callback reference
                        self._progress_callback = progress_callback
                        
                        # Patch update method
                        original_update = self.update
                        def update_with_callback(n=1):
                            result = original_update(n)
                            if hasattr(self, '_progress_callback') and hasattr(self, 'n') and hasattr(self, 'total'):
                                if self.total and self.total > 0:
                                    # Also update the global progress tracker used by the UI
                                    try:
                                        from utils.progress_tracker import global_progress_tracker
                                        global_progress_tracker.on_step_update(
                                            step=self.n,
                                            total=self.total,
                                            desc=f"Qwen Step {self.n}/{self.total}"
                                        )
                                    except Exception as tracker_error:
                                        print(f"⚠️ Progress tracker error: {tracker_error}")
                                    
                                    # Call original callback
                                    self._progress_callback(self.n, 0, None)
                            return result
                        self.update = update_with_callback
                        return result
                    
                    # Apply monkey patch
                    tqdm.__init__ = tqdm_with_callback
                    
                    # Run pipeline with timing
                    model_start_time = time.time()
                    result = self.pipeline(**pipeline_args)
                    model_end_time = time.time()
                    model_generation_time = model_end_time - model_start_time
                    
                finally:
                    # Restore original tqdm
                    if original_tqdm is not None:
                        tqdm.__init__ = original_tqdm
            else:
                model_start_time = time.time()
                result = self.pipeline(**pipeline_args)
                model_end_time = time.time()
                model_generation_time = model_end_time - model_start_time
            
            image = result.images[0]
            
            # Calculate total generation time
            total_end_time = time.time()
            total_generation_time = total_end_time - total_start_time
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"qwen_image_{timestamp}.png"
            filepath = self.output_dir / filename
            
            image.save(filepath, "PNG")
            
            # Memory cleanup after generation
            if self.device in ["cuda", "mps"]:
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                elif self.device == "mps":
                    torch.mps.empty_cache()
                gc.collect()
            
            # Print timing information
            print(f"⏱️ Generation completed in {total_generation_time:.2f}s (model: {model_generation_time:.2f}s)")
            print(f"✅ Image generated successfully: {filename}")
            
            # Return image, status message, and timing information
            return image, f"✅ Image generated successfully: {filename}", {
                'total_generation_time': total_generation_time,
                'model_generation_time': model_generation_time
            }
            
        except Exception as e:
            error_msg = f"❌ Generation failed: {str(e)}"
            print(error_msg)
            return None, error_msg, None
    
    def unload_pipeline(self):
        """Unload pipeline and free memory."""
        if self.pipeline is not None:
            print("🔄 Unloading Qwen-Image pipeline...")
            del self.pipeline
            self.pipeline = None
            
            # Force memory cleanup
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()
            gc.collect()
            
            print("✅ Pipeline unloaded")
    
    def get_info(self) -> dict:
        """Get generator information."""
        return {
            "model": self.model_name,
            "device": self.device,
            "dtype": str(self.dtype),
            "loaded": self.pipeline is not None,
            "magic_prompts_removed": "Basic keyword additions removed"
        }


# Global instance
qwen_generator = None


def get_qwen_generator() -> QwenImageGenerator:
    """Get or create global QwenImageGenerator instance."""
    global qwen_generator
    if qwen_generator is None:
        qwen_generator = QwenImageGenerator()
    return qwen_generator