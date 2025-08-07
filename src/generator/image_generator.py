"""FLUX.1 Image Generation Module

This module provides the core image generation functionality for the diffusers-gradio application.
It handles FLUX.1 model loading, caching, and generation with LoRA support and ControlNet integration.

Key Features:
- FLUX.1 model initialization and caching for performance
- Dynamic LoRA (Low-Rank Adaptation) loading and integration
- ControlNet support for guided image generation
- Model state management to avoid unnecessary reloading
- Comprehensive parameter validation and processing
- Database integration for generation history tracking
- Multi-device support (MPS, CUDA, CPU)

Author: Vincent
License: MIT
"""

import os
import datetime
import random
import torch
import gc
import warnings
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Callable, Any
from PIL import Image

from core import config
from diffusers import FluxPipeline, FluxControlNetPipeline, FluxControlNetModel
try:
    from diffusers import FluxKontextPipeline
except ImportError:
    print("Warning: FluxKontextPipeline not available in this diffusers version")
    FluxKontextPipeline = None
from diffusers.utils import load_image
import gradio as gr
from utils.progress_tracker import global_progress_tracker

class ImageGenerator:
    """Core image generation class for FLUX.1 models with LoRA and ControlNet support.
    
    This class manages the lifecycle of FLUX.1 models using diffusers library, providing 
    efficient caching to avoid unnecessary model reloads. It supports both standard and 
    ControlNet-enabled generation modes, with dynamic LoRA integration for style adaptation.
    
    Attributes:
        current_model_alias (str): Currently loaded model identifier
        current_path (str): Path to local model if using custom model
        current_lora_paths (list): List of currently loaded LoRA file paths
        current_lora_scales (list): Corresponding LoRA influence scales
        current_model_type (str): Type of model loaded ('standard' or 'controlnet')
        flux_pipeline: Cached FLUX.1 pipeline instance
        device (str): Target device (mps, cuda, cpu)
        dtype: Model precision (bfloat16 or float32)
        lora_data (list): Available LoRA models metadata from config
        lora_directory (str): Directory containing LoRA files
        model_options (list): Available model aliases
    """
    
    def __init__(self):
        """Initialize the ImageGenerator with default state and configuration.
        
        Sets up model state tracking variables and loads configuration from
        the config module. All model-related variables start as None to
        indicate no model is currently loaded.
        """
        # Model state tracking variables for caching optimization
        self.current_model_alias = None
        self.current_path = None
        self.current_lora_paths = []
        self.current_lora_scales = []
        self.current_model_type = None
        self.current_quantization = None
        self.flux_pipeline = None
        
        # Device and precision configuration
        self.device = config.device
        # FLUX dtype configuration selon device
        # Tests confirment: float32 et bfloat16 fonctionnent sur MPS, float16 cause images noires
        if self.device == 'mps':
            self.dtype = torch.bfloat16  # Testé OK sur MPS, luminosité 189.0
        elif self.device == 'cuda':
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        
        # Configuration data loaded from config module
        self.lora_data = self.get_lora_data()  # Load from database instead of config
        self.lora_directory = config.lora_directory
        self.model_options = config.model_options
        self.controlnet_options = config.controlnet_options
        self.flux_tools_options = config.flux_tools_options
        self.post_processing_options = config.post_processing_options
    
    def get_lora_data(self):
        """
        Get LoRA data from database.
        
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

    def generate_image(
        self,
        prompt,
        model_alias,
        steps,
        seed,
        metadata,
        guidance,
        height,
        width,
        path,
        controlnet_type,
        controlnet_image_path,
        controlnet_strength,
        controlnet_save_canny,
        enable_stepwise=False,
        progress=gr.Progress(),
        canny_low_threshold=100,
        canny_high_threshold=200,
        upscaler_multiplier=2.0,
        flux_tools_type="None",
        flux_tools_image_path=None,
        flux_tools_guidance=2.5,
        post_processing_type="None",
        post_processing_image_path=None,
        post_processing_multiplier=2.0,
        quantization="None",
        *args
    ):
        """Generate an image using FLUX.1 model with optional LoRA and ControlNet support.
        
        This is the main image generation method that handles model loading, parameter
        processing, LoRA integration, and image generation using diffusers library.
        It includes intelligent model caching to avoid reloading when parameters haven't changed.
        
        Args:
            prompt (str): Text description of the desired image
            model_alias (str): Model identifier ('schnell', 'dev', etc.)
            steps (int): Number of inference steps for generation
            seed (int): Random seed for reproducible generation (0 for random)
            metadata (bool): Whether to embed generation metadata in output
            guidance (float): CFG guidance scale for generation control
            height (int): Output image height in pixels
            width (int): Output image width in pixels
            path (str): Optional path to local model directory
            controlnet_image_path: Input image for ControlNet guidance
            controlnet_strength (float): Strength of ControlNet influence (0.0-1.0)
            controlnet_save_canny (bool): Whether to save Canny edge detection result
            progress: Gradio progress tracker for UI updates
            *args: Variable arguments containing LoRA selections and scales
                  Format: [checkbox_values...] + [scale_values...]
        
        Returns:
            PIL.Image: Generated image ready for display
            
        Raises:
            ValueError: If parameters are invalid or model loading fails
            FileNotFoundError: If specified model or LoRA files don't exist
        """
        # Parameter validation and processing
        # Handle seed generation: 0 or None means generate random seed
        if seed is None or int(seed) == 0:
            seed = random.randint(1, 2**32 - 1)
        else:
            seed = int(seed)
    
        # Ensure image dimensions are integers
        height = int(height)
        width = int(width)
        
        # Ensure minimum of 1 inference step
        steps = max(1, int(steps))
        
        # Set default guidance value if not provided
        guidance = float(guidance) if guidance else 3.5
            
        # Convert ControlNet parameters to appropriate types
        controlnet_strength = float(controlnet_strength)
        controlnet_save_canny = bool(controlnet_save_canny)
    
        # Process LoRA selections and scales from variable arguments
        # Args format: [checkbox1, checkbox2, ...] + [scale1, scale2, ...]
        num_lora = len(self.lora_data)
        lora_checkbox_values = args[:num_lora]  # First half: selection checkboxes
        lora_scale_values = args[num_lora:2*num_lora]  # Second half: influence scales
    
        # Build lists of selected LoRA models and their scales
        lora_paths_list = []
        lora_scales_list = []
        for idx, (selected, scale) in enumerate(zip(lora_checkbox_values, lora_scale_values)):
            if selected:  # Only process selected LoRA models
                lora_info = self.lora_data[idx]
                lora_file = lora_info['file_name']
                lora_path = os.path.join(self.lora_directory, lora_file)
                lora_paths_list.append(lora_path)
                lora_scales_list.append(float(scale))
                
                # Prepend activation keyword to prompt for better LoRA activation
                # This ensures the LoRA's trained concepts are properly triggered
                prompt = f"{lora_info['activation_keyword']}, {prompt}"
    
        # Determine if ControlNet should be used based on type selection and input image
        controlnet_model_id = self.controlnet_options.get(controlnet_type)
        use_controlnet = (controlnet_model_id is not None and 
                         controlnet_image_path is not None)
        
        # Determine if FLUX Tools should be used based on type selection and input image
        flux_tools_model_id = self.flux_tools_options.get(flux_tools_type)
        use_flux_tools = (flux_tools_model_id is not None and 
                         flux_tools_image_path is not None)
    
        # Determine pipeline type based on usage
        current_pipeline_type = 'standard'
        if use_controlnet:
            current_pipeline_type = 'controlnet'
        elif use_flux_tools:
            current_pipeline_type = 'flux_tools'
    
        # Intelligent model caching: only reload if parameters have changed
        # This significantly improves performance by avoiding unnecessary model reloads
        model_needs_reload = (
            model_alias != self.current_model_alias or          # Different model
            path != self.current_path or                       # Different model path
            self.flux_pipeline is None or                      # No model loaded yet
            current_pipeline_type != self.current_model_type or # Pipeline type change
            quantization != self.current_quantization          # Quantization change
        )
        
        # Check if LoRA configuration has changed
        lora_needs_reload = (
            lora_paths_list != self.current_lora_paths or      # Different LoRA selection
            lora_scales_list != self.current_lora_scales        # Different LoRA scales
        )
        
        if model_needs_reload:
            # Map model alias to HuggingFace model ID
            model_id_map = {
                'schnell': 'black-forest-labs/FLUX.1-schnell',
                'dev': 'black-forest-labs/FLUX.1-dev',
                'krea-dev': 'black-forest-labs/FLUX.1-Krea-dev'
            }
            
            # Determine model ID
            if path:
                model_id = path  # Use local path if specified
            else:
                model_id = model_id_map.get(model_alias, 'black-forest-labs/FLUX.1-schnell')
    
            # Initialize the appropriate pipeline type based on usage
            if use_controlnet:
                # Load ControlNet model and pipeline
                controlnet = FluxControlNetModel.from_pretrained(
                    controlnet_model_id,
                    torch_dtype=self.dtype
                )
                flux_pipeline = FluxControlNetPipeline.from_pretrained(
                    model_id,
                    controlnet=controlnet,
                    torch_dtype=self.dtype,
                    use_safetensors=True
                )
                current_model_type = 'controlnet'
            elif use_flux_tools:
                # Load FLUX Tools pipeline (e.g., Kontext for image editing)
                if FluxKontextPipeline is None:
                    raise ValueError("FluxKontextPipeline not available. Please update diffusers library.")
                flux_pipeline = FluxKontextPipeline.from_pretrained(
                    flux_tools_model_id,
                    torch_dtype=self.dtype,
                    use_safetensors=True
                )
                current_model_type = 'flux_tools'
            else:
                # Load standard FLUX.1 pipeline for text-to-image generation
                flux_pipeline = FluxPipeline.from_pretrained(
                    model_id,
                    torch_dtype=self.dtype,
                    use_safetensors=True
                )
                current_model_type = 'standard'
    
            # Move pipeline to appropriate device
            flux_pipeline = flux_pipeline.to(self.device)
            
            # Enable memory efficient attention
            flux_pipeline.enable_attention_slicing()
            
            # IMPORTANT: Quantization sera appliquée APRÈS le chargement des LoRA
            # pour éviter les incompatibilités de noms de paramètres
            
            # Update cached model state
            self.current_model_alias = model_alias
            self.current_path = path
            self.current_model_type = current_model_type
            self.current_quantization = quantization
            self.flux_pipeline = flux_pipeline
            
            # Force LoRA reload since model was reloaded
            lora_needs_reload = True
        else:
            # Use cached pipeline
            flux_pipeline = self.flux_pipeline
            current_model_type = self.current_model_type
        
        # Handle LoRA loading/updating separately
        if lora_needs_reload:
            print(f"🔄 LoRA reload needed: {len(lora_paths_list)} LoRA(s) to load")
            
            # Unload existing LoRA if any - more aggressive cleanup
            try:
                flux_pipeline.unload_lora_weights()
                # Force memory cleanup after LoRA unload
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device == 'mps':
                    torch.mps.empty_cache()
                gc.collect()
                print("✅ Previous LoRA weights unloaded successfully")
            except Exception as e:
                print(f"Warning: LoRA unload failed: {e}")
            
            # Load new LoRA weights if any selected
            if lora_paths_list:
                adapter_names = []
                adapter_weights = []
                
                for lora_path, lora_scale in zip(lora_paths_list, lora_scales_list):
                    if os.path.exists(lora_path):
                        # For local files, use the directory path and weight_name
                        lora_dir = os.path.dirname(lora_path)
                        lora_filename = os.path.basename(lora_path)
                        adapter_name = os.path.splitext(lora_filename)[0].replace('.', '_')
                        
                        # Load LoRA with warning suppression and error handling
                        try:
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", message=".*CLIPTextModel.*")
                                warnings.filterwarnings("ignore", message=".*No LoRA keys associated to CLIPTextModel.*")
                                warnings.filterwarnings("ignore", message=".*Already found a.*peft_config.*attribute.*")
                                flux_pipeline.load_lora_weights(
                                    lora_dir, 
                                    weight_name=lora_filename,
                                    adapter_name=adapter_name
                                )
                                print(f"✅ LoRA chargé avec succès: {lora_filename}")
                        except KeyError as e:
                            print(f"❌ LoRA incompatible (paramètre manquant): {lora_filename}")
                            print(f"   Erreur: {e}")
                            print(f"   💡 Vérifiez que quantisation est appliquée APRÈS LoRA")
                            continue  # Skip this LoRA and continue with others
                        except Exception as e:
                            print(f"❌ Erreur chargement LoRA: {lora_filename}")
                            print(f"   Erreur: {e}")
                            continue  # Skip this LoRA and continue with others
                        
                        adapter_names.append(adapter_name)
                        adapter_weights.append(float(lora_scale))
                
                # Set adapter weights if any were loaded
                if adapter_names:
                    flux_pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
                    print(f"✅ {len(adapter_names)} LoRA adapter(s) activated with scales: {adapter_weights}")
            else:
                # No LoRA selected - ensure adapters are completely disabled
                print("📝 No LoRA selected - ensuring clean state")
                try:
                    # Try to explicitly disable any remaining adapters
                    flux_pipeline.disable_lora()
                except AttributeError:
                    # disable_lora might not exist in all diffusers versions
                    pass
                except Exception as e:
                    print(f"Warning: Could not disable LoRA adapters: {e}")
            
            # Update LoRA cache state
            self.current_lora_paths = lora_paths_list
            self.current_lora_scales = lora_scales_list
    
        # Apply quantization AFTER LoRA loading to avoid parameter name conflicts
        if quantization and quantization != "None" and model_alias in ["schnell", "dev"]:
            from utils.quantization import quantize_pipeline_components
            
            # Tests confirment: seul qint8 fonctionne de manière stable
            if quantization in ["8-bit", "Auto"]:
                print(f"🔧 Application quantification qint8 FLUX {model_alias} APRÈS LoRA")
                success, error = quantize_pipeline_components(flux_pipeline, self.device, prefer_4bit=False, verbose=True)
                if not success:
                    print(f"⚠️  Quantification qint8 échouée: {error}")
                    print("🔄 Continuons sans quantification...")
            elif quantization == "4-bit":
                print(f"⚠️  Quantification 4-bit non supportée sur {self.device} (tests montrent erreurs)")
                print("💡 Conseil: Utilisez '8-bit' pour économie mémoire")
                print("🔄 Continuons sans quantification...")
            else:
                print(f"⚠️  Quantification {quantization} non supportée")
                print("🔄 Continuons sans quantification...")
        elif quantization and quantization != "None" and model_alias not in ["schnell", "dev", "krea-dev"]:
            print(f"⚠️  Quantification disponible pour FLUX Schnell, Dev et Krea-dev uniquement")
            print("🔄 Continuons sans quantification...")

        # Generate timestamp for file naming and database storage
        timestamp = datetime.datetime.now()
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
    
        # Create output directory and generate unique filename
        # Format: YYYYMMDD_HHMMSS_SEED.png for easy sorting and identification
        output_dir = Path("outputimage")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = output_dir / f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{seed}.png"
        
        # Set up generator for reproducible results
        generator = torch.Generator(device=self.device).manual_seed(seed)
            
        # Generate image using the appropriate method based on ControlNet usage
        if use_controlnet and controlnet_image_path:
            # Load and process control image (controlnet_image_path is already a file path from Gradio)
            control_image = load_image(controlnet_image_path)
            
            # Apply appropriate preprocessing based on ControlNet type
            control_image, target_dimensions = self._preprocess_controlnet_image(control_image, controlnet_type, canny_low_threshold, canny_high_threshold)
            
            # Set generation parameters for ControlNet
            target_width, target_height = width, height
            generation_prompt = prompt
            conditioning_scale = controlnet_strength
            generation_steps = steps
            generation_guidance = guidance
            
            # Save Canny edge image if requested and it's a Canny ControlNet
            if controlnet_save_canny and "Canny" in controlnet_type:
                canny_output_dir = Path("outputimage")
                canny_output_dir.mkdir(parents=True, exist_ok=True)
                canny_filename = canny_output_dir / f"canny_{timestamp.strftime('%Y%m%d_%H%M%S')}_{seed}.png"
                control_image.save(str(canny_filename))
            
            # ControlNet-guided generation using preprocessed image
            image = flux_pipeline(
                prompt=generation_prompt,
                control_image=control_image,
                controlnet_conditioning_scale=conditioning_scale,
                num_inference_steps=generation_steps,
                guidance_scale=generation_guidance,
                height=target_height,
                width=target_width,
                generator=generator
            ).images[0]
        elif use_flux_tools and flux_tools_image_path:
            # Load input image for FLUX Tools
            input_image = load_image(flux_tools_image_path)
            
            # FLUX Tools (Kontext) generation using input image and edit prompt
            image = flux_pipeline(
                image=input_image,
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=flux_tools_guidance,
                height=height,
                width=width,
                generator=generator
            ).images[0]
        else:
            # Standard text-to-image generation with progress tracking
            print(f"🎨 Starting {model_alias} generation with progress tracking...")
            
            # Apply progress tracking for dev, schnell and krea-dev models
            if model_alias in ["dev", "schnell", "krea-dev"]:
                # Reset and start progress tracking
                global_progress_tracker.reset()
                global_progress_tracker.apply_tqdm_patches()
                
                try:
                    image = flux_pipeline(
                        prompt=prompt,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        height=height,
                        width=width,
                        generator=generator
                    ).images[0]
                finally:
                    # Always restore patches after generation
                    global_progress_tracker.remove_tqdm_patches()
                    print(f"✅ {model_alias} generation completed with progress tracking")
            else:
                # Standard generation without progress tracking for other models
                image = flux_pipeline(
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    height=height,
                    width=width,
                    generator=generator
                ).images[0]

        # Save the generated image to disk
        image.save(str(output_filename))
        
        # Clean up memory after generation
        self._cleanup_memory()
    
        # Store generation parameters and results in database for history tracking
        if use_controlnet:
            from core.database import save_controlnet_generation
            save_controlnet_generation(
                timestamp_str,
                seed,
                prompt,
                model_alias,
                controlnet_type,
                controlnet_strength,
                steps,
                guidance,
                height,
                width,
                lora_paths_list,
                lora_scales_list,
                str(output_filename)
            )
        else:
            from core.database import save_standard_generation
            save_standard_generation(
                timestamp_str,
                seed,
                prompt,
                model_alias,
                steps,
                guidance,
                height,
                width,
                lora_paths_list,
                lora_scales_list,
                str(output_filename)
            )
    
        # Return PIL Image object for Gradio display
        return image

    def _preprocess_canny(self, img: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
        """
        Preprocess image for Canny ControlNet by generating edge detection.
        
        This function converts the input image to Canny edges using OpenCV,
        which is required for proper ControlNet Canny functionality.
        
        Args:
            img (PIL.Image): Input image to process
            low_threshold (int): Lower threshold for edge detection (default: 100)
            high_threshold (int): Higher threshold for edge detection (default: 200)
            
        Returns:
            PIL.Image: Black and white edge detection image suitable for Canny ControlNet
        """
        # Convert PIL image to numpy array
        image_array = np.array(img)
        
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection with configurable thresholds
        canny_edges = cv2.Canny(image_array, low_threshold, high_threshold)
        
        # Convert single channel to 3 channels (RGB)
        canny_rgb = np.stack([canny_edges, canny_edges, canny_edges], axis=2)
        
        # Convert back to PIL Image
        return Image.fromarray(canny_rgb)

    def _preprocess_controlnet_image(self, control_image: Image.Image, controlnet_type: str, canny_low_threshold: int = 100, canny_high_threshold: int = 200) -> tuple[Image.Image, tuple[int, int]]:
        """
        Preprocess control image based on ControlNet type.
        
        Different ControlNet types require different preprocessing:
        - Canny: Edge detection preprocessing
        - Others: No preprocessing
        
        Args:
            control_image (PIL.Image): Input control image
            controlnet_type (str): Type of ControlNet being used
            
        Returns:
            tuple: (processed_image, target_dimensions) - Preprocessed image and target output size
        """
        if "Canny" in controlnet_type:
            # Apply Canny edge detection preprocessing with configurable thresholds
            processed_image = self._preprocess_canny(control_image, canny_low_threshold, canny_high_threshold)
            return processed_image, control_image.size
        else:
            # For other ControlNet types, return image as-is
            return control_image, control_image.size

    def _cleanup_memory(self):
        """Clean up GPU/MPS memory after generation to prevent accumulation."""
        try:
            # Clear PyTorch cache
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif self.device == 'mps':
                torch.mps.empty_cache()
                torch.mps.synchronize()
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            # Memory cleanup should not break generation, just log
            print(f"Warning: Memory cleanup failed: {e}")

    def update_guidance_visibility(self, model_alias):
        """Update UI visibility of guidance parameter based on selected model.
        
        The guidance parameter is only relevant for certain FLUX.1 model variants.
        This method controls when the guidance slider should be visible in the UI.
        
        Args:
            model_alias (str): The selected model identifier
            
        Returns:
            gr.update: Gradio update object to show/hide the guidance control
            
        Note:
            - 'dev' model supports guidance control (CFG)
            - Other models like 'schnell' use fixed guidance internally
        """
        if model_alias in ["dev", "krea-dev"]:
            # Show guidance slider for dev and krea-dev models (supports CFG)
            return gr.update(visible=True)
        else:
            # Hide guidance slider for other models
            return gr.update(visible=False)
    
    def update_steps_for_model(self, model_alias):
        """Update default steps based on selected model.
        
        Different FLUX models have different optimal step counts:
        - schnell: Optimized for 4 steps (speed)
        - dev: Better quality with 25 steps
        
        Args:
            model_alias (str): The selected model identifier
            
        Returns:
            gr.update: Gradio update object to set the steps value
        """
        if model_alias in ["dev", "krea-dev"]:
            # Dev and krea-dev models work better with more steps
            return gr.update(value=25)
        else:
            # Schnell is optimized for fewer steps
            return gr.update(value=4)