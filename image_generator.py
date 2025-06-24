"""FLUX.1 Image Generation Module

This module provides the core image generation functionality for the mflux-gradio application.
It handles FLUX.1 model loading, caching, and generation with LoRA support and ControlNet integration.

Key Features:
- FLUX.1 model initialization and caching for performance
- Dynamic LoRA (Low-Rank Adaptation) loading and integration
- ControlNet support for guided image generation
- Model state management to avoid unnecessary reloading
- Comprehensive parameter validation and processing
- Database integration for generation history tracking

Author: Vincent
License: MIT
"""

import os
import datetime
import random
import json
from pathlib import Path

import config
from database import save_image_info
from mflux import Flux1, Flux1Controlnet, Config, ModelConfig
import gradio as gr

class ImageGenerator:
    """Core image generation class for FLUX.1 models with LoRA and ControlNet support.
    
    This class manages the lifecycle of FLUX.1 models, providing efficient caching
    to avoid unnecessary model reloads. It supports both standard and ControlNet-enabled
    generation modes, with dynamic LoRA integration for style adaptation.
    
    Attributes:
        current_model_alias (str): Currently loaded model identifier
        current_quantize (int): Current quantization level (4, 8, or None)
        current_path (str): Path to local model if using custom model
        current_lora_paths (list): List of currently loaded LoRA file paths
        current_lora_scales (list): Corresponding LoRA influence scales
        current_model_type (str): Type of model loaded ('standard' or 'controlnet')
        flux_model: Cached FLUX.1 model instance
        lora_data (list): Available LoRA models metadata from config
        lora_directory (str): Directory containing LoRA files
        model_options (list): Available model aliases
        quantize_options (list): Available quantization levels
    """
    
    def __init__(self):
        """Initialize the ImageGenerator with default state and configuration.
        
        Sets up model state tracking variables and loads configuration from
        the config module. All model-related variables start as None to
        indicate no model is currently loaded.
        """
        # Model state tracking variables for caching optimization
        self.current_model_alias = None
        self.current_quantize = None
        self.current_path = None
        self.current_lora_paths = []
        self.current_lora_scales = []
        self.current_model_type = None
        self.flux_model = None
        
        # Configuration data loaded from config module
        self.lora_data = config.lora_data
        self.lora_directory = config.lora_directory
        self.model_options = config.model_options
        self.quantize_options = config.quantize_options

    def generate_image(
        self,
        prompt,
        model_alias,
        quantize,
        steps,
        seed,
        metadata,
        guidance,
        height,
        width,
        path,
        controlnet_image_path,
        controlnet_strength,
        controlnet_save_canny,
        progress=gr.Progress(),
        *args
    ):
        """Generate an image using FLUX.1 model with optional LoRA and ControlNet support.
        
        This is the main image generation method that handles model loading, parameter
        processing, LoRA integration, and image generation. It includes intelligent
        model caching to avoid reloading when parameters haven't changed.
        
        Args:
            prompt (str): Text description of the desired image
            model_alias (str): Model identifier ('schnell', 'dev', etc.)
            quantize (int|str): Quantization level (4, 8, or None/"None")
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
        # Create directory for intermediate step images (used for debugging/visualization)
        stepwise_dir = Path("stepwise_output")
        stepwise_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean up previous intermediate images to avoid clutter
        for file in stepwise_dir.glob("*.png"):
            try:
                os.remove(file)
            except Exception:
                # Ignore errors during cleanup (file may be in use)
                pass

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
        
        # Handle quantization: convert "None" string to None type
        if quantize == 'None' or quantize is None:
            quantize = None
        else:
            quantize = int(quantize)
            
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
    
        # Determine if ControlNet should be used based on input image presence
        use_controlnet = controlnet_image_path is not None
    
        # Intelligent model caching: only reload if parameters have changed
        # This significantly improves performance by avoiding unnecessary model reloads
        model_needs_reload = (
            model_alias != self.current_model_alias or          # Different model
            quantize != self.current_quantize or               # Different quantization
            path != self.current_path or                       # Different model path
            lora_paths_list != self.current_lora_paths or      # Different LoRA selection
            lora_scales_list != self.current_lora_scales or    # Different LoRA scales
            self.flux_model is None or                         # No model loaded yet
            use_controlnet != (self.current_model_type == 'controlnet')  # ControlNet mode change
        )
        
        if model_needs_reload:
    
            # Load model configuration from local path or remote repository
            if path:
                # Use local model directory if specified
                model_config = ModelConfig.from_pretrained(path=path)
            else:
                # Use standard model alias (e.g., 'schnell', 'dev')
                model_config = ModelConfig.from_name(model_name=model_alias, base_model=None)
    
            # Initialize the appropriate model type based on ControlNet usage
            if use_controlnet:
                # Load ControlNet-enabled model for guided generation
                flux_model = Flux1Controlnet(
                    model_config=model_config,
                    quantize=quantize,
                    local_path=path,
                    lora_paths=lora_paths_list,
                    lora_scales=lora_scales_list,
                )
                current_model_type = 'controlnet'
            else:
                # Load standard FLUX.1 model for text-to-image generation
                flux_model = Flux1(
                    model_config=model_config,
                    quantize=quantize,
                    local_path=path,
                    lora_paths=lora_paths_list,
                    lora_scales=lora_scales_list,
                )
                current_model_type = 'standard'
    
            # Update cached model state for future comparisons
            self.current_model_alias = model_alias
            self.current_quantize = quantize
            self.current_path = path
            self.current_lora_paths = lora_paths_list
            self.current_lora_scales = lora_scales_list
            self.current_model_type = current_model_type
            self.flux_model = flux_model
        else:
            # Use cached model if no parameters have changed (performance optimization)
            flux_model = self.flux_model
            current_model_type = self.current_model_type
    
        # Build generation configuration object
        # ControlNet strength is only applied when ControlNet is being used
        config_obj = Config(
            num_inference_steps=steps,
            guidance=guidance,
            height=height,
            width=width,
            controlnet_strength=controlnet_strength if use_controlnet else None,
        )
        # Generate timestamp for file naming and database storage
        timestamp = datetime.datetime.now()
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
    
        # Create output directory and generate unique filename
        # Format: YYYYMMDD_HHMMSS_SEED.png for easy sorting and identification
        output_dir = Path("outputimage")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = output_dir / f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{seed}.png"
        # Generate image using the appropriate method based on ControlNet usage
        if use_controlnet:
            # ControlNet-guided generation using reference image
            image = flux_model.generate_image(
                seed=seed,
                prompt=prompt,
                controlnet_image_path=controlnet_image_path.name,
                config=config_obj
            )
        else:
            # Standard text-to-image generation
            image = flux_model.generate_image(
                seed=seed,
                prompt=prompt,
                config=config_obj
            )

        # Save the generated image to disk with optional metadata embedding
        image.save(path=str(output_filename), export_json_metadata=metadata)
    
        # Store generation parameters and results in database for history tracking
        # LoRA paths and scales are serialized as JSON for database storage
        save_image_info((
            timestamp_str,
            seed,
            prompt,
            model_alias,
            quantize,
            steps,
            guidance,
            height,
            width,
            path,
            controlnet_image_path.name if controlnet_image_path else None,
            controlnet_strength,
            controlnet_save_canny,
            json.dumps(lora_paths_list),    # Serialize LoRA paths as JSON
            json.dumps(lora_scales_list),   # Serialize LoRA scales as JSON
            str(output_filename)
        ))
    
        # Return PIL Image object for Gradio display
        return image.image

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
        if model_alias == "dev":
            # Show guidance slider for dev model (supports CFG)
            return gr.update(visible=True)
        else:
            # Hide guidance slider for other models
            return gr.update(visible=False)