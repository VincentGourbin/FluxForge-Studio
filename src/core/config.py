# config.py
"""
Configuration module for mflux-gradio application.
Handles global settings, device configuration, LoRA data loading, and Ollama model discovery.
"""

import os
# Enable MPS fallback for Apple Silicon compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import json
import torch


# Device configuration (MPS for Apple Silicon, CUDA for NVIDIA GPUs, CPU fallback)
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Global variables for storing the currently loaded model state (deprecated - now handled in ImageGenerator)
current_model_alias = None
current_path = None
current_lora_paths = None
current_lora_scales = None
current_model_type = None  # 'standard' or 'controlnet'
flux_pipeline = None

# Path to the directory containing LoRA files
lora_directory = 'lora'

# LoRA data is now managed in the database - see database.py for LoRA functions

# Model options (no quantization needed with diffusers)
model_options = ["schnell", "dev", "krea-dev", "qwen-image"]

# ControlNet options
controlnet_options = {
    "None": None,
    "Canny (InstantX) - Edge detection": "InstantX/FLUX.1-dev-controlnet-canny"
}

# FLUX Tools options
flux_tools_options = {
    "None": None,
    "Kontext (Black Forest Labs) - Text-based image editing": "black-forest-labs/FLUX.1-Kontext-dev"
}

# Post Processing options
post_processing_options = {
    "None": None,
    "Upscaler (JasperAI) - Resolution enhancement": "jasperai/Flux.1-dev-Controlnet-Upscaler",
    "Background Remover - Background removal": "background_remover",
    "FLUX Fill Tools - Inpainting and Outpainting": "flux_fill_tools"
}

def get_ollama_models():
    """
    Retrieve the list of available models from Ollama.
    
    Returns:
        tuple: (models_info: dict, model_names: list) containing model metadata
               and list of available model names
    """
    try:
        import ollama
        ollama_response = ollama.list()
        models_info = {}
        
        # Handle different response formats (newer versions might return object with .models attribute)
        if hasattr(ollama_response, 'models'):
            models_list = ollama_response.models
        elif isinstance(ollama_response, dict) and 'models' in ollama_response:
            models_list = ollama_response['models']
        else:
            # Fallback: treat the response directly as models list
            models_list = ollama_response if isinstance(ollama_response, list) else []
        
        for model in models_list:
            # Handle different model object formats
            name = None
            if hasattr(model, 'model'):  # New ollama format uses 'model' attribute
                name = model.model
            elif hasattr(model, 'name'):  # Legacy format
                name = model.name
            elif isinstance(model, dict):
                name = model.get('name', '') or model.get('model', '')
            
            if not name:
                continue
            
            # Get detailed model information using ollama.show()
            try:
                model_details = ollama.show(name)
                capabilities = model_details.get('capabilities', [])
                
                # Store capabilities instead of families
                # We're particularly interested in 'vision' capability
                models_info[name] = capabilities
                
            except Exception as e:
                # Fallback to families if show() fails
                families = []
                if hasattr(model, 'details') and hasattr(model.details, 'families'):
                    families = model.details.families
                models_info[name] = families
        
        model_names = list(models_info.keys())
        return models_info, model_names
    except Exception as e:
        print(f"Error retrieving Ollama models: {e}")
        return {}, []

# Model cache and download settings
USE_OFFLINE_MODE = False  # Set to True to use only cached models
PREDOWNLOAD_MODELS = False  # Set to True to pre-download all models on startup
CACHE_DIRECTORY = None  # Use default HuggingFace cache or specify custom path

# Get available Ollama models
models_info, model_names = get_ollama_models()