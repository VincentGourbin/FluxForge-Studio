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


# Device configuration (CPU, GPU, MPS for Apple Silicon)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Global variables for storing the currently loaded model state
current_model_alias = None
current_quantize = None
current_path = None
current_lora_paths = None
current_lora_scales = None
current_model_type = None  # 'standard' or 'controlnet'
flux_model = None

# Path to the directory containing LoRA files
lora_directory = 'lora'

# Path to the JSON file containing LoRA information
lora_json_file = 'lora_info.json'

def load_lora_data():
    """
    Load LoRA model information from the JSON configuration file.
    
    Returns:
        list: List of dictionaries containing LoRA model metadata
              (file_name, description, activation_keyword)
    """
    try:
        with open(lora_json_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {lora_json_file} not found. No LoRA models will be available.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing {lora_json_file}: {e}")
        return []

# Load LoRA information from JSON file
lora_data = load_lora_data()

# Model and quantization options
model_options = ["schnell", "dev"]
quantize_options = [4, 8, None]

def get_ollama_models():
    """
    Retrieve the list of available models from Ollama.
    
    Returns:
        tuple: (models_info: dict, model_names: list) containing model metadata
               and list of available model names
    """
    try:
        import ollama
        ollama_models = ollama.list()
        models_info = {}
        for model in ollama_models['models']:
            name = model['name']
            families = model['details'].get('families', [])
            models_info[name] = families
        model_names = list(models_info.keys())
        return models_info, model_names
    except Exception as e:
        print(f"Erreur lors de la récupération des modèles Ollama : {e}")
        return {}, []

# Get available Ollama models
models_info, model_names = get_ollama_models()