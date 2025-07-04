"""
Image Generation Module

Handles FLUX.1 model management and image generation with advanced features:
- Model loading and caching (FLUX.1-schnell, FLUX.1-dev)
- LoRA integration and dynamic loading
- ControlNet support for guided generation
- Memory management and optimization

Key Components:
- ImageGenerator: Main generation class with caching and LoRA support
"""

from .image_generator import ImageGenerator