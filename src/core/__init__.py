"""
Core Components Module

Contains fundamental components for the MFLUX-Gradio application:
- Configuration management and device setup
- Database operations for image history
- LoRA metadata loading and management

Key Components:
- config: Global configuration, device detection, LoRA data
- database: SQLite operations for generation history
"""

from .config import *
from .database import *