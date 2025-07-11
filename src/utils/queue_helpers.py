"""
Queue Helper Functions

This module provides helper functions to add different types of generation tasks
to the processing queue. These functions serve as adapters between the UI components
and the queue system.

Features:
- Helper functions for all generation types
- Parameter validation and formatting
- Queue integration with proper task descriptions

Author: FluxForge Team
License: MIT
"""

from typing import Dict, Any, Optional
from core.processing_queue import processing_queue, TaskType

def queue_standard_generation(
    prompt: str,
    model_alias: str,
    quantization: str,
    steps: int,
    seed: int,
    metadata: bool,
    guidance: float,
    height: int,
    width: int,
    lora_state: Any,
    lora_strength_1: float,
    lora_strength_2: float,
    lora_strength_3: float
) -> str:
    """Add a standard image generation task to the queue."""
    
    parameters = {
        'prompt': prompt,
        'model_alias': model_alias,
        'quantization': quantization,
        'steps': steps,
        'seed': seed,
        'metadata': metadata,
        'guidance': guidance,
        'height': height,
        'width': width,
        'lora_state': lora_state,
        'lora_strength_1': lora_strength_1,
        'lora_strength_2': lora_strength_2,
        'lora_strength_3': lora_strength_3
    }
    
    # Generate description
    model_part = f"Model: {model_alias}"
    quant_part = f"Quantization: {quantization}" if quantization != "None" else ""
    size_part = f"Size: {width}x{height}"
    description_parts = [model_part, quant_part, size_part]
    description = f"Generate: {prompt[:30]}... ({', '.join(filter(None, description_parts))})"
    
    task_id = processing_queue.add_task(TaskType.STANDARD_GENERATION, parameters, description)
    
    return f"✅ Added to queue: {description[:50]}... (ID: {task_id[:8]})"

def queue_flux_fill(
    fill_mode: str,
    image_editor_data: Any,
    outpaint_image: Any,
    prompt: str,
    steps: int,
    guidance_scale: float,
    quantization: str,
    top_percent: float,
    bottom_percent: float,
    left_percent: float,
    right_percent: float,
    lora_state: Any,
    lora_strength_1: float,
    lora_strength_2: float,
    lora_strength_3: float
) -> str:
    """Add a FLUX Fill task to the queue."""
    
    parameters = {
        'fill_mode': fill_mode,
        'image_editor_data': image_editor_data,
        'outpaint_image': outpaint_image,
        'prompt': prompt,
        'steps': steps,
        'guidance_scale': guidance_scale,
        'quantization': quantization,
        'top_percent': top_percent,
        'bottom_percent': bottom_percent,
        'left_percent': left_percent,
        'right_percent': right_percent,
        'lora_state': lora_state,
        'lora_strength_1': lora_strength_1,
        'lora_strength_2': lora_strength_2,
        'lora_strength_3': lora_strength_3
    }
    
    description = f"FLUX Fill ({fill_mode}): {prompt[:30]}... (Steps: {steps}, Guidance: {guidance_scale})"
    
    task_id = processing_queue.add_task(TaskType.FLUX_FILL, parameters, description)
    
    return f"✅ Added to queue: {description[:50]}... (ID: {task_id[:8]})"

def queue_kontext(
    input_image: Any,
    prompt: str,
    steps: int,
    guidance_scale: float,
    quantization: str,
    lora_state: Any,
    lora_strength_1: float,
    lora_strength_2: float,
    lora_strength_3: float
) -> str:
    """Add a Kontext task to the queue."""
    
    parameters = {
        'input_image': input_image,
        'prompt': prompt,
        'steps': steps,
        'guidance_scale': guidance_scale,
        'quantization': quantization,
        'lora_state': lora_state,
        'lora_strength_1': lora_strength_1,
        'lora_strength_2': lora_strength_2,
        'lora_strength_3': lora_strength_3
    }
    
    description = f"Kontext Edit: {prompt[:40]}... (Steps: {steps}, Guidance: {guidance_scale})"
    
    task_id = processing_queue.add_task(TaskType.KONTEXT, parameters, description)
    
    return f"✅ Added to queue: {description[:50]}... (ID: {task_id[:8]})"

def queue_flux_depth(
    input_image: Any,
    prompt: str,
    steps: int,
    guidance_scale: float,
    quantization: str,
    lora_state: Any,
    lora_strength_1: float,
    lora_strength_2: float,
    lora_strength_3: float
) -> str:
    """Add a FLUX Depth task to the queue."""
    
    parameters = {
        'input_image': input_image,
        'prompt': prompt,
        'steps': steps,
        'guidance_scale': guidance_scale,
        'quantization': quantization,
        'lora_state': lora_state,
        'lora_strength_1': lora_strength_1,
        'lora_strength_2': lora_strength_2,
        'lora_strength_3': lora_strength_3
    }
    
    description = f"FLUX Depth: {prompt[:40]}... (Steps: {steps}, Guidance: {guidance_scale})"
    
    task_id = processing_queue.add_task(TaskType.FLUX_DEPTH, parameters, description)
    
    return f"✅ Added to queue: {description[:50]}... (ID: {task_id[:8]})"

def queue_flux_canny(
    input_image: Any,
    prompt: str,
    steps: int,
    guidance_scale: float,
    quantization: str,
    low_threshold: int,
    high_threshold: int,
    lora_state: Any,
    lora_strength_1: float,
    lora_strength_2: float,
    lora_strength_3: float
) -> str:
    """Add a FLUX Canny task to the queue."""
    
    parameters = {
        'input_image': input_image,
        'prompt': prompt,
        'steps': steps,
        'guidance_scale': guidance_scale,
        'quantization': quantization,
        'low_threshold': low_threshold,
        'high_threshold': high_threshold,
        'lora_state': lora_state,
        'lora_strength_1': lora_strength_1,
        'lora_strength_2': lora_strength_2,
        'lora_strength_3': lora_strength_3
    }
    
    description = f"FLUX Canny: {prompt[:40]}... (Thresholds: {low_threshold}-{high_threshold})"
    
    task_id = processing_queue.add_task(TaskType.FLUX_CANNY, parameters, description)
    
    return f"✅ Added to queue: {description[:50]}... (ID: {task_id[:8]})"

def queue_flux_redux(
    input_image: Any,
    guidance_scale: float,
    steps: int,
    variation_strength: float,
    quantization: str
) -> str:
    """Add a FLUX Redux task to the queue."""
    
    parameters = {
        'input_image': input_image,
        'guidance_scale': guidance_scale,
        'steps': steps,
        'variation_strength': variation_strength,
        'quantization': quantization
    }
    
    description = f"FLUX Redux: Image variation (Strength: {variation_strength}, Steps: {steps})"
    
    task_id = processing_queue.add_task(TaskType.FLUX_REDUX, parameters, description)
    
    return f"✅ Added to queue: {description[:50]}... (ID: {task_id[:8]})"

def queue_background_removal(input_image: Any) -> str:
    """Add a background removal task to the queue."""
    
    parameters = {
        'input_image': input_image
    }
    
    description = "Background Removal: Remove background from image"
    
    task_id = processing_queue.add_task(TaskType.BACKGROUND_REMOVAL, parameters, description)
    
    return f"✅ Added to queue: {description} (ID: {task_id[:8]})"

def queue_upscaling(input_image: Any, upscale_factor: float, quantization: str = "None") -> str:
    """Add an upscaling task to the queue."""
    
    parameters = {
        'input_image': input_image,
        'upscale_factor': upscale_factor,
        'quantization': quantization
    }
    
    quant_part = f"Quantization: {quantization}" if quantization != "None" else ""
    description_parts = [f"{upscale_factor}x scale enhancement", quant_part]
    description = f"Upscaling: {', '.join(filter(None, description_parts))}"
    
    task_id = processing_queue.add_task(TaskType.UPSCALING, parameters, description)
    
    return f"✅ Added to queue: {description} (ID: {task_id[:8]})"