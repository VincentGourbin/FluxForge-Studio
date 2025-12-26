"""
Queue Helper Functions

This module provides helper functions to add different types of generation tasks
to the processing queue. These functions serve as adapters between the UI components
and the queue system.

Features:
- Helper functions for all generation types
- Parameter validation and formatting
- Queue integration with proper task descriptions
- Deep copy of LoRA parameters to prevent UI state interference

Author: FluxForge Team
License: MIT
"""

import copy
from typing import Dict, Any, Optional
from core.processing_queue import processing_queue, TaskType

def queue_standard_generation(
    prompt: str,
    negative_prompt: str,
    model_alias: str,
    quantization: str,
    steps: int,
    seed: int,
    guidance: float,
    height: int,
    width: int,
    lora_state: Any,
    lora_strength_1: float,
    lora_strength_2: float,
    lora_strength_3: float
) -> str:
    """Add a standard image generation task to the queue (FLUX models)."""

    parameters = {
        'prompt': prompt,
        'model_alias': model_alias,
        'quantization': quantization,
        'steps': steps,
        'seed': seed,
        'metadata': True,  # Always save metadata
        'guidance': guidance,
        'height': height,
        'width': width,
        'lora_state': copy.deepcopy(lora_state),  # Deep copy to prevent UI interference
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
        'lora_state': copy.deepcopy(lora_state),  # Deep copy to prevent UI interference
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
        'lora_state': copy.deepcopy(lora_state),  # Deep copy to prevent UI interference
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
        'lora_state': copy.deepcopy(lora_state),  # Deep copy to prevent UI interference
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
        'lora_state': copy.deepcopy(lora_state),  # Deep copy to prevent UI interference
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


def ensure_multiangles_lora() -> tuple:
    """Ensure Multi-Angles LoRA is downloaded and registered.

    Downloads the LoRA from HuggingFace if not present locally,
    and registers it in the database if not already registered.

    Returns:
        tuple: (success: bool, message: str, lora_filename: str)
    """
    import os
    from pathlib import Path

    MULTIANGLES_LORA = "flux-multi-angles-v2-72poses-diffusers.safetensors"
    MULTIANGLES_REPO = "VincentGOURBIN/flux_qint_8bit"
    LORA_DIR = Path("lora")

    # Ensure lora directory exists
    LORA_DIR.mkdir(exist_ok=True)

    lora_path = LORA_DIR / MULTIANGLES_LORA

    # Step 1: Check if LoRA file exists locally
    if not lora_path.exists():
        print(f"📥 Multi-Angles LoRA not found locally. Downloading from HuggingFace...")
        try:
            from huggingface_hub import hf_hub_download

            # Download the LoRA file
            downloaded_path = hf_hub_download(
                repo_id=MULTIANGLES_REPO,
                filename=MULTIANGLES_LORA,
                local_dir=str(LORA_DIR)
            )
            print(f"✅ Downloaded Multi-Angles LoRA to {lora_path}")

        except Exception as e:
            error_msg = f"Failed to download Multi-Angles LoRA: {e}"
            print(f"❌ {error_msg}")
            return False, error_msg, MULTIANGLES_LORA

    # Step 2: Check if LoRA is registered in database
    try:
        from core.database import get_all_lora, add_lora

        all_loras = get_all_lora()
        lora_registered = any(l['file_name'] == MULTIANGLES_LORA for l in all_loras)

        if not lora_registered:
            print(f"📝 Registering Multi-Angles LoRA in database...")
            success, msg, lora_id = add_lora(
                file_name=MULTIANGLES_LORA,
                description="Multi-Angles LoRA v2 - 72 camera positions (auto-installed)",
                activation_keyword="<sks>",
                compatible_models=["flux2-dev"]
            )
            if success:
                print(f"✅ Multi-Angles LoRA registered in database (ID: {lora_id})")
            else:
                print(f"⚠️ Could not register LoRA: {msg}")

    except Exception as e:
        print(f"⚠️ Database registration check failed: {e}")

    return True, "Multi-Angles LoRA ready", MULTIANGLES_LORA


def queue_multiangles_generation(
    composed_prompt: str,
    reference_image: Any,
    steps: int,
    lora_scale: float,
    width: int,
    height: int,
    seed: int,
    quantization: str = "qint8"
) -> str:
    """Add a Multi-Angles generation task using FLUX.2 with Multi-Angles LoRA.

    This function queues a FLUX.2 generation with the Multi-Angles LoRA
    pre-configured for controlled camera angle generation.
    The LoRA is automatically downloaded and registered if not present.

    Args:
        composed_prompt: Full prompt with <sks> token and angle descriptors
        reference_image: Reference image for the subject
        steps: Number of inference steps
        lora_scale: LoRA influence scale (0.5-1.0 recommended)
        width: Output image width
        height: Output image height
        seed: Random seed (0 for random)
        quantization: Quantization mode for FLUX.2 ("qint8" or "full")

    Returns:
        str: Queue feedback message
    """
    # Check if reference image is provided (required for Multi-Angles)
    if reference_image is None:
        return """**❌ Reference Image Required**

Multi-Angles generation requires a reference image to maintain subject consistency across different camera angles.

Please upload a reference image and try again.
"""

    # Ensure Multi-Angles LoRA is available (auto-download if needed)
    success, message, lora_filename = ensure_multiangles_lora()

    if not success:
        return f"""**❌ Multi-Angles LoRA Not Available**

{message}

Please check your internet connection and try again.
"""

    # Multi-Angles always uses image-to-image mode
    generation_mode = "image-to-image"

    # Create LoRA state format expected by the queue
    lora_state = [{'name': lora_filename}]

    parameters = {
        'generation_mode': generation_mode,
        'prompt': composed_prompt,
        'steps': steps,
        'guidance': 4.0,  # Recommended for Multi-Angles
        'seed': seed,
        'width': width,
        'height': height,
        'quantization': quantization,
        'lora_state': copy.deepcopy(lora_state),
        'lora_strength_1': lora_scale,
        'lora_strength_2': 0.8,
        'lora_strength_3': 0.8,
        'reference_image': reference_image,
        'variation_strength': 0.7  # Good default for multi-angles
    }

    description = f"Multi-Angles: {composed_prompt[:40]}..."

    task_id = processing_queue.add_task(TaskType.FLUX2_GENERATION, parameters, description)

    # Get queue stats
    stats = processing_queue.get_stats()

    return f"""**✅ Multi-Angles Task Added to Queue**

**Task ID:** `{task_id[:8]}`
**Prompt:** {composed_prompt}
**LoRA Scale:** {lora_scale}

**Queue Status:**
- Pending: {stats['pending']}
- Processing: {stats['processing']}
- Completed: {stats['completed']}
"""