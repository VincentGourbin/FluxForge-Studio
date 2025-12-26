"""FLUX.2 UI Controls Module

Dynamic visibility control logic for the unified FLUX.2 Generation tab.
Manages conditional rendering of UI panels based on selected generation mode.

Features:
- Mode-based UI panel visibility
- Dynamic parameter defaults (steps, guidance)
- Preview generation for depth/canny modes
- Integration with Gradio update system

Author: FluxForge Team
License: MIT
"""

import gradio as gr
from typing import List, Any


def update_generation_mode_visibility(mode: str) -> List[Any]:
    """Update UI component visibility based on selected generation mode.

    This function controls which panels are shown/hidden when the user
    selects a different generation mode in the dropdown.

    Args:
        mode: Selected generation mode string (e.g., "✨ Text-to-Image")

    Returns:
        list: Gradio update objects for all dynamic components in order:
            [img2img_group, inpaint_group, outpaint_group, depth_group,
             canny_group, multiref_group, schnell_notice, steps, guidance]
    """

    # Configuration map for each mode
    visibility_configs = {
        "✨ Text-to-Image": {
            'img2img': False,
            'inpaint': False,
            'outpaint': False,
            'depth': False,
            'canny': False,
            'multiref': False,
            'schnell_notice': False,
            'steps': 28,
            'guidance': 4.0
        },
        "🔄 Image-to-Image": {
            'img2img': True,
            'inpaint': False,
            'outpaint': False,
            'depth': False,
            'canny': False,
            'multiref': False,
            'schnell_notice': False,
            'steps': 28,
            'guidance': 3.5
        },
        "🎨 Inpainting": {
            'img2img': False,
            'inpaint': True,
            'outpaint': False,
            'depth': False,
            'canny': False,
            'multiref': False,
            'schnell_notice': False,
            'steps': 28,
            'guidance': 4.0
        },
        "📐 Outpainting": {
            'img2img': False,
            'inpaint': False,
            'outpaint': True,
            'depth': False,
            'canny': False,
            'multiref': False,
            'schnell_notice': False,
            'steps': 28,
            'guidance': 4.0
        },
        "🌊 Depth-Guided": {
            'img2img': False,
            'inpaint': False,
            'outpaint': False,
            'depth': True,
            'canny': False,
            'multiref': False,
            'schnell_notice': False,
            'steps': 28,
            'guidance': 3.5
        },
        "🖋️ Canny-Guided": {
            'img2img': False,
            'inpaint': False,
            'outpaint': False,
            'depth': False,
            'canny': True,
            'multiref': False,
            'schnell_notice': False,
            'steps': 28,
            'guidance': 4.0
        },
        "🔀 Multi-Reference": {
            'img2img': False,
            'inpaint': False,
            'outpaint': False,
            'depth': False,
            'canny': False,
            'multiref': True,
            'schnell_notice': False,
            'steps': 28,
            'guidance': 4.0
        },
        "⚡ Quick Mode (schnell 4-step)": {
            'img2img': False,
            'inpaint': False,
            'outpaint': False,
            'depth': False,
            'canny': False,
            'multiref': False,
            'schnell_notice': True,
            'steps': 4,
            'guidance': 3.5
        }
    }

    # Get configuration for selected mode (default to text-to-image)
    config = visibility_configs.get(mode, visibility_configs["✨ Text-to-Image"])

    # Return Gradio update objects in the correct order
    return [
        gr.update(visible=config['img2img']),        # image_to_image_group
        gr.update(visible=config['inpaint']),        # inpainting_group
        gr.update(visible=config['outpaint']),       # outpainting_group
        gr.update(visible=config['depth']),          # depth_group
        gr.update(visible=config['canny']),          # canny_group
        gr.update(visible=config['multiref']),       # multi_ref_group
        gr.update(visible=config['schnell_notice']), # schnell_notice
        gr.update(value=config['steps']),            # steps slider
        gr.update(value=config['guidance'])          # guidance slider
    ]


def generate_depth_preview(input_image):
    """Generate depth map preview for depth-guided mode.

    Args:
        input_image: Input PIL Image

    Returns:
        PIL.Image: Depth map visualization
    """
    if input_image is None:
        return None

    try:
        from postprocessing.flux_depth import generate_depth_map
        depth_map = generate_depth_map(input_image)
        return depth_map
    except Exception as e:
        print(f"⚠️ Depth preview generation failed: {e}")
        return None


def generate_canny_preview(input_image, low_threshold, high_threshold):
    """Generate Canny edge preview for canny-guided mode.

    Args:
        input_image: Input PIL Image
        low_threshold: Canny low threshold (1-255)
        high_threshold: Canny high threshold (1-255)

    Returns:
        PIL.Image: Canny edge map
    """
    if input_image is None:
        return None

    try:
        from utils.canny_processing import preprocess_canny
        canny_edges = preprocess_canny(input_image, low_threshold, high_threshold)
        return canny_edges
    except Exception as e:
        print(f"⚠️ Canny preview generation failed: {e}")
        return None


def generate_outpaint_preview(base_image, top_percent, bottom_percent, left_percent, right_percent):
    """Generate outpainting expansion preview showing the extended canvas.

    Args:
        base_image: Base PIL Image to extend
        top_percent: Top expansion percentage (0-100)
        bottom_percent: Bottom expansion percentage (0-100)
        left_percent: Left expansion percentage (0-100)
        right_percent: Right expansion percentage (0-100)

    Returns:
        PIL.Image: Preview showing expanded canvas with original image
    """
    if base_image is None:
        return None

    try:
        from PIL import Image, ImageDraw
        import numpy as np

        # Get original dimensions
        orig_width, orig_height = base_image.size

        # Calculate new dimensions
        new_width = int(orig_width * (1 + (left_percent + right_percent) / 100))
        new_height = int(orig_height * (1 + (top_percent + bottom_percent) / 100))

        # Create new canvas (gray background)
        preview = Image.new('RGB', (new_width, new_height), color=(128, 128, 128))

        # Calculate paste position
        paste_x = int(orig_width * left_percent / 100)
        paste_y = int(orig_height * top_percent / 100)

        # Paste original image
        preview.paste(base_image, (paste_x, paste_y))

        # Draw border around original image area
        draw = ImageDraw.Draw(preview)
        draw.rectangle(
            [(paste_x, paste_y), (paste_x + orig_width, paste_y + orig_height)],
            outline=(255, 0, 0),
            width=3
        )

        return preview
    except Exception as e:
        print(f"⚠️ Outpaint preview generation failed: {e}")
        return None


def extract_inpainting_mask_preview(image_editor_data):
    """Extract and display the mask from ImageEditor for preview.

    Args:
        image_editor_data: Gradio ImageEditor data dictionary

    Returns:
        PIL.Image: Mask visualization (white = masked areas to regenerate)
    """
    if image_editor_data is None:
        return None

    try:
        from utils.mask_utils import extract_inpainting_mask_from_editor
        mask = extract_inpainting_mask_from_editor(image_editor_data)
        return mask
    except Exception as e:
        print(f"⚠️ Mask preview extraction failed: {e}")
        return None


def queue_flux2_generation(
    generation_mode, prompt, steps, guidance, seed,
    width, height, quantization, lora_state, lora_strength_1, lora_strength_2, lora_strength_3,
    # Image-to-Image parameters
    reference_image=None, variation_strength=0.6
):
    """Queue a FLUX.2 generation task (simplified 3-mode interface).

    This function is called by the Generate button and adds the task to the processing queue.

    Returns:
        str: Markdown feedback message for the user
    """
    import copy
    from core.processing_queue import processing_queue, TaskType

    # Map UI mode to internal mode string
    mode_mapping = {
        "✨ Text-to-Image": "text-to-image",
        "🔄 Image-to-Image": "image-to-image",
        "🚀 Z-Image-Turbo (fast)": "text-to-image"  # Uses Z-Image-Turbo
    }

    internal_mode = mode_mapping.get(generation_mode, "text-to-image")

    # Determine which mode
    is_zimage = generation_mode == "🚀 Z-Image-Turbo (fast)"

    if is_zimage:
        # Use Z-Image-Turbo generation
        task_type = TaskType.ZIMAGE_GENERATION
        parameters = {
            'prompt': prompt,
            'steps': steps,  # User configurable (default 9 = 8 NFEs)
            'seed': seed,
            'width': width,
            'height': height,
            # LoRA support for Z-Image
            'lora_state': copy.deepcopy(lora_state),
            'lora_strength_1': lora_strength_1,
            'lora_strength_2': lora_strength_2,
            'lora_strength_3': lora_strength_3
        }
        description = f"🚀 Z-Image-Turbo: {prompt[:40]}..."

    else:
        # Use FLUX.2 generation
        task_type = TaskType.FLUX2_GENERATION
        parameters = {
            'generation_mode': internal_mode,
            'prompt': prompt,
            'steps': steps,
            'guidance': guidance,
            'seed': seed,
            'width': width,
            'height': height,
            'quantization': quantization,
            'lora_state': copy.deepcopy(lora_state),
            'lora_strength_1': lora_strength_1,
            'lora_strength_2': lora_strength_2,
            'lora_strength_3': lora_strength_3
        }

        # Add mode-specific parameters for image-to-image
        if internal_mode == "image-to-image":
            parameters['reference_image'] = reference_image
            parameters['variation_strength'] = variation_strength

        # Description generated by ProcessingTask._generate_description()
        description = None

    # Add task to queue
    task_id = processing_queue.add_task(task_type, parameters, description)

    # Get queue stats
    stats = processing_queue.get_stats()

    return f"""**✅ Task Added to Queue**

**Task ID:** `{task_id[:8]}`
**Mode:** {generation_mode}
**Prompt:** {prompt[:60]}{'...' if len(prompt) > 60 else ''}

**Queue Status:**
- Pending: {stats['pending']}
- Processing: {stats['processing']}
- Completed: {stats['completed']}
"""
