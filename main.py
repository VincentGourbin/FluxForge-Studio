#!/usr/bin/env python3
"""
FluxForge Studio - Professional AI Image Generation Platform

Modern, modular main entry point for the professional FLUX.1 image generation interface.
Uses clean architecture with components organized in src/ subdirectories.

This application provides a comprehensive web interface for:
- FLUX.1 image generation with LoRA support  
- Advanced post-processing tools (FLUX Fill, Kontext, Depth, Canny, Redux, Background Removal, Upscaling)
- AI-powered prompt enhancement with Ollama integration
- Comprehensive image history management with gallery view

Features:
- 🎨 Multiple FLUX.1 model variants (dev, schnell, Fill, Kontext, Redux, Depth, Canny)
- 🛠️ Advanced post-processing pipeline with real-time previews
- 🧠 AI prompt enhancement with vision model support
- 📚 Complete image history with detailed metadata
- ⚡ Memory-efficient processing with device optimization

Author: FluxForge Team
License: MIT
"""

import os
import sys
import warnings
import argparse
import secrets
import string
from pathlib import Path

# Set PyTorch MPS fallback for Apple Silicon compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Filter out timm deprecation warnings  
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.*")

# Add src directory to Python path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import Gradio
import gradio as gr

# Import core modules
from core.config import *
from core.database import init_db, load_history, get_image_details, delete_image, sync_gallery_and_disk

# Import generator
from generator.image_generator import ImageGenerator

# Import post-processing modules
from postprocessing.flux_fill import process_flux_fill, update_flux_fill_controls_visibility, update_flux_fill_mode_visibility, generate_flux_fill_preview
from postprocessing.kontext import process_kontext
from postprocessing.flux_depth import generate_depth_map, process_flux_depth
from postprocessing.flux_canny import generate_canny_preview as flux_canny_preview, process_flux_canny
from postprocessing.flux_redux import process_flux_redux
from postprocessing.background_remover import load_background_removal_model, remove_background
from postprocessing.upscaler import upscale_image

# Import enhancement modules
from enhancement.prompt_enhancer import (
    enhance_prompt, 
    update_image_input_visibility, 
    update_button_label, 
    model_names
)

# Import UI modules
from ui.components import (
    create_generation_parameters,
    create_image_dimensions_controls,
    create_seed_control,
    create_prompt_input,
    create_model_selector,
    create_quantization_selector,
    create_post_processing_selector,
    create_expansion_controls,
    create_image_editor_component,
    create_preview_image,
    create_output_image,
    create_generation_button,
    create_metadata_checkbox
)
from ui.lora_manager import (
    create_lora_manager_interface,
    setup_lora_events
)

# Import utility modules
from utils.image_processing import ensure_rgb_format, cleanup_memory, save_image_with_metadata
from utils.mask_utils import extract_inpainting_mask_from_editor, create_outpainting_mask
from utils.canny_processing import preprocess_canny, generate_canny_preview
from utils.hf_cache_manager import refresh_hf_cache_for_gradio, delete_selected_hf_items
from utils.queue_helpers import (
    queue_standard_generation,
    queue_flux_fill,
    queue_kontext,
    queue_flux_depth,
    queue_flux_canny,
    queue_flux_redux,
    queue_background_removal,
    queue_upscaling
)

# Import processing queue modules
from core.processing_queue import processing_queue
from ui.processing_tab import create_processing_tab, setup_processing_tab_events


# Global constants
TEMP_IMAGES_DIR = "temp_images"

def initialize_application():
    """Initialize all application components and dependencies."""
    print("🔄 Initializing FluxForge Studio...")
    
    # Initialize database
    init_db()
    
    # Load background removal model
    modelbgrm = load_background_removal_model()
    
    # Initialize image generator
    image_generator = ImageGenerator()
    
    # Create temporary directories
    os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)
    
    print("✅ FluxForge Studio initialized successfully!")
    
    return modelbgrm, image_generator

def show_image_details(evt: gr.SelectData):
    """Display detailed information for a selected image from the history gallery."""
    if evt.index is None:
        return "No image selected.", None
    
    index = evt.index
    details_text = get_image_details(index)
    return details_text, index

def generate_image_wrapper(prompt, model_alias, quantization, steps, seed, metadata, guidance, height, width, 
                         lora_state, lora_strength_1, lora_strength_2, lora_strength_3):
    """Wrapper function to adapt new LoRA state system to old ImageGenerator interface."""
    
    # Get the image generator instance
    image_generator = ImageGenerator()
    
    # Convert LoRA state to individual checkbox/scale format expected by ImageGenerator
    lora_checkboxes = []
    lora_scales = []
    
    # Initialize with False for all LoRA models
    for _ in image_generator.lora_data:
        lora_checkboxes.append(False)
        lora_scales.append(1.0)
    
    # Process selected LoRA from state
    if lora_state:
        strengths = [lora_strength_1, lora_strength_2, lora_strength_3]
        for i, selected_lora in enumerate(lora_state):
            if i < len(strengths):
                # Find the index of this LoRA in lora_data
                lora_name = selected_lora['name']
                for j, lora_info in enumerate(image_generator.lora_data):
                    if lora_info['file_name'] == lora_name:
                        lora_checkboxes[j] = True
                        lora_scales[j] = strengths[i] if strengths[i] is not None else 0.8
                        break
    
    # Call the original generate_image with positional arguments (matching signature)
    return image_generator.generate_image(
        prompt,                    # prompt
        model_alias,              # model_alias
        steps,                    # steps
        seed,                     # seed
        metadata,                 # metadata
        guidance,                 # guidance
        height,                   # height
        width,                    # width
        "",                       # path - empty for HuggingFace models
        "None",                   # controlnet_type
        None,                     # controlnet_image_path
        1.0,                      # controlnet_strength
        False,                    # controlnet_save_canny
        False,                    # enable_stepwise
        gr.Progress(),            # progress
        100,                      # canny_low_threshold
        200,                      # canny_high_threshold
        2.0,                      # upscaler_multiplier
        "None",                   # flux_tools_type
        None,                     # flux_tools_image_path
        2.5,                      # flux_tools_guidance
        "None",                   # post_processing_type
        None,                     # post_processing_image_path
        2.0,                      # post_processing_multiplier
        quantization,             # quantization
        *lora_checkboxes,         # LoRA selections
        *lora_scales              # LoRA scales
    )

def create_main_interface():
    """Create the main Gradio interface with all tabs and components."""
    
    # Initialize application components
    modelbgrm, image_generator = initialize_application()
    
    with gr.Blocks(title="FluxForge Studio") as demo:
        gr.Markdown("# 🎨 FluxForge Studio")
        gr.Markdown("**Professional AI Image Generation Platform** - Create, enhance, and refine images with FLUX.1 models and advanced post-processing tools.")

        # ==============================================================================
        # TAB 1: CONTENT CREATION
        # ==============================================================================
        with gr.Tab("Content Creation"):
            gr.Markdown("## Image generation with advanced parameters")
            
            # Main prompt input
            prompt = create_prompt_input("Prompt (image description)", 2, "Describe the image you want to generate...")
            prompt.value = "Luxury food photograph"

            gr.Markdown("### Main Parameters")
            
            # Primary generation parameters
            with gr.Row():
                model_alias = create_model_selector(image_generator.model_options, "schnell")
                quantization = create_quantization_selector()
                steps = gr.Number(
                    label="Inference steps - More steps = better quality but slower", 
                    value=4, 
                    precision=0, 
                    minimum=1
                )
                seed = create_seed_control()
                metadata = create_metadata_checkbox()

            # Guidance parameter (conditionally visible)
            with gr.Row():
                guidance = gr.Number(
                    label="Guidance scale - Controls prompt adherence", 
                    value=3.5, 
                    visible=False
                )

            # Update guidance visibility and steps based on model selection
            model_alias.change(
                fn=image_generator.update_guidance_visibility, 
                inputs=model_alias, 
                outputs=guidance
            )
            model_alias.change(
                fn=image_generator.update_steps_for_model,
                inputs=model_alias,
                outputs=steps
            )

            # Image dimensions
            dimension_controls = create_image_dimensions_controls(1024, 1024)
            height = dimension_controls['height']
            width = dimension_controls['width']

            # LoRA management interface
            gr.Markdown("### 🎨 LoRA Models")
            lora_components = create_lora_manager_interface("", image_generator.lora_data)
            setup_lora_events(lora_components, image_generator.lora_data)

            # Generation button and queue feedback
            generate_btn = create_generation_button("🎨 Add to Queue")
            queue_feedback = gr.Markdown(value="**Ready to queue tasks** - Click Generate to add to processing queue")
            
            generate_btn.click(
                fn=queue_standard_generation,
                inputs=[
                    prompt, model_alias, quantization, steps, seed, metadata, guidance, height, width,
                    lora_components['state'], lora_components['strength_1'], 
                    lora_components['strength_2'], lora_components['strength_3']
                ],
                outputs=queue_feedback
            )

        # ==============================================================================
        # TAB 2: POST-PROCESSING
        # ==============================================================================
        with gr.Tab("Post-Processing"):
            gr.Markdown("## Advanced post-processing tools")
            
            # Queue feedback for post-processing
            post_processing_queue_feedback = gr.Markdown(
                value="**Ready to queue post-processing tasks** - Select a tool and click its button to add to processing queue"
            )
            
            # Processing type selector
            processing_type = create_post_processing_selector([
                "None", 
                "FLUX Fill - Inpainting (fill masked areas)", 
                "FLUX Fill - Outpainting (extend image borders)", 
                "Kontext (edit with text descriptions)", 
                "FLUX Depth (control with depth maps)", 
                "FLUX Canny (control with edge detection)", 
                "FLUX Redux (create image variations)", 
                "Background Removal (transparent backgrounds)", 
                "Upscaling (increase resolution)"
            ])

            # FLUX Fill controls
            with gr.Group(visible=False) as flux_fill_group:
                gr.Markdown("### 🎨 FLUX Fill Tools")
                
                fill_mode = gr.Dropdown(
                    label="Fill Mode",
                    choices=["Inpainting", "Outpainting"],
                    value="Inpainting"
                )
                
                with gr.Row():
                    with gr.Column(scale=2):
                        # Inpainting controls
                        with gr.Group(visible=True) as inpainting_group:
                            gr.Markdown("### ✏️ Inpainting - Fill masked areas")
                            flux_fill_editor = create_image_editor_component("Image Editor - Upload image and draw mask", 400)
                        
                        # Outpainting controls
                        with gr.Group(visible=False) as outpainting_group:
                            gr.Markdown("### 🔄 Outpainting - Extend image boundaries")
                            flux_outpaint_image = gr.Image(
                                label="Base Image - Image to extend",
                                type="pil",
                                height=400
                            )
                            expansion_controls = create_expansion_controls()
                            flux_outpaint_top = expansion_controls['top']
                            flux_outpaint_bottom = expansion_controls['bottom']
                            flux_outpaint_left = expansion_controls['left']
                            flux_outpaint_right = expansion_controls['right']
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 👁️ Mask Preview")
                        flux_fill_mask_preview = create_preview_image("Mask Preview - BLACK=keep, WHITE=fill (automatic)", 400)

                # LoRA section for FLUX Fill
                gr.Markdown("### 🎨 LoRA Models - Optional style enhancements")
                flux_fill_lora_components = create_lora_manager_interface("flux_fill_", image_generator.lora_data)
                setup_lora_events(flux_fill_lora_components, image_generator.lora_data, "flux_fill_")

                # Generation parameters
                with gr.Row():
                    flux_fill_prompt = create_prompt_input("Fill Prompt", 2, "Describe what should fill the masked area...")
                    with gr.Column():
                        flux_fill_steps = gr.Slider(
                            label="Inference Steps",
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=20
                        )
                        flux_fill_guidance = gr.Slider(
                            label="Guidance Scale",
                            minimum=1.0,
                            maximum=50.0,
                            step=1.0,
                            value=30.0
                        )
                        flux_fill_quantization = create_quantization_selector()

                # Generation button
                flux_fill_generate_btn = create_generation_button("🎨 Add FLUX Fill to Queue", "primary", "lg")

            # Kontext controls
            with gr.Group(visible=False) as kontext_group:
                gr.Markdown("### 🖼️ Kontext - Text-based image editing")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        kontext_input_image = gr.Image(
                            label="Input Image - Image to edit/transform",
                            type="pil",
                            height=400
                        )
                    
                    with gr.Column(scale=1):
                        kontext_prompt = gr.Textbox(
                            label="Edit Prompt - Describe the changes you want to make",
                            placeholder="e.g., 'change the sky to sunset colors', 'add glasses to the person'...",
                            lines=4
                        )
                        kontext_steps = gr.Slider(
                            label="Inference Steps",
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=25
                        )
                        kontext_guidance = gr.Slider(
                            label="Guidance Scale",
                            minimum=1.0,
                            maximum=10.0,
                            step=0.1,
                            value=2.5
                        )
                        kontext_quantization = create_quantization_selector()

                # LoRA section for Kontext
                gr.Markdown("### 🎨 LoRA Models - Optional style enhancements")
                kontext_lora_components = create_lora_manager_interface("kontext_", image_generator.lora_data)
                setup_lora_events(kontext_lora_components, image_generator.lora_data, "kontext_")

                kontext_generate_btn = create_generation_button("🖼️ Add Kontext Edit to Queue", "primary", "lg")

            # FLUX Depth controls
            with gr.Group(visible=False) as flux_depth_group:
                gr.Markdown("### 🌊 FLUX Depth - Depth-guided image generation")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        depth_input_image = gr.Image(
                            label="Input Image - Image to extract depth from",
                            type="pil",
                            height=400
                        )
                        
                        # Step 1: Generate depth preview
                        depth_preview_btn = create_generation_button("🔍 Generate Depth Map", "secondary", "lg")
                        depth_map_preview = create_preview_image("Depth Map Preview", 400)
                    
                    with gr.Column(scale=1):
                        depth_prompt = gr.Textbox(
                            label="Generation Prompt - Describe what you want to generate",
                            placeholder="e.g., 'a futuristic cityscape', 'underwater scene', 'fantasy landscape'...",
                            lines=4
                        )
                        depth_steps = gr.Slider(
                            label="Inference Steps",
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=28
                        )
                        depth_guidance = gr.Slider(
                            label="Guidance Scale",
                            minimum=1.0,
                            maximum=10.0,
                            step=0.1,
                            value=3.5
                        )
                        depth_quantization = create_quantization_selector()

                # LoRA section for FLUX Depth
                gr.Markdown("### 🎨 LoRA Models - Optional style enhancements")
                depth_lora_components = create_lora_manager_interface("depth_", image_generator.lora_data)
                setup_lora_events(depth_lora_components, image_generator.lora_data, "depth_")

                # Step 2: Generate final image
                depth_generate_btn = create_generation_button("🌊 Add FLUX Depth to Queue", "primary", "lg")

            # FLUX Canny controls
            with gr.Group(visible=False) as flux_canny_group:
                gr.Markdown("### 🖋️ FLUX Canny - Edge-guided image generation")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        canny_input_image = gr.Image(
                            label="Input Image - Image to extract edges from",
                            type="pil",
                            height=400
                        )
                        
                        # Canny threshold controls
                        with gr.Row():
                            canny_low_threshold = gr.Slider(
                                label="Low Threshold - Lower edge detection threshold",
                                minimum=1,
                                maximum=255,
                                step=1,
                                value=100
                            )
                            canny_high_threshold = gr.Slider(
                                label="High Threshold - Higher edge detection threshold",
                                minimum=1,
                                maximum=255,
                                step=1,
                                value=200
                            )
                        
                        # Step 1: Generate Canny preview
                        canny_preview_btn = create_generation_button("🔍 Generate Edge Preview", "secondary", "lg")
                        canny_map_preview = create_preview_image("Canny Edge Preview", 400)
                    
                    with gr.Column(scale=1):
                        canny_prompt = gr.Textbox(
                            label="Generation Prompt - Describe what you want to generate",
                            placeholder="e.g., 'a futuristic robot', 'architectural building', 'portrait painting'...",
                            lines=4
                        )
                        canny_steps = gr.Slider(
                            label="Inference Steps",
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=28
                        )
                        canny_guidance = gr.Slider(
                            label="Guidance Scale",
                            minimum=1.0,
                            maximum=50.0,
                            step=0.5,
                            value=30.0
                        )
                        canny_quantization = create_quantization_selector()

                # LoRA section for FLUX Canny
                gr.Markdown("### 🎨 LoRA Models - Optional style enhancements")
                canny_lora_components = create_lora_manager_interface("canny_", image_generator.lora_data)
                setup_lora_events(canny_lora_components, image_generator.lora_data, "canny_")

                # Step 2: Generate final image
                canny_generate_btn = create_generation_button("🖋️ Add FLUX Canny to Queue", "primary", "lg")

            # Background Removal controls
            with gr.Group(visible=False) as bg_removal_group:
                gr.Markdown("### 🎭 Background Removal")
                
                bg_input_image = gr.Image(
                    label="Input Image",
                    type="pil",
                    height=400
                )
                bg_remove_btn = create_generation_button("🎭 Add Background Removal to Queue", "primary", "lg")

            # FLUX Redux controls
            with gr.Group(visible=False) as flux_redux_group:
                gr.Markdown("### 🔄 FLUX Redux - Image variation and refinement")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        redux_input_image = gr.Image(
                            label="Input Image - Image to create variations from",
                            type="pil",
                            height=400
                        )
                    
                    with gr.Column(scale=1):
                        redux_guidance = gr.Slider(
                            label="Guidance Scale - Controls how closely output follows input",
                            minimum=1.0,
                            maximum=5.0,
                            step=0.1,
                            value=3.0
                        )
                        redux_steps = gr.Slider(
                            label="Inference Steps",
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=28
                        )
                        redux_variation_strength = gr.Slider(
                            label="Variation Strength - Intensity of variation",
                            minimum=0.1,
                            maximum=1.0,
                            step=0.1,
                            value=0.6
                        )
                        redux_quantization = create_quantization_selector()

                redux_generate_btn = create_generation_button("🔄 Add FLUX Redux to Queue", "primary", "lg")

            # Upscaling controls
            with gr.Group(visible=False) as upscaling_group:
                gr.Markdown("### 📈 Image Upscaling")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        upscale_input_image = gr.Image(
                            label="Input Image",
                            type="pil",
                            height=400
                        )
                    
                    with gr.Column(scale=1):
                        upscale_factor = gr.Slider(
                            label="Upscale Factor",
                            minimum=1.0,
                            maximum=10.0,
                            step=0.5,
                            value=2.0
                        )
                        upscale_quantization = create_quantization_selector()
                
                upscale_btn = create_generation_button("📈 Add Upscale Image to Queue", "primary", "lg")

            # Set up processing type change event
            processing_type.change(
                fn=update_flux_fill_controls_visibility,
                inputs=processing_type,
                outputs=[flux_fill_group, kontext_group, flux_depth_group, flux_canny_group, flux_redux_group, bg_removal_group, upscaling_group]
            )
            
            # Set up FLUX Fill mode change event with preview
            def combined_mode_change(mode):
                # Update visibility
                visibility_updates = update_flux_fill_mode_visibility(mode)
                # Generate preview
                preview = generate_flux_fill_preview(mode, None, None, 25, 25, 25, 25)
                return (*visibility_updates, preview)
            
            fill_mode.change(
                fn=combined_mode_change,
                inputs=fill_mode,
                outputs=[inpainting_group, outpainting_group, flux_fill_mask_preview]
            )
            
            # Automatic preview generation for inpainting when drawing on image
            flux_fill_editor.change(
                fn=generate_flux_fill_preview,
                inputs=[
                    fill_mode, flux_fill_editor, flux_outpaint_image,
                    flux_outpaint_top, flux_outpaint_bottom, flux_outpaint_left, flux_outpaint_right
                ],
                outputs=flux_fill_mask_preview
            )
            
            # Automatic preview generation for outpainting when image or percentages change
            flux_outpaint_image.change(
                fn=generate_flux_fill_preview,
                inputs=[
                    fill_mode, flux_fill_editor, flux_outpaint_image,
                    flux_outpaint_top, flux_outpaint_bottom, flux_outpaint_left, flux_outpaint_right
                ],
                outputs=flux_fill_mask_preview
            )
            
            # Auto-update preview when outpainting percentages change
            for slider in [flux_outpaint_top, flux_outpaint_bottom, flux_outpaint_left, flux_outpaint_right]:
                slider.change(
                    fn=generate_flux_fill_preview,
                    inputs=[
                        fill_mode, flux_fill_editor, flux_outpaint_image,
                        flux_outpaint_top, flux_outpaint_bottom, flux_outpaint_left, flux_outpaint_right
                    ],
                    outputs=flux_fill_mask_preview
                )
            
            # FLUX Fill generation event - wrapper to pass image_generator
            def flux_fill_wrapper(fill_mode, image_editor_data, outpaint_image, prompt, steps, guidance_scale, quantization,
                                top_percent, bottom_percent, left_percent, right_percent,
                                lora_state, lora_strength_1, lora_strength_2, lora_strength_3):
                return process_flux_fill(
                    fill_mode, image_editor_data, outpaint_image, prompt, steps, guidance_scale, quantization,
                    top_percent, bottom_percent, left_percent, right_percent,
                    lora_state, lora_strength_1, lora_strength_2, lora_strength_3,
                    image_generator
                )
            
            flux_fill_generate_btn.click(
                fn=queue_flux_fill,
                inputs=[
                    fill_mode, flux_fill_editor, flux_outpaint_image, flux_fill_prompt, flux_fill_steps, flux_fill_guidance, flux_fill_quantization,
                    flux_outpaint_top, flux_outpaint_bottom, flux_outpaint_left, flux_outpaint_right,
                    flux_fill_lora_components['state'],
                    flux_fill_lora_components['strength_1'], flux_fill_lora_components['strength_2'], flux_fill_lora_components['strength_3']
                ],
                outputs=post_processing_queue_feedback,
                show_progress=True
            )
            
            # Kontext generation event - wrapper to pass image_generator
            def kontext_wrapper(input_image, prompt, steps, guidance_scale, quantization, lora_state, lora_strength_1, lora_strength_2, lora_strength_3):
                return process_kontext(
                    input_image, prompt, steps, guidance_scale, quantization, lora_state,
                    lora_strength_1, lora_strength_2, lora_strength_3,
                    image_generator
                )
            
            kontext_generate_btn.click(
                fn=queue_kontext,
                inputs=[
                    kontext_input_image, kontext_prompt, kontext_steps, kontext_guidance, kontext_quantization,
                    kontext_lora_components['state'],
                    kontext_lora_components['strength_1'], kontext_lora_components['strength_2'], kontext_lora_components['strength_3']
                ],
                outputs=post_processing_queue_feedback,
                show_progress=True
            )
            
            # Background removal event
            bg_remove_btn.click(
                fn=queue_background_removal,
                inputs=bg_input_image,
                outputs=post_processing_queue_feedback,
                show_progress=True
            )
            
            # Upscaling event
            upscale_btn.click(
                fn=queue_upscaling,
                inputs=[upscale_input_image, upscale_factor, upscale_quantization],
                outputs=post_processing_queue_feedback,
                show_progress=True
            )
            
            # FLUX Depth events
            # Step 1: Generate depth map preview
            depth_preview_btn.click(
                fn=generate_depth_map,
                inputs=depth_input_image,
                outputs=depth_map_preview,
                show_progress=True
            )
            
            # Step 2: FLUX Depth generation event - wrapper to pass image_generator
            def flux_depth_wrapper(input_image, prompt, steps, guidance_scale, quantization, lora_state, lora_strength_1, lora_strength_2, lora_strength_3):
                return process_flux_depth(
                    input_image, prompt, steps, guidance_scale, quantization, lora_state,
                    lora_strength_1, lora_strength_2, lora_strength_3,
                    image_generator
                )
            
            depth_generate_btn.click(
                fn=queue_flux_depth,
                inputs=[
                    depth_input_image, depth_prompt, depth_steps, depth_guidance, depth_quantization,
                    depth_lora_components['state'],
                    depth_lora_components['strength_1'], depth_lora_components['strength_2'], depth_lora_components['strength_3']
                ],
                outputs=post_processing_queue_feedback,
                show_progress=True
            )
            
            # FLUX Canny events
            # Step 1: Generate Canny edge preview
            canny_preview_btn.click(
                fn=flux_canny_preview,
                inputs=[canny_input_image, canny_low_threshold, canny_high_threshold],
                outputs=canny_map_preview,
                show_progress=True
            )
            
            # Real-time Canny preview updates when image or thresholds change
            canny_input_image.change(
                fn=flux_canny_preview,
                inputs=[canny_input_image, canny_low_threshold, canny_high_threshold],
                outputs=canny_map_preview
            )
            
            canny_low_threshold.change(
                fn=flux_canny_preview,
                inputs=[canny_input_image, canny_low_threshold, canny_high_threshold],
                outputs=canny_map_preview
            )
            
            canny_high_threshold.change(
                fn=flux_canny_preview,
                inputs=[canny_input_image, canny_low_threshold, canny_high_threshold],
                outputs=canny_map_preview
            )
            
            # Step 2: FLUX Canny generation event - wrapper to pass image_generator
            def flux_canny_wrapper(input_image, prompt, steps, guidance_scale, quantization, low_threshold, high_threshold, lora_state, lora_strength_1, lora_strength_2, lora_strength_3):
                return process_flux_canny(
                    input_image, prompt, steps, guidance_scale, quantization, low_threshold, high_threshold, lora_state,
                    lora_strength_1, lora_strength_2, lora_strength_3,
                    image_generator
                )
            
            canny_generate_btn.click(
                fn=queue_flux_canny,
                inputs=[
                    canny_input_image, canny_prompt, canny_steps, canny_guidance, canny_quantization, canny_low_threshold, canny_high_threshold,
                    canny_lora_components['state'],
                    canny_lora_components['strength_1'], canny_lora_components['strength_2'], canny_lora_components['strength_3']
                ],
                outputs=post_processing_queue_feedback,
                show_progress=True
            )
            
            # FLUX Redux generation event - wrapper to pass image_generator
            def flux_redux_wrapper(input_image, guidance_scale, steps, variation_strength, quantization):
                return process_flux_redux(
                    input_image, guidance_scale, steps, variation_strength, quantization,
                    image_generator
                )
            
            redux_generate_btn.click(
                fn=queue_flux_redux,
                inputs=[
                    redux_input_image, redux_guidance, redux_steps, redux_variation_strength, redux_quantization
                ],
                outputs=post_processing_queue_feedback,
                show_progress=True
            )

        # ==============================================================================
        # TAB 3: PROCESSING
        # ==============================================================================
        with gr.Tab("Processing"):
            processing_components = create_processing_tab()
            setup_processing_tab_events(processing_components, image_generator, modelbgrm)

        # ==============================================================================
        # TAB 4: PROMPT ENHANCER
        # ==============================================================================
        with gr.Tab("Prompt Enhancer"):
            gr.Markdown("## Prompt Enhancement using Ollama")
            gr.Markdown("Use Ollama models to enhance and detail your image generation prompts.")

            # Import enhancer functions from modular structure
            try:
                from enhancement.prompt_enhancer import enhance_prompt, update_image_input_visibility, update_button_label, model_names, models_info
                enhancer_available = True
            except ImportError as e:
                print(f"Prompt enhancer import error: {e}")
                enhancer_available = False
                model_names = ["ollama-not-available"]
                models_info = {}

            if enhancer_available:
                with gr.Row():
                    selected_model = gr.Dropdown(
                        label="Select an Ollama model - AI model for prompt enhancement",
                        choices=model_names,
                        value=model_names[0] if model_names else None
                    )
                    input_text = gr.Textbox(
                        label="Text to process - Simple prompt you want to enhance",
                        placeholder="Enter your basic prompt here...",
                        lines=3
                    )
                
                # Image input (conditionally visible based on model)
                # Set initial visibility based on first model
                initial_image_visible = False
                if model_names and model_names[0]:
                    first_model_capabilities = models_info.get(model_names[0], [])
                    initial_image_visible = 'vision' in first_model_capabilities
                    
                input_image = gr.Image(
                    label="Image to provide to the model (if required) - Some models can analyze images",
                    type='filepath',
                    visible=initial_image_visible
                )

                # Update image visibility based on model selection
                selected_model.change(
                    fn=update_image_input_visibility,
                    inputs=selected_model,
                    outputs=input_image
                )

                # Enhancement button with dynamic label
                # Set initial button label based on first model
                initial_button_label = "🚀 Enhance Prompt"
                if model_names and model_names[0]:
                    first_model_capabilities = models_info.get(model_names[0], [])
                    if 'vision' in first_model_capabilities:
                        initial_button_label = "🚀 Analyze Image"
                        
                enhance_button = gr.Button(initial_button_label, variant="primary")

                # Update button label based on model selection
                selected_model.change(
                    fn=update_button_label,
                    inputs=selected_model,
                    outputs=enhance_button
                )
                
                # Enhanced output display
                enhanced_output = gr.Markdown(
                    label="Enhanced Prompt",
                    value="Enhanced prompt will appear here...",
                    elem_classes=["enhanced-output"]
                )

                # Streaming wrapper for real-time updates
                def enhance_prompt_streaming(selected_model, input_text, input_image):
                    """Streaming wrapper that yields each part as it's generated."""
                    try:
                        # Call original generator function and yield each part
                        for part in enhance_prompt(selected_model, input_text, input_image):
                            if isinstance(part, str):
                                yield part
                    except Exception as e:
                        yield f"Error: {str(e)}"
                
                # Connect enhancement functionality with streaming
                enhance_button.click(
                    fn=enhance_prompt_streaming,
                    inputs=[selected_model, input_text, input_image],
                    outputs=enhanced_output,
                    show_progress=True
                )
            else:
                gr.Markdown("⚠️ **Ollama not available**. Please install and start Ollama to use prompt enhancement features.")

        # ==============================================================================
        # TAB 5: HISTORY
        # ==============================================================================
        with gr.Tab("History"):
            gr.Markdown("## Image generation history")
            
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
            
            with gr.Row():
                with gr.Column(scale=3):
                    gallery = gr.Gallery(
                        label="Generated Images",
                        show_label=True,
                        elem_id="gallery",
                        columns=4,
                        rows=2,
                        height="auto"
                    )
                
                with gr.Column(scale=1):
                    image_details = gr.Textbox(
                        label="Image Details",
                        lines=20,
                        max_lines=20,
                        interactive=False
                    )
                    
                    with gr.Row():
                        delete_btn = gr.Button("🗑️ Delete", variant="stop")
                        selected_index = gr.State(value=None)

            # Set up history events
            refresh_btn.click(
                fn=load_history,
                outputs=gallery
            )
            
            gallery.select(
                fn=show_image_details,
                outputs=[image_details, selected_index]
            )
            
            def delete_and_refresh(idx):
                """Delete image and refresh gallery."""
                if idx is not None:
                    result = delete_image(idx)
                    refreshed_gallery = load_history()
                    return refreshed_gallery, "Image deleted successfully."
                else:
                    return load_history(), "No image selected to delete."
            
            delete_btn.click(
                fn=delete_and_refresh,
                inputs=selected_index,
                outputs=[gallery, image_details]
            )

        # ==============================================================================
        # TAB 6: ADMIN
        # ==============================================================================
        with gr.Tab("Admin"):
            gr.Markdown("## Administration and maintenance tools")
            
            with gr.Column():
                gr.Markdown("### 🔄 Gallery Synchronization")
                gr.Markdown("Synchronize the gallery with disk files. This will:")
                gr.Markdown("- Move orphaned images (on disk but not in database) to `orphaned_pictures/` folder")
                gr.Markdown("- Remove database entries pointing to non-existent files")
                gr.Markdown("- Handle associated JSON metadata files")
                
                sync_btn = gr.Button("🔄 Sync Gallery with Disk", variant="primary", size="lg")
                sync_status = gr.Textbox(
                    label="Sync Status",
                    lines=10,
                    interactive=False,
                    placeholder="Click 'Sync Gallery with Disk' to see results..."
                )
                
                gr.Markdown("### 📊 System Information")
                gr.Markdown(f"- Device: {image_generator.device}")
                gr.Markdown(f"- Available LoRA models: {len(image_generator.lora_data)}")
                gr.Markdown(f"- Model options: {', '.join(image_generator.model_options)}")
                
                gr.Markdown("### 🗂️ Model Cache Management")
                gr.Markdown("Manage downloaded models to save disk space and bandwidth.")
                
                cache_info_btn = gr.Button("📊 Show Cache Status", variant="secondary", size="lg")
                cache_info_display = gr.Textbox(
                    label="Cache Information",
                    lines=15,
                    interactive=False,
                    placeholder="Click 'Show Cache Status' to see model cache information..."
                )
                
                gr.Markdown("### 🗂️ HuggingFace Cache Management")
                gr.Markdown("View and manage all HuggingFace cached models with selective deletion.")
                
                hf_refresh_btn = gr.Button("🔄 Refresh HF Cache", variant="primary", size="lg")
                hf_checkbox_group = gr.CheckboxGroup(
                    label="Select HuggingFace cache items to delete",
                    choices=[],
                    value=[],
                    interactive=True,
                    info="Click 'Refresh HF Cache' to load available models"
                )
                
                with gr.Row():
                    hf_delete_btn = gr.Button("🗑️ Delete Selected Items", variant="stop", size="lg")
                    hf_status_display = gr.Markdown("**Status:** Ready to manage cache")
            
            # Set up admin events
            def sync_with_status():
                """Sync and return status message."""
                try:
                    result = sync_gallery_and_disk()
                    return "✅ Synchronization completed successfully! Check console for details."
                except Exception as e:
                    return f"❌ Synchronization failed: {str(e)}"
            
            sync_btn.click(
                fn=sync_with_status,
                outputs=sync_status,
                show_progress=True
            )
            
            # HuggingFace Cache management functions (imported from utils.hf_cache_manager)
            
            # Cache management event
            def show_cache_status():
                """Show detailed cache status for all FLUX models."""
                try:
                    from utils.model_cache import get_cache_stats, format_size
                    
                    stats = get_cache_stats()
                    
                    output_lines = [
                        "🗂️ FLUX MODELS CACHE STATUS",
                        "=" * 50,
                        "",
                        f"📊 Cache Summary:",
                        f"   Total models: {stats['total_models']}",
                        f"   Cached models: {stats['cached_models']}",
                        f"   Cache percentage: {stats['cache_percentage']:.1f}%",
                        f"   Total cache size: {stats['total_size_formatted']}",
                        "",
                        "📋 Individual Models:"
                    ]
                    
                    for model_name, info in stats["models"].items():
                        status = "✅ Cached" if info["cached"] else "❌ Not cached"
                        size_info = f"({info['size_formatted']})" if info["cached"] else ""
                        output_lines.append(f"   {model_name}: {status} {size_info}")
                    
                    output_lines.extend([
                        "",
                        "💡 Tips:",
                        "   • Models are automatically cached after first download",
                        "   • Cache location: ~/.cache/huggingface/",
                        "   • Cached models load faster and work offline",
                        "   • Large models (>1GB) benefit most from caching"
                    ])
                    
                    return "\n".join(output_lines)
                    
                except Exception as e:
                    return f"❌ Error retrieving cache status: {str(e)}"
            
            cache_info_btn.click(
                fn=show_cache_status,
                outputs=cache_info_display,
                show_progress=True
            )
            
            # HuggingFace Cache management events
            hf_cache_info_state = gr.State(None)
            
            
            hf_refresh_btn.click(
                fn=refresh_hf_cache_for_gradio,
                outputs=[hf_checkbox_group, hf_status_display, hf_cache_info_state],
                show_progress=True
            )
            
            hf_delete_btn.click(
                fn=delete_selected_hf_items,
                inputs=[hf_checkbox_group, hf_cache_info_state],
                outputs=[hf_checkbox_group, hf_status_display, hf_cache_info_state],
                show_progress=True
            )

    return demo

def generate_random_credentials():
    """Generate random username and password for authentication."""
    # Generate random username (8 characters)
    username = ''.join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(8))
    
    # Generate random password (12 characters with mixed case, digits, and symbols)
    password_chars = string.ascii_letters + string.digits + "!@#$%^&*"
    password = ''.join(secrets.choice(password_chars) for _ in range(12))
    
    return username, password

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="FluxForge Studio - Professional AI Image Generation Platform")
    parser.add_argument(
        "-s", "--share", 
        action="store_true", 
        help="Generate a public shareable link with random authentication credentials"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting FluxForge Studio...")
    
    # Configure launch parameters
    launch_params = {
        "show_error": True,
        "quiet": False
    }
    
    if args.share:
        # Generate random credentials
        username, password = generate_random_credentials()
        
        print("🌐 Creating public shareable link...")
        print(f"🔐 Authentication credentials:")
        print(f"   Username: {username}")
        print(f"   Password: {password}")
        print("⚠️  Please save these credentials - they won't be shown again!")
        print("📋 Share link will be displayed once the app starts...")
        
        launch_params.update({
            "share": True,
            "auth": (username, password)
        })
    else:
        launch_params["share"] = False
    
    try:
        # Create and launch the interface
        demo = create_main_interface()
        
        # Launch with configuration
        demo.launch(**launch_params)
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()