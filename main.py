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

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Set PyTorch MPS fallback for Apple Silicon compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Disable tokenizers parallelism warning (forking after parallelism)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Filter out warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.*")

# Suppress pkg_resources stderr warnings about multiple distributions
# This filters the "Multiple distributions found for package optimum" message
import io
import contextlib

class StderrFilter:
    """Context manager to filter specific stderr messages."""
    def __init__(self, filter_text):
        self.filter_text = filter_text
        self.old_stderr = None

    def __enter__(self):
        self.old_stderr = sys.stderr
        sys.stderr = self
        return self

    def __exit__(self, *args):
        sys.stderr = self.old_stderr

    def write(self, text):
        # Only write if the text doesn't contain the filter pattern
        if self.filter_text not in text:
            self.old_stderr.write(text)

    def flush(self):
        self.old_stderr.flush()

# Add src directory to Python path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import Gradio and core modules with stderr filtering to suppress pkg_resources warnings
with StderrFilter("Multiple distributions found for package"):
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
    from postprocessing.background_remover import remove_background
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
        create_generation_button
    )
    from ui.lora_manager import (
        create_lora_manager_interface,
        setup_lora_events,
        get_lora_dropdown_choices_for_mode
    )
    # Qwen-Image is now integrated into Content Creation tab

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
        queue_upscaling,
        queue_multiangles_generation
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

    # Background removal model is now lazy-loaded on first use
    # No need to load it at startup (saves time and avoids auth errors if not used)
    modelbgrm = None

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

def generate_image_wrapper(prompt, model_alias, quantization, steps, seed, guidance, height, width, 
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
        True,                     # metadata - always save metadata
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
        # TAB 1: GENERATION (FLUX.2 UNIFIED)
        # ==============================================================================
        with gr.Tab("Generation"):
            gr.Markdown("## 🎨 FLUX.2 Multi-Modal Generation")
            gr.Markdown("**Unified generation interface** - Text-to-image, image-to-image, inpainting, outpainting, depth/canny control, and multi-reference composition.")

            # Import flux2 controls
            from ui.flux2_controls import (
                update_generation_mode_visibility,
                generate_depth_preview,
                generate_canny_preview,
                generate_outpaint_preview,
                extract_inpainting_mask_preview,
                queue_flux2_generation
            )

            # ===== MODE SELECTOR (Primary Control) =====
            generation_mode = gr.Dropdown(
                label="Generation Mode",
                choices=[
                    "✨ Text-to-Image",
                    "🔄 Image-to-Image",
                    "🚀 Z-Image-Turbo (fast)"
                ],
                value="✨ Text-to-Image",
                info="Select the type of generation you want to perform"
            )

            # ===== COMMON CONTROLS (Always Visible) =====
            gr.Markdown("### Prompt & Settings")

            prompt = create_prompt_input("Prompt", 3, "Describe the image you want to generate...")
            prompt.value = "A serene mountain landscape at sunset"

            with gr.Row():
                steps = gr.Slider(
                    label="Inference Steps",
                    minimum=1,
                    maximum=50,
                    value=28,
                    step=1,
                    info="More steps = better quality but slower (28 recommended)"
                )
                guidance = gr.Slider(
                    label="Guidance Scale",
                    minimum=1.0,
                    maximum=10.0,
                    value=4.0,
                    step=0.1,
                    info="How closely to follow the prompt (4.0 default)"
                )

            with gr.Row():
                seed = create_seed_control()
                quantization = gr.Dropdown(
                    label="Quantization",
                    choices=["qint8", "full"],
                    value="qint8",
                    info="qint8: Transformer quantized (~35GB FLUX.2) | full: Full precision (~115GB)"
                )

            # ===== DYNAMIC PANELS (Visibility Controlled by Mode) =====

            # Z-Image-Turbo Mode Notice
            with gr.Column(visible=False) as zimage_notice:
                gr.Markdown("### 🚀 Z-Image-Turbo Mode")
                gr.Markdown("**Fast generation** - Uses Tongyi-MAI/Z-Image-Turbo (6B model). Default: 9 steps. Adjust steps as needed.")

            # Image-to-Image Mode
            with gr.Column(visible=False) as image_to_image_group:
                gr.Markdown("### 🔄 Image-to-Image Settings")
                reference_image = gr.Image(
                    label="Reference Image",
                    type="pil",
                    height=400
                )
                variation_strength = gr.Slider(
                    label="Variation Strength",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.6,
                    step=0.1,
                    info="Higher = more variation from reference"
                )


            # ===== LORA ENHANCEMENT (Always Available) =====
            gr.Markdown("### 🎨 LoRA Style Enhancement")
            # Import filter function for initial dropdown population
            from ui.lora_manager import filter_lora_by_model
            # Default mode is Text-to-Image (FLUX.2), so filter for flux2-dev initially
            initial_lora_data = filter_lora_by_model(image_generator.lora_data, "flux2-dev")
            lora_components = create_lora_manager_interface("flux2_", initial_lora_data)
            # Setup events with FULL lora_data for dynamic filtering
            setup_lora_events(lora_components, image_generator.lora_data, "flux2_")

            # ===== ADVANCED OPTIONS =====
            with gr.Accordion("⚙️ Advanced Options", open=False):
                dimension_controls = create_image_dimensions_controls(1024, 1024)
                flux2_width = dimension_controls['width']
                flux2_height = dimension_controls['height']

            # ===== GENERATION BUTTON =====
            generate_btn = create_generation_button("🎨 Add to Queue")
            queue_feedback = gr.Markdown(value="**Ready to queue** - Configure parameters and click Generate")

            # ===== EVENT HANDLERS =====

            # Simple visibility update function with LoRA filtering
            def update_simple_mode_visibility(mode):
                """Update UI visibility for simplified 3-mode interface and filter LoRA dropdown"""
                is_img2img = (mode == "🔄 Image-to-Image")
                is_zimage = (mode == "🚀 Z-Image-Turbo (fast)")

                # Filter LoRA dropdown by model compatibility
                lora_dropdown_update = get_lora_dropdown_choices_for_mode(image_generator.lora_data, mode)

                # Determine steps and guidance based on mode
                if is_zimage:
                    steps_value = 9  # 9 steps = 8 NFEs for Z-Image (can be adjusted)
                    steps_interactive = True
                else:
                    steps_value = 28
                    steps_interactive = True

                # Z-Image doesn't use guidance (must be 0.0) or quantization
                guidance_visible = not is_zimage
                quantization_visible = not is_zimage  # Z-Image has no quantized version

                return [
                    gr.update(visible=is_img2img),      # image_to_image_group
                    gr.update(visible=is_zimage),       # zimage_notice
                    gr.update(value=steps_value, interactive=steps_interactive),  # steps
                    gr.update(visible=guidance_visible),  # guidance
                    gr.update(visible=quantization_visible),  # quantization
                    lora_dropdown_update                # lora dropdown filtered by model
                ]

            # Mode change updates UI visibility and LoRA dropdown
            generation_mode.change(
                fn=update_simple_mode_visibility,
                inputs=generation_mode,
                outputs=[
                    image_to_image_group,
                    zimage_notice,
                    steps,
                    guidance,
                    quantization,
                    lora_components['available_dropdown']
                ]
            )

            # Generate button queues task
            generate_btn.click(
                fn=queue_flux2_generation,
                inputs=[
                    generation_mode, prompt, steps, guidance, seed,
                    flux2_width, flux2_height, quantization,
                    lora_components['state'], lora_components['strength_1'],
                    lora_components['strength_2'], lora_components['strength_3'],
                    # Image-to-Image parameter
                    reference_image, variation_strength
                ],
                outputs=queue_feedback
            )


        # ==============================================================================
        # TAB 2b: SPECIFIC PROCESSING
        # ==============================================================================
        with gr.Tab("Specific Processing"):
            gr.Markdown("## Specific Processing Tools")
            gr.Markdown("Standalone tools for image processing that don't require generation.")

            # Processing type selector
            specific_processing_type = gr.Dropdown(
                label="Processing Type",
                choices=["Background Removal (RMBG)", "Multi-Angles Generation"],
                value="Background Removal (RMBG)"
            )

            # === RMBG Section ===
            with gr.Column(visible=True) as rmbg_section:
                gr.Markdown("### Background Removal")
                gr.Markdown("Remove background from any image using RMBG-2.0 AI model.")

                rmbg_input = gr.Image(
                    label="Input Image",
                    type="pil",
                    height=400
                )

                rmbg_btn = gr.Button("Remove Background", variant="primary", size="lg")
                rmbg_feedback = gr.Markdown(value="**Ready** - Upload an image and click Remove Background")

            # === Multi-Angles Section ===
            with gr.Column(visible=False) as multiangles_section:
                gr.Markdown("### Multi-Angles Generation (FLUX.2)")
                gr.Markdown("Generate images with controlled camera angles using Multi-Angles LoRA")

                # Reference image (required for consistent subject across angles)
                ma_reference_image = gr.Image(
                    label="Reference Image",
                    type="pil",
                    height=300
                )

                # === Angle Selectors ===
                with gr.Row():
                    ma_azimuth = gr.Dropdown(
                        label="Azimuth (Horizontal Rotation)",
                        choices=[
                            "front view",
                            "front-right quarter view",
                            "right side view",
                            "back-right quarter view",
                            "back view",
                            "back-left quarter view",
                            "left side view",
                            "front-left quarter view"
                        ],
                        value="front view"
                    )
                    ma_elevation = gr.Dropdown(
                        label="Elevation (Vertical Angle)",
                        choices=[
                            "eye-level",
                            "low-angle",
                            "mid-low",
                            "mid-angle",
                            "high-mid",
                            "high-angle",
                            "steep-mid",
                            "steep-angle",
                            "overhead"
                        ],
                        value="eye-level"
                    )
                    ma_distance = gr.Dropdown(
                        label="Distance",
                        choices=["close-up", "medium shot", "wide shot"],
                        value="medium shot"
                    )

                # Preview of composed prompt (auto-updated)
                ma_composed_prompt = gr.Textbox(
                    label="Composed Prompt (auto-generated)",
                    interactive=False,
                    value="<sks> front view eye-level shot medium shot"
                )

                # Parameters
                with gr.Row():
                    ma_steps = gr.Slider(
                        label="Steps",
                        minimum=10,
                        maximum=50,
                        value=28,
                        step=1
                    )
                    ma_lora_scale = gr.Slider(
                        label="LoRA Scale",
                        minimum=0.5,
                        maximum=1.0,
                        value=0.9,
                        step=0.05
                    )
                    ma_seed = gr.Number(
                        label="Seed (0 = random)",
                        value=0,
                        precision=0
                    )

                with gr.Row():
                    ma_width = gr.Slider(
                        label="Width",
                        minimum=512,
                        maximum=2048,
                        value=1024,
                        step=64
                    )
                    ma_height = gr.Slider(
                        label="Height",
                        minimum=512,
                        maximum=2048,
                        value=1024,
                        step=64
                    )
                    ma_quantization = gr.Dropdown(
                        label="Quantization",
                        choices=["qint8", "full"],
                        value="qint8",
                        info="qint8: ~35GB | full: ~115GB"
                    )

                # Generate button
                ma_generate_btn = gr.Button("Generate Multi-Angle Image", variant="primary", size="lg")
                ma_feedback = gr.Markdown(value="**Ready** - Upload a reference image and configure angles")

                gr.Markdown("""
                **Note:** The Multi-Angles LoRA (~317 MB) will be automatically downloaded on first use.
                """)

            # === Specific Processing Event Handlers ===

            # Compose Multi-Angles prompt function
            def compose_multiangles_prompt(azimuth, elevation, distance):
                """Compose the full prompt with <sks> token and angle descriptors."""
                return f"<sks> {azimuth} {elevation} shot {distance}"

            # Toggle visibility between RMBG and Multi-Angles sections
            def update_specific_processing_visibility(processing_type):
                is_rmbg = (processing_type == "Background Removal (RMBG)")
                return [
                    gr.update(visible=is_rmbg),       # rmbg_section
                    gr.update(visible=not is_rmbg)   # multiangles_section
                ]

            specific_processing_type.change(
                fn=update_specific_processing_visibility,
                inputs=specific_processing_type,
                outputs=[rmbg_section, multiangles_section]
            )

            # Auto-update composed prompt when selectors change
            for selector in [ma_azimuth, ma_elevation, ma_distance]:
                selector.change(
                    fn=compose_multiangles_prompt,
                    inputs=[ma_azimuth, ma_elevation, ma_distance],
                    outputs=ma_composed_prompt
                )

            # RMBG button click
            rmbg_btn.click(
                fn=queue_background_removal,
                inputs=rmbg_input,
                outputs=rmbg_feedback
            )

            # Multi-Angles generate button click
            ma_generate_btn.click(
                fn=queue_multiangles_generation,
                inputs=[
                    ma_composed_prompt, ma_reference_image, ma_steps, ma_lora_scale,
                    ma_width, ma_height, ma_seed, ma_quantization
                ],
                outputs=ma_feedback
            )

        # ==============================================================================
        # TAB 3: PROCESSING
        # ==============================================================================
        with gr.Tab("Processing") as processing_tab:
            processing_components = create_processing_tab()
            setup_processing_tab_events(processing_components, image_generator, modelbgrm, demo)

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
        with gr.Tab("History") as history_tab:
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
            
            # Auto-refresh when History tab is selected
            history_tab.select(
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
        # TAB 7: LORA MANAGEMENT
        # ==============================================================================
        from ui.lora_management import create_lora_management_tab, setup_lora_management_events
        
        lora_management_components = create_lora_management_tab()
        
        # ==============================================================================
        # TAB 8: ADMIN
        # ==============================================================================
        with gr.Tab("Admin"):
            gr.Markdown("## Administration and maintenance tools")
            
            # ==============================================================================
            # SYSTEM INFORMATION SECTION
            # ==============================================================================
            with gr.Group():
                gr.Markdown("### 📊 System Information")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(f"**Device:** {image_generator.device}")
                        gr.Markdown(f"**Available LoRA models:** {len(image_generator.lora_data)}")
                        gr.Markdown(f"**Model options:** {', '.join(image_generator.model_options)}")
                    with gr.Column(scale=1):
                        gr.Markdown("") # Empty column for spacing
            
            # ==============================================================================
            # GALLERY SYNCHRONIZATION SECTION
            # ==============================================================================
            with gr.Group():
                gr.Markdown("### 🔄 Gallery Synchronization")
                gr.Markdown("Synchronize the gallery with disk files. This will move orphaned images to `orphaned_pictures/` folder and remove database entries pointing to non-existent files.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        sync_btn = gr.Button("🔄 Sync Gallery with Disk", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        sync_status = gr.Textbox(
                            label="Sync Status",
                            lines=8,
                            interactive=False,
                            placeholder="Click 'Sync Gallery with Disk' to see results..."
                        )
            
            # ==============================================================================
            # MODEL CACHE MANAGEMENT SECTION
            # ==============================================================================
            with gr.Group():
                gr.Markdown("### 🗂️ Model Cache Management")
                gr.Markdown("Manage downloaded models to save disk space and bandwidth.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        cache_info_btn = gr.Button("📊 Show Cache Status", variant="secondary", size="lg")
                    with gr.Column(scale=2):
                        cache_info_display = gr.Textbox(
                            label="Cache Information",
                            lines=12,
                            interactive=False,
                            placeholder="Click 'Show Cache Status' to see model cache information..."
                        )
            
            # ==============================================================================
            # HUGGINGFACE CACHE MANAGEMENT SECTION
            # ==============================================================================
            with gr.Group():
                gr.Markdown("### 🗂️ HuggingFace Cache Management")
                gr.Markdown("View and manage all HuggingFace cached models with selective deletion.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        hf_refresh_btn = gr.Button("🔄 Refresh HF Cache", variant="primary", size="lg")
                        hf_delete_btn = gr.Button("🗑️ Delete Selected Items", variant="stop", size="lg")
                        hf_status_display = gr.Markdown("**Status:** Ready to manage cache")
                    with gr.Column(scale=2):
                        hf_checkbox_group = gr.CheckboxGroup(
                            label="Select HuggingFace cache items to delete",
                            choices=[],
                            value=[],
                            interactive=True,
                            info="Click 'Refresh HF Cache' to load available models"
                        )
            
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
            
        # ==============================================================================
        # LORA MANAGEMENT EVENTS SETUP
        # ==============================================================================
        setup_lora_management_events(lora_management_components)
        
        # Function to refresh LoRA data after management operations
        def refresh_lora_for_generation(sync_state):
            """Refresh LoRA data in the image generator after management operations."""
            try:
                # Only refresh if sync_state is not 0 (meaning a change occurred)
                if sync_state == 0:
                    return gr.update()

                image_generator.refresh_lora_data()

                # Refresh all LoRA dropdown choices
                from ui.lora_manager import refresh_lora_dropdown_choices
                dropdown_update = refresh_lora_dropdown_choices(image_generator.lora_data)

                # Return the update for the Generation tab LoRA dropdown
                return dropdown_update

            except Exception as e:
                # Return empty update in case of error
                return gr.update()

        # Connect the sync state to refresh the Generation tab LoRA dropdown when it changes
        lora_management_components['sync_state'].change(
            fn=refresh_lora_for_generation,
            inputs=[lora_management_components['sync_state']],
            outputs=lora_components['available_dropdown']
        )
        
        # ==============================================================================
        # PROCESSING TAB AUTO-REFRESH ON LOAD
        # ==============================================================================
        # Refresh Processing tab data immediately when the tab is selected
        def refresh_processing_on_load():
            """Refresh processing data when tab is selected."""
            try:
                from ui.processing_tab import update_queue_status, update_current_task
                (status_msg, stats_html, dataframe_rows, dataframe_visible) = update_queue_status()
                current_desc, memory_text = update_current_task()
                dataframe_update = gr.update(value=dataframe_rows, visible=dataframe_visible)
                return (status_msg, stats_html, dataframe_update, current_desc, memory_text)
            except Exception as e:
                # Return empty/default values on error
                return ("⚠️ Error loading processing data", "", gr.update(value=[], visible=False), "No data", "No data")
        
        # Set up tab selection event for Processing tab
        processing_tab.select(
            fn=refresh_processing_on_load,
            outputs=[
                processing_components['status_display'],
                processing_components['queue_stats_html'],
                processing_components['pending_tasks_dataframe'],
                processing_components['current_task_display'],
                processing_components['memory_display']
            ]
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
        "quiet": False,
        "theme": gr.themes.Glass()
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