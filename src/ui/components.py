"""
UI Components Module

Reusable Gradio interface components and layouts.
Provides standardized UI elements for consistent design across the application.

Features:
- Standardized parameter controls (steps, guidance, etc.)
- Common layout patterns
- Consistent styling and labeling
- Tool selection interfaces

Author: MFLUX Team
"""

import gradio as gr

def create_generation_parameters():
    """
    Create standard generation parameter controls.
    
    Returns:
        dict: Dictionary containing parameter components
    """
    components = {}
    
    with gr.Row():
        components['steps'] = gr.Slider(
            label="Inference Steps",
            minimum=1,
            maximum=50,
            step=1,
            value=20
        )
        
        components['guidance'] = gr.Slider(
            label="Guidance Scale",
            minimum=1.0,
            maximum=50.0,
            step=0.5,
            value=3.5
        )
    
    return components

def create_image_dimensions_controls(default_width=1024, default_height=1024):
    """
    Create image dimension controls.
    
    Args:
        default_width (int): Default width value
        default_height (int): Default height value
        
    Returns:
        dict: Dictionary containing dimension components
    """
    components = {}
    
    with gr.Row():
        components['width'] = gr.Slider(
            label="Width",
            minimum=256,
            maximum=2048,
            step=64,
            value=default_width
        )
        
        components['height'] = gr.Slider(
            label="Height", 
            minimum=256,
            maximum=2048,
            step=64,
            value=default_height
        )
    
    return components

def create_seed_control():
    """
    Create seed control with random generation option.
    
    Returns:
        gr.Number: Seed number input component
    """
    return gr.Number(
        label="Seed (0 = random)",
        value=0,
        minimum=0,
        maximum=2**32-1
    )

def create_prompt_input(label="Prompt", lines=3, placeholder="Describe what you want to generate..."):
    """
    Create standardized prompt input.
    
    Args:
        label (str): Label for the input
        lines (int): Number of text lines
        placeholder (str): Placeholder text
        
    Returns:
        gr.Textbox: Prompt textbox component
    """
    return gr.Textbox(
        label=label,
        lines=lines,
        placeholder=placeholder
    )

def create_model_selector(model_options, default_model="schnell"):
    """
    Create model selection dropdown.
    
    Args:
        model_options (list): Available model options
        default_model (str): Default selected model
        
    Returns:
        gr.Dropdown: Model selection dropdown
    """
    return gr.Dropdown(
        label="Model",
        choices=model_options,
        value=default_model
    )

def create_quantization_selector():
    """
    Create quantization selection dropdown with tested MPS compatibility.
    
    Returns:
        gr.Dropdown: Quantization selection dropdown
    """
    return gr.Dropdown(
        label="Quantization - Memory optimisation",
        choices=["None", "8-bit"],
        value="8-bit",
        info="8-bit: ~70% memory reduction, None: no optimization"
    )

def create_controlnet_controls():
    """
    Create ControlNet parameter controls.
    
    Returns:
        dict: Dictionary containing ControlNet components
    """
    components = {}
    
    components['image'] = gr.File(
        label="Control Image",
        file_types=["image"]
    )
    
    with gr.Row():
        components['strength'] = gr.Slider(
            label="ControlNet Strength",
            minimum=0.0,
            maximum=2.0,
            step=0.1,
            value=1.0
        )
        
        components['save_canny'] = gr.Checkbox(
            label="Save Canny Edge Image",
            value=False
        )
    
    # Canny-specific parameters
    with gr.Group(visible=False) as canny_params:
        gr.Markdown("**Canny Edge Detection Parameters**")
        with gr.Row():
            components['canny_low'] = gr.Slider(
                label="Low Threshold",
                minimum=1,
                maximum=255,
                step=1,
                value=100
            )
            
            components['canny_high'] = gr.Slider(
                label="High Threshold", 
                minimum=1,
                maximum=255,
                step=1,
                value=200
            )
    
    components['canny_params'] = canny_params
    
    # Preview image
    components['preview'] = gr.Image(
        label="Control Preview",
        height=200,
        visible=False
    )
    
    return components

def create_post_processing_selector(options):
    """
    Create post-processing type selector.
    
    Args:
        options (list): Available post-processing options
        
    Returns:
        gr.Dropdown: Post-processing selection dropdown
    """
    return gr.Dropdown(
        label="Processing Type",
        choices=options,
        value="None"
    )

def create_expansion_controls():
    """
    Create outpainting expansion controls.
    
    Returns:
        dict: Dictionary containing expansion sliders
    """
    components = {}
    
    gr.Markdown("**Expansion Percentages:**")
    
    with gr.Row():
        components['top'] = gr.Slider(
            label="Top (%)",
            minimum=0,
            maximum=100,
            step=5,
            value=25
        )
        
        components['bottom'] = gr.Slider(
            label="Bottom (%)",
            minimum=0,
            maximum=100,
            step=5,
            value=25
        )
    
    with gr.Row():
        components['left'] = gr.Slider(
            label="Left (%)",
            minimum=0,
            maximum=100,
            step=5,
            value=25
        )
        
        components['right'] = gr.Slider(
            label="Right (%)",
            minimum=0,
            maximum=100,
            step=5,
            value=25
        )
    
    return components

def create_image_editor_component(label="Image Editor", height=400):
    """
    Create ImageEditor component for inpainting.
    
    Args:
        label (str): Component label
        height (int): Component height
        
    Returns:
        gr.ImageEditor: Configured image editor
    """
    return gr.ImageEditor(
        label=label,
        type="pil",
        brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"),
        height=height
    )

def create_preview_image(label="Preview", height=300):
    """
    Create preview image component.
    
    Args:
        label (str): Component label
        height (int): Component height
        
    Returns:
        gr.Image: Preview image component
    """
    return gr.Image(
        label=label,
        height=height
    )

def create_output_image(label="Generated Image", height=500):
    """
    Create output image component.
    
    Args:
        label (str): Component label
        height (int): Component height
        
    Returns:
        gr.Image: Output image component
    """
    return gr.Image(
        label=label,
        height=height
    )

def create_generation_button(text="🎨 Generate", variant="primary", size="lg"):
    """
    Create standardized generation button.
    
    Args:
        text (str): Button text
        variant (str): Button variant (primary, secondary)
        size (str): Button size (sm, lg)
        
    Returns:
        gr.Button: Generation button component
    """
    return gr.Button(
        text,
        variant=variant,
        size=size
    )

def create_tool_selector_modal(tool_types, available_choices):
    """
    Create tool selection modal interface.
    
    Args:
        tool_types (list): Available tool types
        available_choices (dict): Available choices for each tool type
        
    Returns:
        dict: Dictionary containing modal components
    """
    components = {}
    
    with gr.Group(visible=False) as modal:
        gr.Markdown("### Select a Tool")
        
        components['type_selector'] = gr.Dropdown(
            label="Tool Type",
            choices=tool_types,
            value=tool_types[0] if tool_types else None
        )
        
        # Create selectors for each tool type
        for tool_type in tool_types:
            selector_key = f"{tool_type.lower()}_selector"
            visible = tool_type == tool_types[0] if tool_types else False
            
            with gr.Group(visible=visible) as selector_group:
                gr.Markdown(f"**Select a {tool_type}**")
                components[selector_key] = gr.Dropdown(
                    label=f"Available {tool_type}",
                    choices=available_choices.get(tool_type, []),
                    value=None
                )
            
            components[f"{selector_key}_group"] = selector_group
        
        with gr.Row():
            components['confirm_btn'] = gr.Button("Add Tool", variant="primary")
            components['cancel_btn'] = gr.Button("Cancel", variant="secondary")
    
    components['modal'] = modal
    
    return components

