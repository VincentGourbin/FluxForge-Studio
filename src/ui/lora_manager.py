"""
LoRA Manager UI Module

Provides comprehensive LoRA management interface for all tools (Generation, FLUX Fill, Kontext).
Handles LoRA selection, display, parameter adjustment, and state management.

Features:
- Dynamic LoRA selection with modal interface
- HTML display with descriptions and keywords
- Individual intensity sliders (0.0-1.0 range)
- Add/remove functionality with real-time updates
- Unified interface across all tools

Author: MFLUX Team
"""

import gradio as gr

def create_lora_manager_interface(prefix="", lora_data=None):
    """
    Create a complete LoRA management interface.
    
    Args:
        prefix (str): Prefix for component IDs (e.g., "flux_fill_", "kontext_")
        lora_data (list): Available LoRA models data
        
    Returns:
        dict: Dictionary containing all LoRA interface components
    """
    components = {}
    
    # Add LoRA button
    components['add_btn'] = gr.Button("+ Add LoRA", variant="secondary")
    
    # LoRA selection modal (initially hidden)
    with gr.Group(visible=False) as lora_modal:
        gr.Markdown("### Select a LoRA")
        components['available_dropdown'] = gr.Dropdown(
            label="Available LoRA",
            choices=[f"{lora['file_name']} - {lora['description']}" for lora in lora_data] if lora_data else [],
            value=None
        )
        
        with gr.Row():
            components['confirm_btn'] = gr.Button("Add LoRA", variant="primary")
            components['cancel_btn'] = gr.Button("Cancel", variant="secondary")
    
    components['modal'] = lora_modal
    
    # Selected LoRA display area
    with gr.Group():
        with gr.Row():
            # Individual LoRA removal buttons (hidden by default)
            components['remove_btn_0'] = gr.Button("🗑️ #1", variant="secondary", size="sm", visible=False)
            components['remove_btn_1'] = gr.Button("🗑️ #2", variant="secondary", size="sm", visible=False)
            components['remove_btn_2'] = gr.Button("🗑️ #3", variant="secondary", size="sm", visible=False)
            components['clear_btn'] = gr.Button("🗑️ Clear All", variant="secondary", size="sm")
        
        components['display'] = gr.HTML("No LoRA selected")
        
        # LoRA parameter sliders (shown/hidden based on selection)
        with gr.Column():
            components['params_placeholder'] = gr.Markdown("Select LoRA to see parameters here.", visible=True)
            
            components['strength_1'] = gr.Slider(
                label="LoRA Intensity 1",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.8,
                visible=False
            )
            components['strength_2'] = gr.Slider(
                label="LoRA Intensity 2",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.8,
                visible=False
            )
            components['strength_3'] = gr.Slider(
                label="LoRA Intensity 3",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.8,
                visible=False
            )
    
    # Hidden state for selected LoRA
    components['state'] = gr.State([])
    
    return components

def show_lora_modal():
    """Show the LoRA selection modal."""
    return gr.update(visible=True)

def hide_lora_modal():
    """Hide the LoRA selection modal."""
    return gr.update(visible=False)

def add_lora_to_selection(selected_lora, lora_choice, lora_data):
    """
    Add a LoRA to the selection.
    
    Args:
        selected_lora (list): Current selected LoRA list
        lora_choice (str): Selected LoRA choice from dropdown
        lora_data (list): Available LoRA models data
        
    Returns:
        tuple: (updated_selection, display_html, modal_update)
    """
    if not lora_choice:
        return selected_lora, update_lora_display(selected_lora), gr.update(visible=False)
    
    # Limit to 3 LoRA
    if len(selected_lora) >= 3:
        return selected_lora, update_lora_display(selected_lora), gr.update(visible=False)
    
    # Extract LoRA info
    lora_filename = lora_choice.split(" - ")[0]
    lora_info = next((lora for lora in lora_data if lora['file_name'] == lora_filename), None)
    
    if lora_info:
        new_lora = {
            "id": f"lora_{len(selected_lora)}",
            "name": lora_info['file_name'],
            "description": lora_info['description'],
            "activation_keyword": lora_info['activation_keyword'],
            "strength": 0.8
        }
        selected_lora.append(new_lora)
    
    return selected_lora, update_lora_display(selected_lora), gr.update(visible=False)

def update_lora_display(selected_lora):
    """
    Generate HTML display for selected LoRA.
    
    Args:
        selected_lora (list): List of selected LoRA models
        
    Returns:
        str: HTML string for display
    """
    if not selected_lora:
        return "No LoRA selected"
    
    html = "<div style='border: 1px solid #ddd; padding: 15px; border-radius: 8px; background: #ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>"
    
    for i, lora in enumerate(selected_lora):
        bg_color = "#e8f4fd"
        border_color = "#bee5eb"
        
        html += f"<div style='margin: 8px 0; padding: 12px; background: {bg_color}; border: 1px solid {border_color}; border-radius: 6px;'>"
        html += f"<div style='color: #2c3e50; font-weight: bold; margin-bottom: 6px; font-size: 14px;'>LoRA: {lora['name']}</div>"
        html += f"<div style='color: #2c3e50; font-size: 13px; line-height: 1.4; margin-bottom: 4px; background: rgba(255,255,255,0.8); padding: 4px; border-radius: 3px;'><strong style='color: #1a1a1a;'>Description:</strong> {lora['description']}</div>"
        html += f"<div style='color: #2c3e50; font-size: 12px; background: rgba(255,255,255,0.8); padding: 4px; border-radius: 3px;'><strong style='color: #1a1a1a;'>Keyword:</strong> <code style='background: #2c3e50; color: white; padding: 2px 6px; border-radius: 3px; font-weight: bold;'>{lora['activation_keyword']}</code></div>"
        html += "</div>"
    
    html += "</div>"
    
    return html

def remove_lora_by_index(selected_lora, index):
    """
    Remove a LoRA from selection by index.
    
    Args:
        selected_lora (list): Current selected LoRA list
        index (int): Index of LoRA to remove
        
    Returns:
        tuple: (updated_selection, display_html)
    """
    if 0 <= index < len(selected_lora):
        selected_lora = selected_lora[:index] + selected_lora[index+1:]
    return selected_lora, update_lora_display(selected_lora)

def clear_all_lora():
    """
    Clear all selected LoRA.
    
    Returns:
        tuple: (empty_list, display_html)
    """
    return [], "No LoRA selected"

def update_lora_params_visibility(selected_lora):
    """
    Update visibility of LoRA parameter sliders.
    
    Args:
        selected_lora (list): List of selected LoRA models
        
    Returns:
        tuple: Updates for placeholder and 3 strength sliders
    """
    updates = []
    for i in range(3):
        if i < len(selected_lora):
            lora = selected_lora[i]
            updates.append(gr.update(
                visible=True,
                label=f"{lora['name']} Intensity",
                value=lora.get('strength', 0.8)
            ))
        else:
            updates.append(gr.update(visible=False))
    
    # Placeholder visibility
    placeholder_visible = len(selected_lora) == 0
    placeholder_update = gr.update(visible=placeholder_visible)
    
    return (placeholder_update, *updates)

def setup_lora_events(components, lora_data, prefix=""):
    """
    Set up all event handlers for LoRA management interface.
    
    Args:
        components (dict): Dictionary of LoRA interface components
        lora_data (list): Available LoRA models data
        prefix (str): Prefix for component identification
        
    Returns:
        None (sets up events in place)
    """
    # Show modal
    components['add_btn'].click(
        fn=show_lora_modal,
        outputs=components['modal']
    )
    
    # Hide modal
    components['cancel_btn'].click(
        fn=hide_lora_modal,
        outputs=components['modal']
    )
    
    # Add LoRA with parameter updates
    def add_lora_and_update_params(selected_lora, lora_choice):
        new_lora, new_display, modal_update = add_lora_to_selection(selected_lora, lora_choice, lora_data)
        param_updates = update_lora_params_visibility(new_lora)
        return (new_lora, new_display, modal_update, *param_updates)
    
    components['confirm_btn'].click(
        fn=add_lora_and_update_params,
        inputs=[components['state'], components['available_dropdown']],
        outputs=[
            components['state'], components['display'], components['modal'],
            components['params_placeholder'], components['strength_1'], 
            components['strength_2'], components['strength_3']
        ]
    )
    
    # Individual removal buttons
    for i in range(3):
        components[f'remove_btn_{i}'].click(
            fn=lambda lora, idx=i: remove_lora_by_index(lora, idx),
            inputs=components['state'],
            outputs=[components['state'], components['display']]
        )
    
    # Clear all with parameter updates
    def clear_lora_and_update_params():
        lora, display = clear_all_lora()
        param_updates = update_lora_params_visibility(lora)
        return (lora, display, *param_updates)
    
    components['clear_btn'].click(
        fn=clear_lora_and_update_params,
        outputs=[
            components['state'], components['display'],
            components['params_placeholder'], components['strength_1'], 
            components['strength_2'], components['strength_3']
        ]
    )