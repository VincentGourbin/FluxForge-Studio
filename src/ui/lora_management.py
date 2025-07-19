"""
LoRA Management UI Components

This module provides the user interface for managing LoRA models including:
- Adding new LoRA models (file upload and metadata)
- Editing existing LoRA models
- Deleting LoRA models
- Viewing LoRA model information
- Refreshing the LoRA list

Features:
- Upload .safetensors files to the lora directory
- Edit descriptions and activation keywords
- Delete LoRA models from database and optionally from disk
- Real-time refresh of the LoRA list
- File size and status information
"""

import gradio as gr
import os
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from core.database import (
    get_all_lora, add_lora, update_lora, delete_lora, 
    get_lora_by_id, refresh_lora_file_sizes
)

def format_file_size(size_bytes):
    """
    Format file size in human readable format.
    
    Args:
        size_bytes (int): File size in bytes
        
    Returns:
        str: Formatted file size
    """
    if size_bytes is None:
        return "Unknown"
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def get_lora_dataframe():
    """
    Get LoRA data formatted for Gradio dataframe display.
    
    Returns:
        list: List of lists for dataframe display
    """
    lora_list = get_all_lora()
    
    dataframe_data = []
    for lora in lora_list:
        # Check if file exists
        lora_file_path = Path('lora') / lora['file_name']
        status = "✅ Exists" if lora_file_path.exists() else "❌ Missing"
        
        # Format file size
        file_size = format_file_size(lora['file_size'])
        
        # Format activation keyword
        activation_keyword = lora['activation_keyword'] or "-"
        
        dataframe_data.append([
            False,  # Checkbox for selection
            lora['id'],  # Hidden ID for operations
            lora['file_name'],
            lora['description'][:50] + "..." if len(lora['description']) > 50 else lora['description'],
            activation_keyword,
            file_size,
            status
        ])
    
    return dataframe_data

def create_lora_management_tab():
    """
    Create the LoRA Management tab interface.
    
    Returns:
        dict: Dictionary containing all UI components for event setup
    """
    
    with gr.Tab("LoRA Management"):
        gr.Markdown("## 🎨 LoRA Management")
        gr.Markdown("**Manage your LoRA models** - Add, edit, and delete LoRA models with a graphical interface.")
        
        # ==============================================================================
        # LORA LIST SECTION
        # ==============================================================================
        with gr.Group():
            gr.Markdown("### 📋 Current LoRA Models")
            
            # Refresh buttons above dataframe
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh List", variant="secondary", size="lg")
                refresh_files_btn = gr.Button("📁 Refresh File Sizes", variant="secondary", size="lg")
                status_display = gr.Markdown("**Status:** Ready")
            
            # Dataframe
            lora_dataframe = gr.Dataframe(
                value=get_lora_dataframe(),
                headers=["Select", "ID", "File Name", "Description", "Activation Keyword", "File Size", "Status"],
                datatype=["bool", "number", "str", "str", "str", "str", "str"],
                label="LoRA Models - Edit descriptions and keywords directly, check boxes to select for delete",
                interactive=True,
                row_count=(0, "dynamic"),
                col_count=7,
                wrap=True,
                column_widths=[60, 0, 200, 300, 150, 100, 100]  # Hide ID column by setting width to 0
            )
            
            # Action buttons below dataframe
            with gr.Row():
                save_modifications_btn = gr.Button("💾 Save Modifications", variant="primary", size="lg")
                delete_selected_btn = gr.Button("🗑️ Delete Selected", variant="stop", size="lg")
            
            # Selection helper buttons
            with gr.Row():
                select_all_btn = gr.Button("☑️ Select All", variant="secondary", size="sm")
                clear_selection_btn = gr.Button("☐ Clear Selection", variant="secondary", size="sm")
        
        # ==============================================================================
        # ADD NEW LORA SECTION
        # ==============================================================================
        with gr.Group():
            gr.Markdown("### ➕ Add New LoRA")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Upload LoRA File**")
                    file_upload = gr.File(
                        label="Upload .safetensors file",
                        file_types=[".safetensors"],
                        type="filepath"
                    )
                    
                    add_description = gr.Textbox(
                        label="Description",
                        placeholder="Enter a description for this LoRA model",
                        lines=3
                    )
                    
                    add_activation_keyword = gr.Textbox(
                        label="Activation Keyword",
                        placeholder="Enter activation keyword (optional)"
                    )
                    
                    add_btn = gr.Button("➕ Add LoRA", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    gr.Markdown("**Instructions**")
                    gr.Markdown("""
                    1. **Upload File**: Select a .safetensors LoRA file
                    2. **Add Description**: Describe what this LoRA does
                    3. **Set Keyword**: Add trigger words (optional)
                    4. **Click Add**: The file will be saved to the lora directory
                    
                    **Note**: The file will be automatically copied to the `lora/` directory.
                    """)
        
        # ==============================================================================
        # EDIT LORA SECTION (Hidden by default, shown when editing)
        # ==============================================================================
        with gr.Group(visible=False) as edit_group:
            gr.Markdown("### ✏️ Edit Selected LoRA")
            
            with gr.Row():
                with gr.Column(scale=1):
                    edit_description = gr.Textbox(
                        label="Description",
                        placeholder="LoRA description will appear here",
                        lines=3
                    )
                    
                    edit_activation_keyword = gr.Textbox(
                        label="Activation Keyword",
                        placeholder="Activation keyword will appear here"
                    )
                    
                    with gr.Row():
                        update_btn = gr.Button("💾 Update LoRA", variant="primary", size="lg")
                        cancel_edit_btn = gr.Button("❌ Cancel Edit", variant="secondary", size="lg")
                
                with gr.Column(scale=1):
                    edit_info_display = gr.Markdown("**LoRA information will appear here**")
        
        # ==============================================================================
        # DELETE CONFIRMATION SECTION
        # ==============================================================================
        with gr.Group():
            gr.Markdown("### 🗑️ Delete Confirmation")
            
            delete_info_display = gr.Markdown("**Click 'Delete Selected' to see selected LoRA(s) here**")
            
            with gr.Row():
                delete_file_checkbox = gr.Checkbox(
                    label="Also delete file(s) from disk",
                    value=False,
                    info="Check this to also remove the .safetensors file(s)",
                    visible=False
                )
                
            with gr.Row():
                confirm_delete_btn = gr.Button("🗑️ Confirm Delete", variant="stop", size="lg", visible=False)
                cancel_delete_btn = gr.Button("❌ Cancel Delete", variant="secondary", size="lg", visible=False)
        
        # Hidden state to trigger dropdown refresh
        sync_state = gr.State(value=0)
    
    return {
        'lora_dataframe': lora_dataframe,
        'refresh_btn': refresh_btn,
        'refresh_files_btn': refresh_files_btn,
        'status_display': status_display,
        'save_modifications_btn': save_modifications_btn,
        'delete_selected_btn': delete_selected_btn,
        'select_all_btn': select_all_btn,
        'clear_selection_btn': clear_selection_btn,
        'file_upload': file_upload,
        'add_description': add_description,
        'add_activation_keyword': add_activation_keyword,
        'add_btn': add_btn,
        'edit_group': edit_group,
        'edit_description': edit_description,
        'edit_activation_keyword': edit_activation_keyword,
        'update_btn': update_btn,
        'cancel_edit_btn': cancel_edit_btn,
        'edit_info_display': edit_info_display,
        'delete_file_checkbox': delete_file_checkbox,
        'delete_info_display': delete_info_display,
        'confirm_delete_btn': confirm_delete_btn,
        'cancel_delete_btn': cancel_delete_btn,
        'sync_state': sync_state
    }

def refresh_lora_list():
    """
    Refresh the LoRA list display.
    
    Returns:
        tuple: (dataframe_data, status_message)
    """
    try:
        dataframe_data = get_lora_dataframe()
        count = len(dataframe_data)
        status_message = f"**Status:** List refreshed - {count} LoRA models found"
        return dataframe_data, status_message
    except Exception as e:
        return [], f"**Status:** Error refreshing list - {str(e)}"

def refresh_file_sizes():
    """
    Refresh file sizes for all LoRA entries.
    
    Returns:
        tuple: (dataframe_data, status_message)
    """
    try:
        success, message, updated_count = refresh_lora_file_sizes()
        if success:
            dataframe_data = get_lora_dataframe()
            status_message = f"**Status:** {message}"
            return dataframe_data, status_message
        else:
            return get_lora_dataframe(), f"**Status:** Error - {message}"
    except Exception as e:
        return get_lora_dataframe(), f"**Status:** Error refreshing file sizes - {str(e)}"

def upload_and_add_lora(file_path, description, activation_keyword):
    """
    Upload a LoRA file and add it to the database.
    
    Args:
        file_path (str): Path to the uploaded file
        description (str): Description of the LoRA
        activation_keyword (str): Activation keyword
        
    Returns:
        tuple: (dataframe_data, status_message, cleared_inputs, sync_trigger)
    """
    try:
        if not file_path:
            return get_lora_dataframe(), "**Status:** No file selected", "", ""
        
        if not description.strip():
            return get_lora_dataframe(), "**Status:** Description is required", file_path, description
        
        # Get filename
        file_name = os.path.basename(file_path)
        
        # Check if it's a .safetensors file
        if not file_name.lower().endswith('.safetensors'):
            return get_lora_dataframe(), "**Status:** File must be a .safetensors file", file_path, description
        
        # Create lora directory if it doesn't exist
        lora_dir = Path('lora')
        lora_dir.mkdir(exist_ok=True)
        
        # Copy file to lora directory
        destination = lora_dir / file_name
        
        if destination.exists():
            return get_lora_dataframe(), f"**Status:** File '{file_name}' already exists in lora directory", file_path, description
        
        shutil.copy2(file_path, destination)
        
        # Add to database
        success, message, lora_id = add_lora(file_name, description.strip(), activation_keyword.strip())
        
        if success:
            dataframe_data = get_lora_dataframe()
            status_message = f"**Status:** {message}"
            # Trigger sync with timestamp
            return dataframe_data, status_message, None, "", "", time.time()  # Clear inputs + trigger sync
        else:
            # Remove file if database addition failed
            if destination.exists():
                destination.unlink()
            return get_lora_dataframe(), f"**Status:** {message}", file_path, description, "", 0
    
    except Exception as e:
        return get_lora_dataframe(), f"**Status:** Error uploading LoRA - {str(e)}", file_path, description, "", 0

def load_lora_details(lora_id):
    """
    Load LoRA details for editing.
    
    Args:
        lora_id (float): ID of the LoRA to load
        
    Returns:
        tuple: (description, activation_keyword, info_display)
    """
    try:
        if not lora_id or lora_id <= 0:
            return "", "", "**Enter a valid LoRA ID and click 'Load LoRA Details'**"
        
        lora_id = int(lora_id)
        lora = get_lora_by_id(lora_id)
        
        if not lora:
            return "", "", f"**LoRA with ID {lora_id} not found**"
        
        # Format info display
        lora_file_path = Path('lora') / lora['file_name']
        file_status = "✅ Exists" if lora_file_path.exists() else "❌ Missing"
        file_size = format_file_size(lora['file_size'])
        
        info_display = f"""
        **LoRA Details**
        - **ID:** {lora['id']}
        - **File Name:** {lora['file_name']}
        - **File Size:** {file_size}
        - **Status:** {file_status}
        - **Created:** {lora['created_at'][:19]}
        - **Updated:** {lora['updated_at'][:19]}
        """
        
        return lora['description'], lora['activation_keyword'] or "", info_display
        
    except Exception as e:
        return "", "", f"**Error loading LoRA details: {str(e)}**"

def update_lora_details(lora_id, description, activation_keyword):
    """
    Update LoRA details in the database.
    
    Args:
        lora_id (float): ID of the LoRA to update
        description (str): New description
        activation_keyword (str): New activation keyword
        
    Returns:
        tuple: (dataframe_data, status_message)
    """
    try:
        if not lora_id or lora_id <= 0:
            return get_lora_dataframe(), "**Status:** Invalid LoRA ID"
        
        if not description.strip():
            return get_lora_dataframe(), "**Status:** Description is required"
        
        lora_id = int(lora_id)
        success, message = update_lora(lora_id, description.strip(), activation_keyword.strip())
        
        if success:
            dataframe_data = get_lora_dataframe()
            status_message = f"**Status:** {message}"
            return dataframe_data, status_message
        else:
            return get_lora_dataframe(), f"**Status:** {message}"
    
    except Exception as e:
        return get_lora_dataframe(), f"**Status:** Error updating LoRA - {str(e)}"

def delete_lora_from_system(lora_id, delete_file):
    """
    Delete LoRA from the database and optionally from disk.
    
    Args:
        lora_id (float): ID of the LoRA to delete
        delete_file (bool): Whether to also delete the file
        
    Returns:
        tuple: (dataframe_data, status_message)
    """
    try:
        if not lora_id or lora_id <= 0:
            return get_lora_dataframe(), "**Status:** Invalid LoRA ID"
        
        lora_id = int(lora_id)
        success, message = delete_lora(lora_id, delete_file)
        
        if success:
            dataframe_data = get_lora_dataframe()
            status_message = f"**Status:** {message}"
            return dataframe_data, status_message
        else:
            return get_lora_dataframe(), f"**Status:** {message}"
    
    except Exception as e:
        return get_lora_dataframe(), f"**Status:** Error deleting LoRA - {str(e)}"

def select_all_loras():
    """
    Select all LoRAs in the dataframe.
    
    Returns:
        list: Dataframe data with all checkboxes selected
    """
    dataframe_data = get_lora_dataframe()
    # Set all checkboxes to True
    for row in dataframe_data:
        if len(row) > 0:
            row[0] = True
    return dataframe_data

def clear_lora_selection():
    """
    Clear all LoRA selections in the dataframe.
    
    Returns:
        list: Dataframe data with all checkboxes cleared
    """
    dataframe_data = get_lora_dataframe()
    # Set all checkboxes to False
    for row in dataframe_data:
        if len(row) > 0:
            row[0] = False
    return dataframe_data

def get_selected_loras(dataframe_data):
    """
    Get selected LoRAs from dataframe data.
    
    Args:
        dataframe_data: Dataframe data with selections
        
    Returns:
        list: List of selected LoRA IDs
    """
    selected_ids = []
    try:
        if dataframe_data is not None:
            # Handle dataframe data properly
            if hasattr(dataframe_data, 'values'):
                rows = dataframe_data.values.tolist()
            else:
                rows = dataframe_data
            
            for row in rows:
                if len(row) >= 2:
                    # Check if checkbox is selected
                    checkbox_value = row[0]
                    is_selected = checkbox_value is True or checkbox_value == 'true' or checkbox_value == True
                    
                    if is_selected:
                        selected_ids.append(int(row[1]))  # ID is in column 1
    except Exception as e:
        # Log error silently, return empty list
        pass
    
    return selected_ids

def start_edit_selected(dataframe_data):
    """
    Start editing the selected LoRA (only allows one selection for edit).
    
    Args:
        dataframe_data: Dataframe data with selections
        
    Returns:
        tuple: (edit_group_visibility, description, activation_keyword, info_display, status_message)
    """
    try:
        selected_ids = get_selected_loras(dataframe_data)
        
        if not selected_ids:
            return False, "", "", "**No LoRA selected for editing**", "**Status:** Please select exactly one LoRA to edit"
        
        if len(selected_ids) > 1:
            return False, "", "", "**Multiple LoRAs selected**", "**Status:** Please select only one LoRA to edit"
        
        # Load the selected LoRA
        lora_id = selected_ids[0]
        lora = get_lora_by_id(lora_id)
        
        if not lora:
            return False, "", "", f"**LoRA with ID {lora_id} not found**", "**Status:** LoRA not found"
        
        # Format info display
        lora_file_path = Path('lora') / lora['file_name']
        file_status = "✅ Exists" if lora_file_path.exists() else "❌ Missing"
        file_size = format_file_size(lora['file_size'])
        
        info_display = f"""
        **Editing LoRA**
        - **ID:** {lora['id']}
        - **File Name:** {lora['file_name']}
        - **File Size:** {file_size}
        - **Status:** {file_status}
        - **Created:** {lora['created_at'][:19]}
        - **Updated:** {lora['updated_at'][:19]}
        """
        
        return True, lora['description'], lora['activation_keyword'] or "", info_display, f"**Status:** Editing LoRA '{lora['file_name']}'"
        
    except Exception as e:
        return False, "", "", f"**Error loading LoRA: {str(e)}**", f"**Status:** Error - {str(e)}"

def start_delete_selected(dataframe_data):
    """
    Start deleting the selected LoRA(s).
    
    Args:
        dataframe_data: Dataframe data with selections
        
    Returns:
        tuple: (delete_info_display, delete_file_checkbox_visibility, confirm_btn_visibility, cancel_btn_visibility, status_message)
    """
    try:
        selected_ids = get_selected_loras(dataframe_data)
        
        if not selected_ids:
            return ("**No LoRA selected for deletion - Please select at least one LoRA using the checkboxes**", 
                   gr.update(visible=False), 
                   gr.update(visible=False), 
                   gr.update(visible=False), 
                   "**Status:** Please select at least one LoRA to delete")
        
        # Get LoRA information for selected IDs
        selected_loras = []
        for lora_id in selected_ids:
            lora = get_lora_by_id(lora_id)
            if lora:
                selected_loras.append(lora)
        
        if not selected_loras:
            return ("**Selected LoRAs not found**", 
                   gr.update(visible=False), 
                   gr.update(visible=False), 
                   gr.update(visible=False), 
                   "**Status:** Selected LoRAs not found")
        
        # Format delete info display
        info_lines = [f"**⚠️ Ready to delete {len(selected_loras)} LoRA(s):**"]
        for lora in selected_loras:
            lora_file_path = Path('lora') / lora['file_name']
            file_status = "✅ Exists" if lora_file_path.exists() else "❌ Missing"
            file_size = format_file_size(lora['file_size'])
            
            info_lines.append(f"- **{lora['file_name']}** ({file_size}) - {file_status}")
        
        info_lines.append("\n**⚠️ This action cannot be undone!**")
        delete_info_display = "\n".join(info_lines)
        
        return (delete_info_display, 
               gr.update(visible=True), 
               gr.update(visible=True), 
               gr.update(visible=True), 
               f"**Status:** Ready to delete {len(selected_loras)} LoRA(s) - Choose options below")
        
    except Exception as e:
        return (f"**Error preparing deletion: {str(e)}**", 
               gr.update(visible=False), 
               gr.update(visible=False), 
               gr.update(visible=False), 
               f"**Status:** Error - {str(e)}")

def confirm_delete_selected(dataframe_data, delete_file):
    """
    Confirm deletion of selected LoRA(s).
    
    Args:
        dataframe_data: Dataframe data with selections
        delete_file (bool): Whether to also delete files from disk
        
    Returns:
        tuple: (delete_info_display, delete_file_checkbox_visibility, confirm_btn_visibility, cancel_btn_visibility, dataframe_data, status_message)
    """
    try:
        selected_ids = get_selected_loras(dataframe_data)
        
        if not selected_ids:
            return ("**Click 'Delete Selected' to see selected LoRA(s) here**", 
                   gr.update(visible=False), 
                   gr.update(visible=False), 
                   gr.update(visible=False), 
                   get_lora_dataframe(), 
                   "**Status:** No LoRA selected for deletion")
        
        # Delete each selected LoRA
        deleted_count = 0
        errors = []
        
        for lora_id in selected_ids:
            success, message = delete_lora(lora_id, delete_file)
            if success:
                deleted_count += 1
            else:
                errors.append(f"ID {lora_id}: {message}")
        
        # Format result message
        if deleted_count > 0:
            result_msg = f"**Status:** Successfully deleted {deleted_count} LoRA(s)"
            if errors:
                result_msg += f" (Errors: {len(errors)})"
        else:
            result_msg = f"**Status:** No LoRAs deleted. Errors: {len(errors)}"
        
        return ("**✅ Deletion completed - Click 'Delete Selected' to delete more LoRAs**", 
               gr.update(visible=False), 
               gr.update(visible=False), 
               gr.update(visible=False), 
               get_lora_dataframe(), 
               result_msg,
               time.time())  # Trigger sync
        
    except Exception as e:
        return ("**❌ Error during deletion**", 
               gr.update(visible=False), 
               gr.update(visible=False), 
               gr.update(visible=False), 
               get_lora_dataframe(), 
               f"**Status:** Error deleting LoRAs - {str(e)}",
               0)

def cancel_edit():
    """
    Cancel editing and hide the edit group.
    
    Returns:
        str: status_message
    """
    return "**Status:** Edit cancelled"

def cancel_delete():
    """
    Cancel deletion and hide the delete controls.
    
    Returns:
        tuple: (delete_info_display, delete_file_checkbox_visibility, confirm_btn_visibility, cancel_btn_visibility, status_message)
    """
    return ("**Click 'Delete Selected' to see selected LoRA(s) here**", 
           gr.update(visible=False), 
           gr.update(visible=False), 
           gr.update(visible=False), 
           "**Status:** Delete cancelled")

def update_selected_lora(dataframe_data, description, activation_keyword):
    """
    Update the selected LoRA with new information.
    
    Args:
        dataframe_data: Dataframe data with selections
        description (str): New description
        activation_keyword (str): New activation keyword
        
    Returns:
        tuple: (dataframe_data, status_message)
    """
    try:
        selected_ids = get_selected_loras(dataframe_data)
        
        if not selected_ids:
            return get_lora_dataframe(), "**Status:** No LoRA selected"
        
        if len(selected_ids) > 1:
            return get_lora_dataframe(), "**Status:** Multiple LoRAs selected, please select only one"
        
        if not description.strip():
            return get_lora_dataframe(), "**Status:** Description is required"
        
        lora_id = selected_ids[0]
        success, message = update_lora(lora_id, description.strip(), activation_keyword.strip())
        
        if success:
            return get_lora_dataframe(), f"**Status:** {message}"
        else:
            return get_lora_dataframe(), f"**Status:** {message}"
    
    except Exception as e:
        return get_lora_dataframe(), f"**Status:** Error updating LoRA - {str(e)}"

def save_dataframe_modifications(dataframe_data):
    """
    Save modifications made directly in the dataframe.
    
    Args:
        dataframe_data: Modified dataframe data
        
    Returns:
        tuple: (dataframe_data, status_message)
    """
    try:
        # Handle dataframe data properly
        if hasattr(dataframe_data, 'values'):
            rows = dataframe_data.values.tolist()
        elif isinstance(dataframe_data, list):
            rows = dataframe_data
        else:
            return get_lora_dataframe(), "**Status:** Invalid dataframe format"
        
        if not rows or len(rows) == 0:
            return get_lora_dataframe(), "**Status:** No data to save"
        
        updated_count = 0
        errors = []
        
        for i, row in enumerate(rows):
            if len(row) >= 5:  # Ensure we have enough columns
                try:
                    # Extract data from row with proper type handling
                    lora_id = int(float(row[1]))  # ID in column 1 (handle float conversion)
                    description = str(row[3]).strip() if row[3] is not None else ""  # Description in column 3
                    activation_keyword = str(row[4]).strip() if row[4] is not None else ""  # Activation keyword in column 4
                    
                    # Skip if description is empty
                    if not description:
                        errors.append(f"Row {i+1} (ID {lora_id}): Description cannot be empty")
                        continue
                    
                    # Get current LoRA data to check if changes were made
                    current_lora = get_lora_by_id(lora_id)
                    if not current_lora:
                        errors.append(f"Row {i+1} (ID {lora_id}): LoRA not found")
                        continue
                    
                    # Check if there are actual changes
                    current_desc = current_lora['description']
                    current_keyword = current_lora['activation_keyword'] or ""
                    
                    if description != current_desc or activation_keyword != current_keyword:
                        # Update the LoRA
                        success, message = update_lora(lora_id, description, activation_keyword)
                        if success:
                            updated_count += 1
                        else:
                            errors.append(f"Row {i+1} (ID {lora_id}): {message}")
                            
                except Exception as e:
                    errors.append(f"Row {i+1} processing error: {str(e)}")
        
        # Format result message
        if updated_count > 0:
            result_msg = f"**Status:** Successfully updated {updated_count} LoRA(s)"
            if errors:
                result_msg += f" ({len(errors)} errors)"
        elif errors:
            result_msg = f"**Status:** No updates made - {len(errors)} error(s)"
        else:
            result_msg = "**Status:** No changes detected"
        
        # Trigger sync if any updates were made
        sync_trigger = time.time() if updated_count > 0 else 0
        return get_lora_dataframe(), result_msg, sync_trigger
        
    except Exception as e:
        return get_lora_dataframe(), f"**Status:** Error saving modifications - {str(e)}", 0

def setup_lora_management_events(components: Dict[str, Any], dropdown_components=None):
    """
    Set up event handlers for the LoRA Management tab.
    
    Args:
        components (dict): Dictionary of UI components
    """
    
    # Refresh list button
    components['refresh_btn'].click(
        fn=refresh_lora_list,
        outputs=[components['lora_dataframe'], components['status_display']]
    )
    
    # Refresh file sizes button
    components['refresh_files_btn'].click(
        fn=refresh_file_sizes,
        outputs=[components['lora_dataframe'], components['status_display']]
    )
    
    # Selection helper buttons
    components['select_all_btn'].click(
        fn=select_all_loras,
        outputs=[components['lora_dataframe']]
    )
    
    components['clear_selection_btn'].click(
        fn=clear_lora_selection,
        outputs=[components['lora_dataframe']]
    )
    
    # Save modifications button
    components['save_modifications_btn'].click(
        fn=save_dataframe_modifications,
        inputs=[components['lora_dataframe']],
        outputs=[
            components['lora_dataframe'],
            components['status_display'],
            components['sync_state']
        ]
    )
    
    # Delete selected button
    components['delete_selected_btn'].click(
        fn=start_delete_selected,
        inputs=[components['lora_dataframe']],
        outputs=[
            components['delete_info_display'],
            components['delete_file_checkbox'],
            components['confirm_delete_btn'],
            components['cancel_delete_btn'],
            components['status_display']
        ]
    )
    
    # Cancel edit button
    components['cancel_edit_btn'].click(
        fn=cancel_edit,
        outputs=[components['status_display']]
    )
    
    # Cancel delete button
    components['cancel_delete_btn'].click(
        fn=cancel_delete,
        outputs=[
            components['delete_info_display'],
            components['delete_file_checkbox'],
            components['confirm_delete_btn'],
            components['cancel_delete_btn'],
            components['status_display']
        ]
    )
    
    # Update LoRA button
    components['update_btn'].click(
        fn=update_selected_lora,
        inputs=[
            components['lora_dataframe'],
            components['edit_description'],
            components['edit_activation_keyword']
        ],
        outputs=[
            components['lora_dataframe'],
            components['status_display']
        ]
    )
    
    # Confirm delete button
    components['confirm_delete_btn'].click(
        fn=confirm_delete_selected,
        inputs=[
            components['lora_dataframe'],
            components['delete_file_checkbox']
        ],
        outputs=[
            components['delete_info_display'],
            components['delete_file_checkbox'],
            components['confirm_delete_btn'],
            components['cancel_delete_btn'],
            components['lora_dataframe'],
            components['status_display'],
            components['sync_state']
        ]
    )
    
    # Add LoRA button
    components['add_btn'].click(
        fn=upload_and_add_lora,
        inputs=[
            components['file_upload'],
            components['add_description'],
            components['add_activation_keyword']
        ],
        outputs=[
            components['lora_dataframe'],
            components['status_display'],
            components['file_upload'],
            components['add_description'],
            components['add_activation_keyword'],
            components['sync_state']
        ]
    )
