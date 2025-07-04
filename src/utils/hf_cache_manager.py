"""
HuggingFace Cache Management Module

Provides functions to scan, display, and manage HuggingFace model cache.
Includes selective deletion capabilities with space calculation.

Features:
- Complete cache scanning with detailed metadata
- Checkbox-formatted display for Gradio integration  
- Safe deletion with freed space reporting
- Error handling and validation

Author: FluxForge Team
"""

import gradio as gr
from huggingface_hub import scan_cache_dir
import datetime


def format_last_modified(last_modified):
    """Format last_modified which can be datetime or timestamp."""
    if not last_modified:
        return 'Unknown'
    try:
        if hasattr(last_modified, 'strftime'):
            # It's a datetime object
            return last_modified.strftime('%Y-%m-%d %H:%M')
        else:
            # It's likely a timestamp (float)
            dt = datetime.datetime.fromtimestamp(last_modified)
            return dt.strftime('%Y-%m-%d %H:%M')
    except:
        return 'Unknown'


def scan_hf_cache():
    """
    Scan HuggingFace cache and create checkbox-formatted list.
    
    Returns:
        tuple: (checkbox_choices, status_message, cache_info)
            - checkbox_choices: List of formatted strings for CheckboxGroup
            - status_message: Summary status with totals
            - cache_info: Raw cache information for deletion operations
    """
    try:
        cache_info = scan_cache_dir()
        checkbox_choices = []
        
        for repo in list(cache_info.repos):
            for revision in repo.revisions:
                # Calculate size
                size_gb = revision.size_on_disk / (1024**3)
                size_mb = revision.size_on_disk / (1024**2)
                
                if size_gb >= 1:
                    size_str = f"{size_gb:.2f} GB"
                else:
                    size_str = f"{size_mb:.1f} MB"
                
                # Format last modified
                last_mod_str = format_last_modified(revision.last_modified)
                
                # Create checkbox label with all info
                choice_label = f"{repo.repo_id} | {revision.commit_hash[:12]}... | {size_str} | {last_mod_str}"
                checkbox_choices.append(choice_label)
        
        # Sort by size (largest first)
        checkbox_choices.sort(key=lambda x: float(x.split(' | ')[2].split()[0]), reverse=True)
        
        total_repos = len(list(cache_info.repos))
        total_revisions = len(checkbox_choices)
        total_size = sum(repo.size_on_disk for repo in list(cache_info.repos))
        total_size_gb = total_size / (1024**3)
        
        status = f"**Status:** Found {total_repos} repositories, {total_revisions} revisions, {total_size_gb:.2f} GB total"
        
        return checkbox_choices, status, cache_info
        
    except Exception as e:
        return [f"Error: {str(e)}"], f"**Status:** ❌ Error scanning cache: {str(e)}", None


def delete_selected_hf_items(selected_checkboxes, cache_info_state):
    """
    Delete selected HuggingFace cache items.
    
    Args:
        selected_checkboxes: List of selected checkbox labels
        cache_info_state: Cache information from scan_cache_dir()
    
    Returns:
        tuple: (updated_checkbox_choices, status_message, updated_cache_info)
    """
    try:
        if not cache_info_state:
            return gr.update(), "**Status:** ❌ No cache info available. Please refresh first.", cache_info_state
        
        if not selected_checkboxes:
            return gr.update(), "**Status:** ⚠️ No items selected for deletion.", cache_info_state
        
        # Parse selected items and find corresponding revisions
        selected_revisions = []
        
        for checkbox_label in selected_checkboxes:
            try:
                # Parse format: "repo_id | hash... | size | date"
                parts = checkbox_label.split(' | ')
                if len(parts) >= 2:
                    repo_id = parts[0]
                    short_hash = parts[1].replace("...", "")
                    
                    # Find full revision hash in cache_info
                    for repo in list(cache_info_state.repos):
                        if repo.repo_id == repo_id:
                            for revision in repo.revisions:
                                if revision.commit_hash.startswith(short_hash):
                                    selected_revisions.append(revision.commit_hash)
                                    break
            except Exception as e:
                print(f"Error parsing checkbox item: {e}")
                continue
        
        if not selected_revisions:
            return gr.update(), "**Status:** ⚠️ No valid items found for deletion", cache_info_state
        
        # Create deletion strategy
        delete_strategy = cache_info_state.delete_revisions(*selected_revisions)
        freed_size = delete_strategy.expected_freed_size_str
        
        # Execute deletion
        delete_strategy.execute()
        
        # Refresh cache after deletion
        new_checkbox_choices, status_msg, new_cache_info = scan_hf_cache()
        
        status = f"**Status:** ✅ Deleted {len(selected_revisions)} items. Freed: {freed_size}"
        
        return gr.update(choices=new_checkbox_choices, value=[]), status, new_cache_info
        
    except Exception as e:
        return gr.update(), f"**Status:** ❌ Error deleting cache: {str(e)}", cache_info_state


def refresh_hf_cache_for_gradio():
    """
    Wrapper function for Gradio to properly update checkbox choices.
    
    Returns:
        tuple: (gr.update_object, status_message, cache_info)
    """
    checkbox_choices, status, cache_info = scan_hf_cache()
    return gr.update(choices=checkbox_choices, value=[]), status, cache_info