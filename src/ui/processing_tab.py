"""
Processing Tab UI Components

This module creates the UI components for the processing queue tab.
Provides interface for managing the generation queue, processing tasks, and viewing results.

Features:
- Queue management interface
- Progress tracking with memory monitoring
- Results display with detailed information
- Real-time status updates

Author: FluxForge Team
License: MIT
"""

import gradio as gr
import threading
import time
import traceback
from typing import List, Dict, Any, Tuple
from pathlib import Path

from core.processing_queue import processing_queue, TaskStatus
from utils.progress_tracker import global_progress_tracker, format_progress_info

def format_elapsed_time(elapsed_time: float) -> str:
    """Format elapsed time in a human-readable format."""
    if elapsed_time < 60:
        return f"{elapsed_time:.0f}s"
    elif elapsed_time < 3600:
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        return f"{minutes}m {seconds}s"
    else:
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        return f"{hours}h {minutes}m"

def calculate_time_remaining(current_step: int, total_steps: int, elapsed_time: float) -> str:
    """Calculate estimated time remaining based on current progress."""
    if current_step <= 0 or total_steps <= 0 or elapsed_time <= 0:
        return "Calculating..."
    
    # Calculate average time per step
    avg_time_per_step = elapsed_time / current_step
    
    # Calculate remaining steps
    remaining_steps = total_steps - current_step
    
    # Calculate estimated time remaining
    estimated_remaining = avg_time_per_step * remaining_steps
    
    # Format the time
    if estimated_remaining < 60:
        return f"{estimated_remaining:.0f}s"
    elif estimated_remaining < 3600:
        minutes = int(estimated_remaining // 60)
        seconds = int(estimated_remaining % 60)
        return f"{minutes}m {seconds}s"
    else:
        hours = int(estimated_remaining // 3600)
        minutes = int((estimated_remaining % 3600) // 60)
        return f"{hours}h {minutes}m"

def create_progress_bar(current: int, total: int, percentage: float, task_name: str = "Generation") -> str:
    """Create a beautiful HTML progress bar."""
    # Ensure percentage is between 0 and 100
    percentage = max(0, min(100, percentage))
    
    # Color based on progress
    if percentage < 30:
        bar_color = "#FF9500"  # Orange
    elif percentage < 70:
        bar_color = "#007AFF"  # Blue
    else:
        bar_color = "#34C759"  # Green
    
    return f"""
    <div style="margin: 15px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <div style="font-weight: bold; color: #000 !important; font-size: 14px;">{task_name}</div>
            <div style="color: #000 !important; font-size: 14px;">{current}/{total} ({percentage:.1f}%)</div>
        </div>
        <div style="width: 100%; height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);">
            <div style="width: {percentage}%; height: 100%; background: linear-gradient(90deg, {bar_color} 0%, {bar_color}88 100%); border-radius: 10px; transition: width 0.3s ease; position: relative;">
                <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.2) 50%, transparent 100%); animation: shimmer 2s infinite;"></div>
            </div>
        </div>
    </div>
    <style>
        @keyframes shimmer {{
            0% {{ transform: translateX(-100%); }}
            100% {{ transform: translateX(100%); }}
        }}
    </style>
    """

def create_processing_tab():
    """Create the Processing tab interface."""
    
    gr.Markdown("## 🔄 Processing Queue")
    gr.Markdown("**Manage your generation queue** - All generation tasks are queued here for batch processing.")
    
    # ==============================================================================
    # QUEUE STATUS SECTION
    # ==============================================================================
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Queue Status")
            
            # Status display
            status_display = gr.Markdown(
                value="**Status:** Ready to process tasks"
            )
            
            # Queue statistics in HTML format
            queue_stats_html = gr.HTML(
                value="""
                <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 12px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="display: flex; justify-content: space-around; text-align: center;">
                        <div style="flex: 1; padding: 10px;">
                            <div style="font-size: 24px; font-weight: bold; color: #FF9500;">0</div>
                            <div style="font-size: 14px; color: #000 !important; margin-top: 5px;">⏳ Pending</div>
                        </div>
                        <div style="flex: 1; padding: 10px; border-left: 1px solid #ddd;">
                            <div style="font-size: 24px; font-weight: bold; color: #007AFF;">0</div>
                            <div style="font-size: 14px; color: #000 !important; margin-top: 5px;">⚙️ Processing</div>
                        </div>
                        <div style="flex: 1; padding: 10px; border-left: 1px solid #ddd;">
                            <div style="font-size: 24px; font-weight: bold; color: #34C759;">0</div>
                            <div style="font-size: 14px; color: #000 !important; margin-top: 5px;">✅ Completed</div>
                        </div>
                        <div style="flex: 1; padding: 10px; border-left: 1px solid #ddd;">
                            <div style="font-size: 24px; font-weight: bold; color: #FF3B30;">0</div>
                            <div style="font-size: 14px; color: #000 !important; margin-top: 5px;">❌ Errors</div>
                        </div>
                    </div>
                </div>
                """
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 🎮 Controls")
            
            # Main control buttons
            process_btn = gr.Button(
                "🚀 Process Queue",
                variant="primary",
                size="lg"
            )
            
            stop_btn = gr.Button(
                "⏹️ Stop Processing",
                variant="stop",
                size="lg"
            )
            
            refresh_btn = gr.Button(
                "🔄 Refresh",
                variant="secondary",
                size="lg"
            )
            
            
    
    # ==============================================================================
    # CURRENT PROCESSING SECTION
    # ==============================================================================
    with gr.Group():
        
        current_task_display = gr.HTML(
            value="""
            <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 12px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="text-align: center; color: #000 !important; font-size: 16px; font-style: italic;">
                    No task currently processing
                </div>
            </div>
            """
        )
        
        memory_display = gr.HTML(
            value="""
            <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 12px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="text-align: center; color: #000 !important; font-size: 14px; font-style: italic;">
                    Memory statistics will appear here during processing
                </div>
            </div>
            """
        )
    
    # ==============================================================================
    # QUEUE DISPLAY SECTION
    # ==============================================================================
    gr.Markdown("### 📋 Pending Tasks")
    
    # Dataframe for pending tasks - full width
    pending_tasks_dataframe = gr.Dataframe(
        value=[],
        headers=["Select", "ID", "Type", "Description"],
        datatype=["bool", "str", "str", "str"],
        label="Pending Tasks - Check boxes to select for removal",
        interactive=True,
        row_count=(0, "dynamic"),
        col_count=4,
        wrap=True,
        visible=False  # Hidden by default when empty
    )
    
    # Bulk task removal
    with gr.Row():
        remove_selected_btn = gr.Button(
            "🗑️ Remove Selected",
            variant="secondary",
            scale=2
        )
        select_all_btn = gr.Button(
            "☑️ Select All",
            variant="secondary",
            scale=1
        )
        clear_selection_btn = gr.Button(
            "☐ Clear Selection",
            variant="secondary",
            scale=1
        )
    
    
    # Hidden state for tracking
    processing_state = gr.State(False)
    last_update = gr.State(0)
    
    # Return all components for event setup
    return {
        'status_display': status_display,
        'queue_stats_html': queue_stats_html,
        'process_btn': process_btn,
        'stop_btn': stop_btn,
        'refresh_btn': refresh_btn,
        'current_task_display': current_task_display,
        'memory_display': memory_display,
        'pending_tasks_dataframe': pending_tasks_dataframe,
        'remove_selected_btn': remove_selected_btn,
        'select_all_btn': select_all_btn,
        'clear_selection_btn': clear_selection_btn,
        'processing_state': processing_state,
        'last_update': last_update
    }

def update_queue_status() -> Tuple[str, str, List[List], bool]:
    """Update queue status display and return current state."""
    try:
        stats = processing_queue.get_stats()
        queue_summary = processing_queue.get_queue_summary()
        
        
        # Format status message
        if stats['is_processing']:
            current_task = stats.get('current_task')
            if current_task:
                status_msg = f"**🔄 Processing:** {current_task['description']}"
            else:
                status_msg = "**🔄 Processing queue...**"
        else:
            status_msg = "**✅ Ready** - Queue idle, ready to process tasks"
        
        # Format for dataframe (with selection checkbox)
        dataframe_rows = []
        
        if queue_summary:
            for i, task in enumerate(queue_summary):
                # Dataframe row: [Select, ID, Type, Description]
                dataframe_rows.append([
                    False,  # Checkbox not selected by default
                    task['id'][:8],
                    task['type'].replace('_', ' ').title(),
                    task['description'][:60] + "..." if len(task['description']) > 60 else task['description']
                ])
        
        # Create HTML stats display
        stats_html = f"""
        <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 12px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-around; text-align: center;">
                <div style="flex: 1; padding: 10px;">
                    <div style="font-size: 24px; font-weight: bold; color: #FF9500;">{stats['pending']}</div>
                    <div style="font-size: 14px; color: #000 !important; margin-top: 5px;">⏳ Pending</div>
                </div>
                <div style="flex: 1; padding: 10px; border-left: 1px solid #ddd;">
                    <div style="font-size: 24px; font-weight: bold; color: #007AFF;">{stats['processing']}</div>
                    <div style="font-size: 14px; color: #000 !important; margin-top: 5px;">⚙️ Processing</div>
                </div>
                <div style="flex: 1; padding: 10px; border-left: 1px solid #ddd;">
                    <div style="font-size: 24px; font-weight: bold; color: #34C759;">{stats['completed']}</div>
                    <div style="font-size: 14px; color: #000 !important; margin-top: 5px;">✅ Completed</div>
                </div>
                <div style="flex: 1; padding: 10px; border-left: 1px solid #ddd;">
                    <div style="font-size: 24px; font-weight: bold; color: #FF3B30;">{stats['errors']}</div>
                    <div style="font-size: 14px; color: #000 !important; margin-top: 5px;">❌ Errors</div>
                </div>
            </div>
        </div>
        """
        
        # Determine visibility based on whether there are tasks
        dataframe_visible = len(dataframe_rows) > 0
        
        return (
            status_msg,
            stats_html,
            dataframe_rows,
            dataframe_visible
        )
        
    except Exception as e:
        error_stats_html = """
        <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 12px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-around; text-align: center;">
                <div style="flex: 1; padding: 10px;">
                    <div style="font-size: 24px; font-weight: bold; color: #FF9500;">0</div>
                    <div style="font-size: 14px; color: #000 !important; margin-top: 5px;">⏳ Pending</div>
                </div>
                <div style="flex: 1; padding: 10px; border-left: 1px solid #ddd;">
                    <div style="font-size: 24px; font-weight: bold; color: #007AFF;">0</div>
                    <div style="font-size: 14px; color: #000 !important; margin-top: 5px;">⚙️ Processing</div>
                </div>
                <div style="flex: 1; padding: 10px; border-left: 1px solid #ddd;">
                    <div style="font-size: 24px; font-weight: bold; color: #34C759;">0</div>
                    <div style="font-size: 14px; color: #000 !important; margin-top: 5px;">✅ Completed</div>
                </div>
                <div style="flex: 1; padding: 10px; border-left: 1px solid #ddd;">
                    <div style="font-size: 24px; font-weight: bold; color: #FF3B30;">0</div>
                    <div style="font-size: 14px; color: #000 !important; margin-top: 5px;">❌ Errors</div>
                </div>
            </div>
        </div>
        """
        return (
            f"**❌ Error:** {str(e)}",
            error_stats_html,
            [],
            False
        )

def update_current_task() -> Tuple[str, str]:
    """Update current task display and memory display with real-time progress."""
    try:
        current_task = processing_queue.get_processing_summary()
        
        if current_task:
            # Get real-time progress information
            progress_info = global_progress_tracker.get_current_progress()
            
            # Check if progress info is from current task (not stale from previous task)
            if progress_info and progress_info.get('elapsed_time', 0) > 0:
                # Extract progress data
                current_step = progress_info.get('current_step', 0)
                total_steps = progress_info.get('total_steps', 0)
                percentage = progress_info.get('percentage', 0)
                elapsed_time = progress_info.get('elapsed_time', 0)
                task_type = progress_info.get('task_type', 'unknown')
                
                # Determine task name based on type
                if task_type == 'generation':
                    task_name = "🎨 Generation"
                elif task_type == 'loading':
                    task_name = "🔄 Loading"
                else:
                    task_name = "⚙️ Processing"
                
                # Calculate time remaining and format elapsed time
                time_remaining = calculate_time_remaining(current_step, total_steps, elapsed_time)
                formatted_elapsed = format_elapsed_time(elapsed_time)
                
                # Create progress bar
                progress_bar = create_progress_bar(current_step, total_steps, percentage, task_name)
                
                # Format complete display
                description = f"""
                <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 12px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="font-size: 16px; font-weight: bold; color: #000 !important; margin-bottom: 10px;">
                        🎯 Currently Processing
                    </div>
                    <div style="font-size: 14px; color: #000 !important; margin-bottom: 15px;">
                        {current_task['description']}
                    </div>
                    {progress_bar}
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;">
                        <div style="color: #000 !important; font-size: 13px;">
                            ⏱️ <span style="color: #000 !important; font-weight: bold;">Elapsed:</span> {formatted_elapsed}
                        </div>
                        <div style="color: #000 !important; font-size: 13px;">
                            ⏰ <span style="color: #000 !important; font-weight: bold;">Remaining:</span> {time_remaining}
                        </div>
                        <div style="color: #000 !important; font-size: 13px;">
                            📊 <span style="color: #000 !important; font-weight: bold;">Step:</span> {current_step}/{total_steps}
                        </div>
                    </div>
                </div>
                """
            else:
                # No progress yet or stale progress, show just the task description
                description = f"""
                <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 12px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="font-size: 16px; font-weight: bold; color: #000 !important; margin-bottom: 10px;">
                        🎯 Currently Processing
                    </div>
                    <div style="font-size: 14px; color: #000 !important; margin-bottom: 15px;">
                        {current_task['description']}
                    </div>
                    <div style="text-align: center; padding: 20px; color: #000 !important; font-style: italic;">
                        ⏳ Initializing...
                    </div>
                </div>
                """
            
            # Format memory stats if available
            memory_stats = current_task.get('memory_stats', {})
            
            if 'before' in memory_stats:
                from core.processing_queue import MemoryMonitor
                memory_before = MemoryMonitor.format_memory_stats(memory_stats['before'])
                
                memory_during = ""
                if 'during' in memory_stats:
                    memory_during = f"""
                    <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;">
                        <div style="font-weight: bold; color: #000 !important; margin-bottom: 8px;">📊 During Processing:</div>
                        <div style="color: #000 !important; font-size: 13px;">{MemoryMonitor.format_memory_stats(memory_stats['during'])}</div>
                    </div>
                    """
                
                memory_text = f"""
                <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 12px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="font-weight: bold; color: #000 !important; margin-bottom: 8px;">💾 Memory Statistics:</div>
                    <div style="color: #000 !important; font-size: 13px;">{memory_before}</div>
                    {memory_during}
                </div>
                """
            else:
                memory_text = """
                <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 12px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="text-align: center; color: #000 !important; font-size: 14px;">
                        <span style="color: #000 !important; font-style: italic;">Memory monitoring in progress...</span>
                    </div>
                </div>
                """
        else:
            description = """
            <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 12px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="text-align: center; color: #000 !important; font-size: 16px; font-style: italic;">
                    No task currently processing
                </div>
            </div>
            """
            memory_text = """
            <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 12px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="text-align: center; color: #000 !important; font-size: 14px; font-style: italic;">
                    Memory statistics will appear here during processing
                </div>
            </div>
            """
        
        return description, memory_text
        
    except Exception as e:
        error_description = f"""
        <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 12px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="text-align: center; color: #FF3B30; font-size: 16px;">
                <em>Error getting task progress: {e}</em>
            </div>
        </div>
        """
        error_memory = """
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 8px; padding: 15px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="text-align: center; color: #FF3B30; font-size: 14px;">
                <em>Error getting memory stats</em>
            </div>
        </div>
        """
        return error_description, error_memory

def process_queue_async(image_generator, modelbgrm) -> str:
    """Start queue processing in a separate thread."""
    def _process():
        processing_queue.process_queue(image_generator, modelbgrm)
    
    if processing_queue.is_processing:
        return "⚠️ Queue is already being processed"
    
    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    
    return "🚀 Queue processing started"

def stop_queue_processing() -> str:
    """Stop queue processing."""
    processing_queue.stop_processing()
    return "⏹️ Stop signal sent to queue processor"

def clear_pending_queue() -> str:
    """Clear all pending tasks from queue."""
    count = processing_queue.clear_queue()
    return f"🧹 Cleared {count} pending tasks from queue"

def clear_completed_tasks() -> str:
    """Clear all completed tasks."""
    count = processing_queue.clear_completed()
    return f"🗑️ Cleared {count} completed tasks"

def remove_selected_tasks(selected_tasks: List[str]) -> str:
    """Remove selected tasks from queue."""
    if not selected_tasks:
        return "⚠️ No tasks selected for removal"
    
    queue_summary = processing_queue.get_queue_summary()
    removed_count = 0
    
    for task_label in selected_tasks:
        # Extract task ID from label format: "ID (8 chars) - Type - Description"
        task_id = task_label.split(' - ')[0]
        
        # Find the matching task
        matching_tasks = [task for task in queue_summary if task['id'].startswith(task_id)]
        
        if matching_tasks:
            full_id = matching_tasks[0]['id']
            success = processing_queue.remove_task(full_id)
            if success:
                removed_count += 1
    
    if removed_count > 0:
        return f"✅ Removed {removed_count} task(s) from queue"
    else:
        return "❌ No tasks could be removed (may be currently processing)"

def select_all_tasks() -> List[str]:
    """Select all pending tasks."""
    try:
        queue_summary = processing_queue.get_queue_summary()
        all_choices = []
        
        for task in queue_summary:
            task_label = f"{task['id'][:8]} - {task['type'].replace('_', ' ').title()} - {task['description'][:50]}"
            if len(task['description']) > 50:
                task_label += "..."
            all_choices.append(task_label)
        
        return all_choices
    except Exception as e:
        return []

def clear_task_selection() -> List[str]:
    """Clear task selection."""
    return []


def get_task_details(evt: gr.SelectData) -> str:
    """Get detailed information for a selected task."""
    try:
        if evt.index is None:
            return "No task selected."
        
        # Get task from completed list (assuming click was on completed dataframe)
        completed_summary = processing_queue.get_completed_summary()
        if evt.index >= len(completed_summary):
            return "Task not found."
        
        task = completed_summary[evt.index]
        
        details = []
        details.append(f"**Task ID:** {task['id']}")
        details.append(f"**Type:** {task['type'].replace('_', ' ').title()}")
        details.append(f"**Status:** {task['status'].title()}")
        details.append(f"**Created:** {task['created_at']}")
        details.append(f"**Description:** {task['description']}")
        
        
        if task.get('error_message'):
            details.append(f"**Error:** {task['error_message']}")
        
        # Memory statistics
        memory_stats = task.get('memory_stats', {})
        if memory_stats:
            details.append("\\n**Memory Statistics:**")
            from core.processing_queue import MemoryMonitor
            
            if 'before' in memory_stats:
                details.append(f"Before: {MemoryMonitor.format_memory_stats(memory_stats['before'])}")
            if 'during' in memory_stats:
                details.append(f"During: {MemoryMonitor.format_memory_stats(memory_stats['during'])}")
            if 'after' in memory_stats:
                details.append(f"After: {MemoryMonitor.format_memory_stats(memory_stats['after'])}")
        
        # Parameters (abbreviated)
        details.append("\\n**Parameters:**")
        params = task.get('parameters', {})
        for key, value in params.items():
            if key in ['prompt', 'model_alias', 'steps', 'guidance', 'quantization']:
                details.append(f"{key}: {value}")
        
        return "\\n".join(details)
        
    except Exception as e:
        return f"Error getting task details: {e}"


def setup_processing_tab_events(components: Dict[str, Any], image_generator, modelbgrm):
    """Set up event handlers for the processing tab."""
    
    # Main control buttons
    components['process_btn'].click(
        fn=lambda: process_queue_async(image_generator, modelbgrm),
        outputs=components['status_display'],
        show_progress=True
    )
    
    components['stop_btn'].click(
        fn=stop_queue_processing,
        outputs=components['status_display']
    )
    
    # Refresh all data
    def refresh_all():
        (status_msg, stats_html, dataframe_rows, dataframe_visible) = update_queue_status()
        
        current_desc, memory_text = update_current_task()
        
        # Use gr.update to control visibility
        dataframe_update = gr.update(value=dataframe_rows, visible=dataframe_visible)
        
        return (
            status_msg, stats_html, dataframe_update, current_desc, memory_text
        )
    
    components['refresh_btn'].click(
        fn=refresh_all,
        outputs=[
            components['status_display'],
            components['queue_stats_html'],
            components['pending_tasks_dataframe'],
            components['current_task_display'],
            components['memory_display']
        ]
    )
    
    
    # Task selection and removal with auto-refresh
    def remove_and_refresh(dataframe_data):
        # Extract selected tasks from dataframe
        selected_task_ids = []
        result_msg = "No tasks selected for removal"
        
        try:
            # Handle dataframe data properly
            if dataframe_data is not None:
                # Convert to list if it's a DataFrame
                if hasattr(dataframe_data, 'values'):
                    rows = dataframe_data.values.tolist()
                else:
                    rows = dataframe_data
                
                for i, row in enumerate(rows):
                    # Handle both string 'true' and boolean True
                    checkbox_value = row[0]
                    is_selected = checkbox_value is True or checkbox_value == 'true' or checkbox_value == True
                    
                    if len(row) >= 2 and is_selected:  # If checkbox is True
                        selected_task_ids.append(row[1])  # ID is in column 1 (8 chars)
            
            if selected_task_ids:
                # Remove tasks by ID - need to find full ID from partial ID
                removed_count = 0
                for short_id in selected_task_ids:
                    # Find the full task ID from the queue
                    queue_summary = processing_queue.get_queue_summary()
                    for task in queue_summary:
                        if task['id'].startswith(short_id):
                            success = processing_queue.remove_task(task['id'])
                            if success:
                                removed_count += 1
                            break
                result_msg = f"✅ Removed {removed_count} task(s) from queue"
            else:
                result_msg = "⚠️ No tasks selected for removal"
                
        except Exception as e:
            result_msg = f"❌ Error removing tasks: {str(e)}"
        
        (status_msg, stats_html, dataframe_rows, dataframe_visible) = update_queue_status()
        
        # Use gr.update to control visibility
        dataframe_update = gr.update(value=dataframe_rows, visible=dataframe_visible)
        
        return result_msg, dataframe_update, stats_html
    
    components['remove_selected_btn'].click(
        fn=remove_and_refresh,
        inputs=components['pending_tasks_dataframe'],
        outputs=[
            components['status_display'],
            components['pending_tasks_dataframe'],
            components['queue_stats_html']
        ]
    )
    
    # Simplified select/clear functions for dataframe
    def select_all_dataframe():
        (status_msg, stats_html, dataframe_rows, dataframe_visible) = update_queue_status()
        # Set all checkboxes to True
        for row in dataframe_rows:
            if len(row) > 0:
                row[0] = True
        return dataframe_rows
    
    def clear_dataframe_selection():
        (status_msg, stats_html, dataframe_rows, dataframe_visible) = update_queue_status()
        # Set all checkboxes to False
        for row in dataframe_rows:
            if len(row) > 0:
                row[0] = False
        return dataframe_rows
    
    components['select_all_btn'].click(
        fn=select_all_dataframe,
        outputs=components['pending_tasks_dataframe']
    )
    
    components['clear_selection_btn'].click(
        fn=clear_dataframe_selection,
        outputs=components['pending_tasks_dataframe']
    )
    
    
