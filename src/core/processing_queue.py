"""
Processing Queue Manager

This module manages a queue system for batch image generation and processing.
It allows users to queue multiple generation tasks and process them sequentially
with memory monitoring.

Features:
- Queue management for different types of generation tasks
- Memory monitoring during processing
- Sequential processing with progress tracking
- Integration with existing generation modules

Author: FluxForge Team
License: MIT
"""

import json
import uuid
import threading
import time
import psutil
import torch
import gc
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from enum import Enum
from utils.progress_tracker import global_progress_tracker

class TaskType(Enum):
    """Enumeration of supported task types."""
    STANDARD_GENERATION = "standard_generation"
    FLUX_FILL = "flux_fill"
    KONTEXT = "kontext"
    FLUX_CANNY = "flux_canny"
    FLUX_DEPTH = "flux_depth"
    FLUX_REDUX = "flux_redux"
    BACKGROUND_REMOVAL = "background_removal"
    UPSCALING = "upscaling"

class TaskStatus(Enum):
    """Enumeration of task statuses."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

class ProcessingTask:
    """Represents a single task in the processing queue."""
    
    def __init__(self, task_type: TaskType, parameters: Dict[str, Any], description: str = ""):
        self.id = str(uuid.uuid4())
        self.type = task_type
        self.status = TaskStatus.PENDING
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.parameters = parameters
        self.description = description or self._generate_description()
        self.progress = 0
        self.error_message = ""
        self.memory_stats = {}
        
    def _generate_description(self) -> str:
        """Generate a human-readable description for the task."""
        if self.type == TaskType.STANDARD_GENERATION:
            prompt = self.parameters.get('prompt', 'Unknown prompt')[:50]
            model = self.parameters.get('model_alias', 'Unknown model')
            return f"Generate: {prompt}... (Model: {model})"
        
        elif self.type == TaskType.FLUX_FILL:
            mode = self.parameters.get('fill_mode', 'Unknown mode')
            prompt = self.parameters.get('prompt', 'Unknown prompt')[:30]
            return f"FLUX Fill ({mode}): {prompt}..."
        
        elif self.type == TaskType.KONTEXT:
            prompt = self.parameters.get('prompt', 'Unknown prompt')[:40]
            return f"Kontext Edit: {prompt}..."
        
        elif self.type == TaskType.FLUX_CANNY:
            prompt = self.parameters.get('prompt', 'Unknown prompt')[:40]
            return f"FLUX Canny: {prompt}..."
        
        elif self.type == TaskType.FLUX_DEPTH:
            prompt = self.parameters.get('prompt', 'Unknown prompt')[:40]
            return f"FLUX Depth: {prompt}..."
        
        elif self.type == TaskType.FLUX_REDUX:
            return f"FLUX Redux: Image variation"
        
        elif self.type == TaskType.BACKGROUND_REMOVAL:
            return f"Background Removal"
        
        elif self.type == TaskType.UPSCALING:
            factor = self.parameters.get('upscale_factor', 2.0)
            quantization = self.parameters.get('quantization', 'None')
            if quantization != 'None':
                return f"Upscaling (x{factor}, {quantization})"
            else:
                return f"Upscaling (x{factor})"
        
        return f"{self.type.value.replace('_', ' ').title()} Task"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "description": self.description,
            "parameters": self.parameters,
            "progress": self.progress,
            "error_message": self.error_message,
            "memory_stats": self.memory_stats
        }

class MemoryMonitor:
    """Monitor memory usage during processing."""
    
    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """Get current memory statistics."""
        stats = {}
        
        # System memory
        memory = psutil.virtual_memory()
        stats['system_total_gb'] = memory.total / (1024**3)
        stats['system_used_gb'] = memory.used / (1024**3)
        stats['system_available_gb'] = memory.available / (1024**3)
        stats['system_percent'] = memory.percent
        
        # GPU memory if available
        if torch.cuda.is_available():
            stats['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
            stats['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
            stats['gpu_max_allocated_gb'] = torch.cuda.max_memory_allocated() / (1024**3)
        elif torch.backends.mps.is_available():
            # MPS doesn't have detailed memory tracking like CUDA
            stats['mps_available'] = True
        
        return stats
    
    @staticmethod
    def format_memory_stats(stats: Dict[str, float]) -> str:
        """Format memory statistics for display."""
        lines = []
        lines.append(f"System: {stats.get('system_used_gb', 0):.1f}/{stats.get('system_total_gb', 0):.1f} GB ({stats.get('system_percent', 0):.1f}%)")
        
        if 'gpu_allocated_gb' in stats:
            lines.append(f"GPU: {stats.get('gpu_allocated_gb', 0):.1f} GB allocated, {stats.get('gpu_reserved_gb', 0):.1f} GB reserved")
        elif stats.get('mps_available'):
            lines.append("MPS: Available (detailed stats not supported)")
        
        return "\n".join(lines)

class ProcessingQueue:
    """Manages the processing queue for image generation tasks."""
    
    def __init__(self):
        self.tasks: List[ProcessingTask] = []
        self.completed_tasks: List[ProcessingTask] = []
        self.is_processing = False
        self.current_task: Optional[ProcessingTask] = None
        self._lock = threading.Lock()
        self._stop_processing = False
        self.memory_monitor = MemoryMonitor()
        
        # Progress callback for UI updates
        self.progress_callback: Optional[Callable] = None
        
    def add_task(self, task_type: TaskType, parameters: Dict[str, Any], description: str = "") -> str:
        """Add a new task to the queue."""
        with self._lock:
            task = ProcessingTask(task_type, parameters, description)
            self.tasks.append(task)
            print(f"✅ Added to queue: {task.description}")
            return task.id
    
    def get_queue_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all tasks in queue."""
        with self._lock:
            return [task.to_dict() for task in self.tasks if task.status == TaskStatus.PENDING]
    
    def get_completed_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all completed tasks."""
        with self._lock:
            return [task.to_dict() for task in self.completed_tasks]
    
    def get_processing_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of currently processing task with real-time progress."""
        with self._lock:
            if self.current_task:
                task_dict = self.current_task.to_dict()
                # Add real-time progress information
                progress_info = global_progress_tracker.get_current_progress()
                if progress_info:
                    task_dict['progress'] = progress_info
                return task_dict
            return None
    
    def remove_task(self, task_id: str) -> bool:
        """Remove a pending task from the queue."""
        with self._lock:
            for i, task in enumerate(self.tasks):
                if task.id == task_id and task.status == TaskStatus.PENDING:
                    removed_task = self.tasks.pop(i)
                    print(f"🗑️ Removed from queue: {removed_task.description}")
                    return True
            return False
    
    def clear_queue(self) -> int:
        """Clear all pending tasks from queue."""
        with self._lock:
            if self.is_processing:
                return 0  # Cannot clear while processing
            
            pending_count = len([task for task in self.tasks if task.status == TaskStatus.PENDING])
            self.tasks = [task for task in self.tasks if task.status != TaskStatus.PENDING]
            print(f"🧹 Cleared {pending_count} pending tasks from queue")
            return pending_count
    
    def clear_completed(self) -> int:
        """Clear all completed tasks."""
        with self._lock:
            count = len(self.completed_tasks)
            self.completed_tasks.clear()
            print(f"🧹 Cleared {count} completed tasks")
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall queue statistics."""
        with self._lock:
            pending = len([task for task in self.tasks if task.status == TaskStatus.PENDING])
            processing = 1 if self.current_task else 0
            completed = len(self.completed_tasks)
            errors = len([task for task in self.completed_tasks if task.status == TaskStatus.ERROR])
            
            return {
                "pending": pending,
                "processing": processing,
                "completed": completed,
                "errors": errors,
                "is_processing": self.is_processing,
                "current_task": self.current_task.to_dict() if self.current_task else None
            }
    
    def set_progress_callback(self, callback: Callable):
        """Set callback function for progress updates."""
        self.progress_callback = callback
    
    def _update_progress(self, task: ProcessingTask, progress: int, status: TaskStatus = None):
        """Update task progress and notify UI."""
        task.progress = progress
        if status:
            task.status = status
        
        if self.progress_callback:
            try:
                self.progress_callback(task.to_dict())
            except Exception as e:
                print(f"⚠️ Progress callback error: {e}")
    
    def process_queue(self, image_generator, modelbgrm) -> bool:
        """Process all tasks in the queue sequentially."""
        if self.is_processing:
            print("⚠️ Queue is already being processed")
            return False
        
        with self._lock:
            pending_tasks = [task for task in self.tasks if task.status == TaskStatus.PENDING]
            if not pending_tasks:
                print("📭 No pending tasks in queue")
                return False
        
        self.is_processing = True
        self._stop_processing = False
        
        try:
            print(f"🚀 Starting queue processing with {len(pending_tasks)} tasks")
            
            for task in pending_tasks:
                if self._stop_processing:
                    print("⏹️ Queue processing stopped by user")
                    break
                
                self.current_task = task
                self._process_single_task(task, image_generator, modelbgrm)
                
                # Move completed task to completed list
                with self._lock:
                    if task in self.tasks:
                        self.tasks.remove(task)
                    self.completed_tasks.append(task)
                
                self.current_task = None
                
                # Force garbage collection between tasks
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                
            print("✅ Queue processing completed")
            return True
            
        except Exception as e:
            print(f"❌ Queue processing error: {e}")
            return False
        
        finally:
            self.is_processing = False
            self.current_task = None
    
    def _process_single_task(self, task: ProcessingTask, image_generator, modelbgrm):
        """Process a single task with memory monitoring."""
        try:
            print(f"🎯 Processing: {task.description}")
            
            # Clear progress tracker from previous task to avoid displaying old progress
            from utils.progress_tracker import global_progress_tracker
            global_progress_tracker.reset()
            
            # Memory stats before processing
            memory_before = self.memory_monitor.get_memory_stats()
            task.memory_stats['before'] = memory_before
            print(f"📊 Memory before: {self.memory_monitor.format_memory_stats(memory_before)}")
            
            # Update status to processing
            self._update_progress(task, 0, TaskStatus.PROCESSING)
            
            # Process the task based on type
            self._update_progress(task, 10)
            result = self._execute_task(task, image_generator, modelbgrm)
            self._update_progress(task, 90)
            
            # Memory stats during processing (peak)
            memory_during = self.memory_monitor.get_memory_stats()
            task.memory_stats['during'] = memory_during
            print(f"📊 Memory during: {self.memory_monitor.format_memory_stats(memory_during)}")
            
            # Store result - image_generator already handled saving
            if result:
                task.status = TaskStatus.COMPLETED
                print(f"✅ Completed: {task.description}")
                # Note: image_generator.generate_image() already saved the image and updated the database
            else:
                raise Exception("Task returned no result")
            
            # Memory stats after processing
            memory_after = self.memory_monitor.get_memory_stats()
            task.memory_stats['after'] = memory_after
            print(f"📊 Memory after: {self.memory_monitor.format_memory_stats(memory_after)}")
            
            self._update_progress(task, 100, TaskStatus.COMPLETED)
            
        except Exception as e:
            task.status = TaskStatus.ERROR
            task.error_message = str(e)
            task.progress = 0
            print(f"❌ Error processing {task.description}: {e}")
            
            if self.progress_callback:
                self.progress_callback(task.to_dict())
    
    def _execute_task(self, task: ProcessingTask, image_generator, modelbgrm):
        """Execute the actual generation task based on its type."""
        params = task.parameters
        
        if task.type == TaskType.STANDARD_GENERATION:
            # Execute standard generation using image_generator directly
            return self._execute_standard_generation(params, image_generator)
        
        elif task.type == TaskType.FLUX_FILL:
            from postprocessing.flux_fill import process_flux_fill
            return process_flux_fill(
                params['fill_mode'], params['image_editor_data'], params['outpaint_image'],
                params['prompt'], params['steps'], params['guidance_scale'], params['quantization'],
                params['top_percent'], params['bottom_percent'], params['left_percent'], params['right_percent'],
                params['lora_state'], params['lora_strength_1'], params['lora_strength_2'], params['lora_strength_3'],
                image_generator
            )
        
        elif task.type == TaskType.KONTEXT:
            from postprocessing.kontext import process_kontext
            return process_kontext(
                params['input_image'], params['prompt'], params['steps'], params['guidance_scale'],
                params['quantization'], params['lora_state'], params['lora_strength_1'],
                params['lora_strength_2'], params['lora_strength_3'], image_generator
            )
        
        elif task.type == TaskType.FLUX_CANNY:
            from postprocessing.flux_canny import process_flux_canny
            return process_flux_canny(
                params['input_image'], params['prompt'], params['steps'], params['guidance_scale'],
                params['quantization'], params['low_threshold'], params['high_threshold'],
                params['lora_state'], params['lora_strength_1'], params['lora_strength_2'], params['lora_strength_3'],
                image_generator
            )
        
        elif task.type == TaskType.FLUX_DEPTH:
            from postprocessing.flux_depth import process_flux_depth
            return process_flux_depth(
                params['input_image'], params['prompt'], params['steps'], params['guidance_scale'],
                params['quantization'], params['lora_state'], params['lora_strength_1'],
                params['lora_strength_2'], params['lora_strength_3'], image_generator
            )
        
        elif task.type == TaskType.FLUX_REDUX:
            from postprocessing.flux_redux import process_flux_redux
            return process_flux_redux(
                params['input_image'], params['guidance_scale'], params['steps'],
                params['variation_strength'], params['quantization'], image_generator
            )
        
        elif task.type == TaskType.BACKGROUND_REMOVAL:
            from postprocessing.background_remover import remove_background
            return remove_background(params['input_image'], modelbgrm)
        
        elif task.type == TaskType.UPSCALING:
            from postprocessing.upscaler import upscale_image
            return upscale_image(
                params['input_image'], 
                params['upscale_factor'], 
                params.get('quantization', 'None')
            )
        
        else:
            raise ValueError(f"Unknown task type: {task.type}")
    
    def _execute_standard_generation(self, params: Dict[str, Any], image_generator):
        """Execute standard generation task using image_generator."""
        # Convert LoRA state to individual checkbox/scale format expected by ImageGenerator
        lora_checkboxes = []
        lora_scales = []
        
        # Initialize with False for all LoRA models
        for _ in image_generator.lora_data:
            lora_checkboxes.append(False)
            lora_scales.append(1.0)
        
        # Process selected LoRA from state
        lora_state = params.get('lora_state')
        if lora_state:
            strengths = [params.get('lora_strength_1', 0.8), 
                        params.get('lora_strength_2', 0.8), 
                        params.get('lora_strength_3', 0.8)]
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
            params['prompt'],                    # prompt
            params['model_alias'],              # model_alias
            params['steps'],                    # steps
            params['seed'],                     # seed
            params['metadata'],                 # metadata
            params['guidance'],                 # guidance
            params['height'],                   # height
            params['width'],                    # width
            "",                                 # path - empty for HuggingFace models
            "None",                             # controlnet_type
            None,                               # controlnet_image_path
            1.0,                                # controlnet_strength
            False,                              # controlnet_save_canny
            False,                              # enable_stepwise
            None,                               # progress - will be None in queue processing
            100,                                # canny_low_threshold
            200,                                # canny_high_threshold
            2.0,                                # upscaler_multiplier
            "None",                             # flux_tools_type
            None,                               # flux_tools_image_path
            2.5,                                # flux_tools_guidance
            "None",                             # post_processing_type
            None,                               # post_processing_image_path
            2.0,                                # post_processing_multiplier
            params['quantization'],             # quantization
            *lora_checkboxes,                   # LoRA selections
            *lora_scales                        # LoRA scales
        )
    
    
    def stop_processing(self):
        """Stop queue processing."""
        self._stop_processing = True
        print("⏹️ Stop signal sent to queue processor")

# Global processing queue instance
processing_queue = ProcessingQueue()