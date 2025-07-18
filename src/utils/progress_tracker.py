"""
Progress Tracker for FLUX Generation Steps
Captures real-time progress from diffusers tqdm and provides thread-safe updates
"""

import time
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

class TaskType(Enum):
    LOADING = "loading"
    GENERATION = "generation"
    UNKNOWN = "unknown"

@dataclass
class StepInfo:
    """Information about a single step"""
    step: int
    total_steps: int
    percentage: float
    elapsed_time: float
    step_duration: float
    description: str = ""
    task_type: TaskType = TaskType.UNKNOWN
    timestamp: float = field(default_factory=time.time)

class ProgressTracker:
    """Thread-safe progress tracker for FLUX generation"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._step_info: List[StepInfo] = []
        self._current_step = 0
        self._total_steps = 0
        self._start_time = None
        self._last_step_time = None
        self._generation_started = False
        self._callbacks: List[Callable] = []
        
        # Patch tqdm
        self._original_tqdm_init = None
        self._original_tqdm_update = None
        self._patches_applied = False
        
    def reset(self):
        """Reset tracker for new generation"""
        with self._lock:
            self._step_info = []
            self._current_step = 0
            self._total_steps = 0
            self._start_time = time.time()
            self._last_step_time = self._start_time
            self._generation_started = False
            
    def add_callback(self, callback: Callable):
        """Add callback to be called on step updates"""
        with self._lock:
            self._callbacks.append(callback)
            
    def remove_callback(self, callback: Callable):
        """Remove callback"""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
                
    def _notify_callbacks(self, step_info: StepInfo):
        """Notify all callbacks of step update"""
        for callback in self._callbacks:
            try:
                callback(step_info)
            except Exception as e:
                print(f"Error in progress callback: {e}")
                
    def _classify_task_type(self, desc: str, total: int) -> TaskType:
        """Classify the type of task based on description and total"""
        if not desc:
            if total <= 10:  # Likely generation steps
                return TaskType.GENERATION
            else:
                return TaskType.LOADING
                
        desc_lower = desc.lower()
        
        # Loading patterns
        loading_patterns = ['loading', 'checkpoint', 'component', 'model']
        if any(pattern in desc_lower for pattern in loading_patterns):
            return TaskType.LOADING
            
        # Generation patterns
        generation_patterns = ['step', 'it/s', 'diffusion', '%|']
        if any(pattern in desc_lower for pattern in generation_patterns):
            return TaskType.GENERATION
            
        # Default classification based on total
        if total <= 10:
            return TaskType.GENERATION
        else:
            return TaskType.LOADING
            
    def on_step_update(self, step: int, total: int, desc: str = ""):
        """Handle step update from tqdm"""
        with self._lock:
            current_time = time.time()
            
            # Calculate timings
            elapsed = current_time - self._start_time if self._start_time else 0
            step_duration = current_time - self._last_step_time if self._last_step_time else 0
            percentage = (step / total * 100) if total > 0 else 0
            
            # Classify task type
            task_type = self._classify_task_type(desc, total)
            
            # Create step info
            step_info = StepInfo(
                step=step,
                total_steps=total,
                percentage=percentage,
                elapsed_time=elapsed,
                step_duration=step_duration,
                description=desc,
                task_type=task_type,
                timestamp=current_time
            )
            
            # Update internal state
            self._step_info.append(step_info)
            self._current_step = step
            self._total_steps = total
            self._last_step_time = current_time
            
            # Mark generation as started for generation tasks
            if task_type == TaskType.GENERATION and not self._generation_started:
                self._generation_started = True
                
            # Notify callbacks
            self._notify_callbacks(step_info)
            
    def get_current_progress(self) -> Optional[Dict]:
        """Get current progress information"""
        with self._lock:
            if not self._step_info:
                return None
                
            latest_step = self._step_info[-1]
            
            # Find the latest generation step
            latest_generation_step = None
            for step in reversed(self._step_info):
                if step.task_type == TaskType.GENERATION:
                    latest_generation_step = step
                    break
                    
            return {
                'current_step': latest_step.step,
                'total_steps': latest_step.total_steps,
                'percentage': latest_step.percentage,
                'elapsed_time': latest_step.elapsed_time,
                'step_duration': latest_step.step_duration,
                'description': latest_step.description,
                'task_type': latest_step.task_type.value,
                'generation_started': self._generation_started,
                'generation_progress': {
                    'step': latest_generation_step.step if latest_generation_step else 0,
                    'total': latest_generation_step.total_steps if latest_generation_step else 0,
                    'percentage': latest_generation_step.percentage if latest_generation_step else 0
                } if latest_generation_step else None
            }
            
    def get_step_history(self) -> List[Dict]:
        """Get complete step history"""
        with self._lock:
            return [
                {
                    'step': step.step,
                    'total_steps': step.total_steps,
                    'percentage': step.percentage,
                    'elapsed_time': step.elapsed_time,
                    'step_duration': step.step_duration,
                    'description': step.description,
                    'task_type': step.task_type.value,
                    'timestamp': step.timestamp
                }
                for step in self._step_info
            ]
            
    def apply_tqdm_patches(self):
        """Apply tqdm patches to capture progress"""
        if self._patches_applied:
            return
            
        import tqdm
        
        # Store original methods
        self._original_tqdm_init = tqdm.tqdm.__init__
        self._original_tqdm_update = tqdm.tqdm.update
        
        # Create patched methods
        def patched_init(tqdm_self, *args, **kwargs):
            result = self._original_tqdm_init(tqdm_self, *args, **kwargs)
            # Mark instances that might be interesting
            if hasattr(tqdm_self, 'desc') and hasattr(tqdm_self, 'total'):
                desc = str(tqdm_self.desc) if tqdm_self.desc else ""
                if tqdm_self.total and tqdm_self.total <= 50:  # Potential generation or loading
                    setattr(tqdm_self, '_is_tracked', True)
            return result
            
        def patched_update(tqdm_self, n=1):
            result = self._original_tqdm_update(tqdm_self, n)
            
            # Check if this is a tracked progress bar
            if hasattr(tqdm_self, '_is_tracked') and getattr(tqdm_self, '_is_tracked', False):
                if hasattr(tqdm_self, 'n') and hasattr(tqdm_self, 'total') and tqdm_self.total:
                    desc = str(tqdm_self.desc) if hasattr(tqdm_self, 'desc') and tqdm_self.desc else ""
                    self.on_step_update(tqdm_self.n, tqdm_self.total, desc)
                    
            return result
            
        # Apply patches
        tqdm.tqdm.__init__ = patched_init
        tqdm.tqdm.update = patched_update
        
        self._patches_applied = True
        
    def remove_tqdm_patches(self):
        """Remove tqdm patches and restore original methods"""
        if not self._patches_applied or not self._original_tqdm_init:
            return
            
        import tqdm
        tqdm.tqdm.__init__ = self._original_tqdm_init
        tqdm.tqdm.update = self._original_tqdm_update
        
        self._patches_applied = False
        
    def __enter__(self):
        """Context manager entry"""
        self.reset()
        self.apply_tqdm_patches()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.remove_tqdm_patches()

# Global progress tracker instance
global_progress_tracker = ProgressTracker()

def format_progress_info(progress_info: Dict) -> str:
    """Format progress information for display"""
    if not progress_info:
        return "No progress information available"
        
    lines = []
    
    # Current task info
    task_type = progress_info.get('task_type', 'unknown')
    description = progress_info.get('description', '')
    
    if task_type == 'generation':
        # Generation progress
        step = progress_info.get('current_step', 0)
        total = progress_info.get('total_steps', 0)
        percentage = progress_info.get('percentage', 0)
        elapsed = progress_info.get('elapsed_time', 0)
        
        lines.append(f"🎨 **Generation**: Step {step}/{total} ({percentage:.1f}%)")
        lines.append(f"⏱️ **Elapsed**: {elapsed:.1f}s")
        
        if description:
            lines.append(f"📝 **Status**: {description}")
            
    elif task_type == 'loading':
        # Loading progress
        step = progress_info.get('current_step', 0)
        total = progress_info.get('total_steps', 0)
        percentage = progress_info.get('percentage', 0)
        
        lines.append(f"🔄 **Loading**: {step}/{total} ({percentage:.1f}%)")
        
        if description:
            lines.append(f"📝 **Status**: {description}")
            
    # Generation-specific progress if available
    gen_progress = progress_info.get('generation_progress')
    if gen_progress and progress_info.get('generation_started'):
        gen_step = gen_progress.get('step', 0)
        gen_total = gen_progress.get('total', 0)
        gen_percentage = gen_progress.get('percentage', 0)
        
        if gen_total > 0:
            lines.append(f"✨ **Generation**: Step {gen_step}/{gen_total} ({gen_percentage:.1f}%)")
            
    return "<br>".join(lines) if lines else "⏳ Initializing..."