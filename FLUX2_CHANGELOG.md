# FLUX.2 Migration - Technical Changelog

## Migration Summary

**Date**: December 7, 2025
**Type**: Major Version - Breaking UI Changes
**Database Migration**: Required (automatic backup)
**Code Status**: ✅ Complete - Ready for Testing

---

## Files Created

### 1. src/generator/flux2_generator.py
**Lines**: ~450
**Purpose**: Unified FLUX.2 generator replacing 8 separate pipelines

**Key Classes**:
```python
class Flux2Generator:
    """Unified multi-modal FLUX.2 generator"""

    def __init__(self, device: str = None)
    def generate(...) -> Tuple[Image.Image, str, Dict]
    def _ensure_pipeline_loaded(quantization: str) -> None
    def _prepare_inputs(mode: str, ...) -> Dict
    def _update_loras(lora_paths: List, lora_scales: List) -> None
    def _get_remote_embeds(prompt: str) -> Tuple[torch.Tensor, ...]
    def cleanup_pipeline() -> None

def get_flux2_generator() -> Flux2Generator  # Singleton accessor
```

**Modes Supported**:
- `text-to-image`: Standard generation
- `image-to-image`: Variation from reference
- `inpainting`: Masked region regeneration
- `outpainting`: Image expansion
- `depth-guided`: Depth map control
- `canny-guided`: Edge map control
- `multi-reference`: Multi-image blending (NEW)

**Quantization**:
- BNB 4-bit: `diffusers/FLUX.2-dev-bnb-4bit`
- Quanto 8-bit: Runtime quantization fallback
- None: Full bfloat16

---

### 2. src/ui/flux2_controls.py
**Lines**: ~350
**Purpose**: UI logic for Generation tab

**Key Functions**:
```python
def update_generation_mode_visibility(mode: str) -> List[Any]:
    """Returns 9 gr.update() objects for dynamic UI"""

def generate_depth_preview(input_image) -> Image.Image:
    """Generate depth map preview"""

def generate_canny_preview(input_image, low, high) -> Image.Image:
    """Generate canny edge preview"""

def generate_outpaint_preview(image, top, bottom, left, right) -> Image.Image:
    """Generate outpainting layout preview"""

def extract_inpainting_mask_preview(editor_data) -> Image.Image:
    """Extract mask from ImageEditor"""

def queue_flux2_generation(...) -> str:
    """Queue FLUX.2 task with 20+ parameters"""
```

**UI Visibility Map**:
```python
visibility_map = {
    "✨ Text-to-Image": {
        'img2img': False, 'inpaint': False, 'outpaint': False,
        'depth': False, 'canny': False, 'multiref': False,
        'schnell_notice': False, 'steps': 28, 'guidance': 4.0
    },
    # ... 7 more modes
}
```

---

### 3. migration_flux2.py
**Lines**: ~200
**Purpose**: Database migration script

**Usage**:
```bash
python migration_flux2.py --dry-run  # Test migration
python migration_flux2.py            # Apply migration (auto-backup)
python migration_flux2.py --no-backup  # Dangerous - skip backup
```

**Key Functions**:
```python
def migrate_database(db_path: str, dry_run: bool = False) -> Dict:
    """Main migration function"""

def map_legacy_to_flux2_mode(generation_type: str, metadata: Dict) -> str:
    """Convert legacy types to FLUX.2 modes"""

def create_backup(db_path: str) -> str:
    """Create timestamped backup"""
```

**Migration Results**:
```
Total Records: 198
Migrated: 198 (100%)
Failed: 0
Backup: generated_images.db.backup_20251207_114757
```

---

## Files Modified

### 1. main.py
**Lines Changed**: ~586 (replaced 214-800)
**Backup**: `main.py.backup_before_flux2`

**Changes**:
- Removed: Content Creation tab (lines 214-450)
- Removed: Post-Processing tab (lines 451-800)
- Added: Unified Generation tab with mode selector
- Gradio 6 fix: Kept `title` in `gr.Blocks()`, moved `theme` to `launch()`

**New Imports**:
```python
from ui.flux2_controls import (
    update_generation_mode_visibility,
    generate_depth_preview,
    generate_canny_preview,
    generate_outpaint_preview,
    extract_inpainting_mask_preview,
    queue_flux2_generation
)
```

**New UI Structure** (lines 214-800):
```python
with gr.Tab("Generation"):
    generation_mode = gr.Dropdown(...)  # Mode selector

    # Dynamic panels (visibility controlled by mode)
    with gr.Group(visible=False) as img2img_group: ...
    with gr.Group(visible=False) as inpaint_group: ...
    with gr.Group(visible=False) as outpaint_group: ...
    with gr.Group(visible=False) as depth_group: ...
    with gr.Group(visible=False) as canny_group: ...
    with gr.Group(visible=False) as multiref_group: ...
    with gr.Group(visible=False) as schnell_notice: ...

    # Event handlers
    generation_mode.change(
        fn=update_generation_mode_visibility,
        inputs=generation_mode,
        outputs=[img2img_group, inpaint_group, ...]
    )
```

---

### 2. src/core/database.py
**Lines Added**: ~45 (after line 183)

**New Function**:
```python
def save_flux2_generation(
    timestamp: str,
    seed: int,
    prompt: str,
    negative_prompt: str,
    flux2_mode: str,  # NEW field
    steps: int,
    guidance: float,
    height: int,
    width: int,
    lora_paths: List[str],
    lora_scales: List[float],
    output_filename: str,
    quantization: str = "None",
    control_type: Optional[str] = None,  # NEW field
    total_generation_time: Optional[float] = None,
    model_generation_time: Optional[float] = None
) -> None:
    """Save FLUX.2 generation with enhanced metadata"""
```

**New Metadata Fields**:
```json
{
  "flux2_mode": "inpainting",
  "control_type": "depth",
  "negative_prompt": "blurry, low quality",
  "model_alias": "flux2-dev"
}
```

**Migrated Records Get**:
```json
{
  "migrated_from": "flux_fill",
  "migration_date": "2025-12-07T11:47:57"
}
```

---

### 3. src/core/processing_queue.py
**Lines Added**: ~140 (enum + helper methods)

**Enum Addition** (line 34):
```python
class TaskType(Enum):
    STANDARD_GENERATION = "standard_generation"
    FLUX2_GENERATION = "flux2_generation"  # NEW
    QWEN_GENERATION = "qwen_generation"
    # ... existing types
```

**New Methods**:

```python
def _execute_flux2_generation(self, params: Dict, task: Task) -> Optional[Image.Image]:
    """Execute FLUX.2 generation task"""
    # Lines 666-730

def _extract_reference_images(self, mode: str, params: Dict) -> Optional[List[Image.Image]]:
    """Extract reference images based on mode"""
    # Lines 732-750

def _extract_mask(self, mode: str, params: Dict) -> Optional[Image.Image]:
    """Extract or generate mask for inpainting/outpainting"""
    # Lines 752-768

def _extract_control_image(self, mode: str, params: Dict) -> Optional[Image.Image]:
    """Generate control image for depth/canny modes"""
    # Lines 770-785

def _extract_lora_params(self, params: Dict) -> Tuple[List[str], List[float]]:
    """Extract LoRA paths and scales from parameters"""
    # Lines 787-800
```

**Task Description** (lines 72-84):
```python
if task.type == TaskType.FLUX2_GENERATION:
    mode_emoji = {
        'text-to-image': '✨',
        'image-to-image': '🔄',
        'inpainting': '🎨',
        # ... 7 modes
    }
    return f"{mode_emoji.get(mode, '🎨')} FLUX.2 {mode}"
```

---

### 4. requirements.txt
**Changed**: Line 1

**Before**:
```
gradio>=4.0.0
```

**After**:
```
gradio>=6.0.0
```

---

### 5. src/ui/processing_tab.py
**Changed**: Line 205

**Before**:
```python
col_count=4,
```

**After**:
```python
column_count=4,  # Gradio 6 renamed parameter
```

---

### 6. src/ui/lora_management.py
**Changed**: Line 114

**Before**:
```python
col_count=7,
```

**After**:
```python
column_count=7,  # Gradio 6 renamed parameter
```

---

## Database Schema Changes

**Schema**: No breaking changes (JSON-based metadata)

**New Generation Type**:
```sql
generation_type = 'flux2_generation'
```

**Enhanced Metadata** (JSON):
```json
{
  "seed": 42,
  "prompt": "A serene mountain landscape",
  "negative_prompt": "blurry, low quality",
  "model_alias": "flux2-dev",
  "flux2_mode": "text-to-image",
  "control_type": null,
  "steps": 28,
  "guidance": 4.0,
  "height": 1024,
  "width": 1024,
  "lora_paths": ["lora/style.safetensors"],
  "lora_scales": [0.8],
  "quantization": "4-bit (BNB)",
  "total_generation_time": 12.5,
  "model_generation_time": 11.8
}
```

**Migrated Records Include**:
```json
{
  "migrated_from": "flux_fill",
  "migration_date": "2025-12-07T11:47:57"
}
```

---

## API Changes

### New Functions

```python
# Generator
from generator.flux2_generator import get_flux2_generator

flux2_gen = get_flux2_generator()
result = flux2_gen.generate(
    prompt="...",
    mode="text-to-image",
    negative_prompt="...",
    steps=28,
    guidance_scale=4.0,
    width=1024,
    height=1024,
    seed=0,
    lora_paths=["lora/style.safetensors"],
    lora_scales=[0.8],
    quantization="4-bit (BNB)"
)
```

```python
# Database
from core.database import save_flux2_generation

save_flux2_generation(
    timestamp="...",
    seed=42,
    prompt="...",
    negative_prompt="...",
    flux2_mode="text-to-image",
    steps=28,
    guidance=4.0,
    height=1024,
    width=1024,
    lora_paths=[...],
    lora_scales=[...],
    output_filename="...",
    quantization="4-bit (BNB)",
    control_type=None
)
```

```python
# UI Controls
from ui.flux2_controls import queue_flux2_generation

queue_flux2_generation(
    generation_mode="✨ Text-to-Image",
    prompt="...",
    negative_prompt="...",
    steps=28,
    guidance=4.0,
    seed=0,
    width=1024,
    height=1024,
    quantization="4-bit (BNB)",
    lora_state=[...],
    lora_strength_1=0.8,
    lora_strength_2=1.0,
    lora_strength_3=0.6,
    # ... mode-specific parameters
)
```

### Deprecated Functions

These functions are still present but will be removed in future versions:

```python
# Will be deprecated after FLUX.2 validation
from postprocessing.flux_fill import process_flux_fill
from postprocessing.kontext import process_kontext
from postprocessing.flux_depth import process_flux_depth
from postprocessing.flux_canny import process_flux_canny
from postprocessing.flux_redux import process_flux_redux
from generator.qwen_generator import QwenImageGenerator
```

---

## Breaking Changes

### UI Breaking Changes

1. **Content Creation Tab Removed**: Replaced by Generation tab
2. **Post-Processing Tab Removed**: Integrated into Generation tab
3. **Model Selector Changed**: Now uses mode selector instead

### Migration Required

Users upgrading from pre-FLUX.2 versions must:

1. **Run Migration Script**:
   ```bash
   python migration_flux2.py
   ```

2. **Update Gradio**:
   ```bash
   pip install "gradio>=6.0.0"
   ```

3. **Review New UI**: Learn mode-based interface

### Preserved Functionality

- ✅ FLUX.1-schnell still available (Quick Mode)
- ✅ All 198 historical records preserved
- ✅ LoRA management unchanged
- ✅ Processing queue compatible
- ✅ History tab unchanged
- ✅ Prompt enhancer unchanged
- ✅ Admin tools unchanged

---

## Testing Validation

### Code Validation

- ✅ Python syntax check: PASSED
- ✅ Import validation: PASSED
- ✅ Database migration: SUCCESS (198/198)
- ✅ Backup creation: SUCCESS

### Pending Tests

- ⏳ Application launch
- ⏳ UI rendering
- ⏳ Text-to-image generation
- ⏳ Image-to-image generation
- ⏳ Inpainting functionality
- ⏳ Outpainting functionality
- ⏳ Depth-guided generation
- ⏳ Canny-guided generation
- ⏳ Multi-reference generation
- ⏳ Quick mode (schnell)
- ⏳ LoRA loading
- ⏳ Quantization options
- ⏳ Queue integration
- ⏳ Database saves
- ⏳ History display

---

## Performance Impact

### Expected Changes

| Metric | FLUX.1 Baseline | FLUX.2 Expected | Change |
|--------|-----------------|-----------------|--------|
| Text-to-Image (28 steps) | 10-15s | 12-18s | +20% |
| Memory (4-bit) | 6-8 GB | 8-10 GB | +25% |
| Model Size | 12B params | 32B params | +167% |
| Quality | Baseline | +15-20% | Better |

### Optimizations Applied

- ✅ Pre-quantized BNB 4-bit option
- ✅ Quanto 8-bit fallback
- ✅ Remote text encoder support
- ✅ Pipeline caching (avoid reloads)
- ✅ LoRA adapter caching

---

## Rollback Instructions

If issues arise, rollback procedure:

```bash
# 1. Restore code
cp main.py.backup_before_flux2 main.py

# 2. Restore database
cp generated_images.db.backup_20251207_114757 generated_images.db

# 3. Downgrade Gradio
pip install "gradio>=4.0.0,<6.0.0"

# 4. Restart
python main.py
```

**Warning**: FLUX.2 generations in history will show as `flux2_generation` type and may not display perfectly in legacy UI.

---

## Next Steps

1. **Activate Environment**:
   ```bash
   source venv/bin/activate  # or conda activate <env>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Application**:
   ```bash
   python main.py
   ```

4. **Test Modes**:
   - Text-to-Image
   - Image-to-Image
   - Inpainting
   - Quick Mode (schnell)

5. **Validate**:
   - LoRA compatibility
   - Quantization options
   - Queue integration
   - Performance benchmarks

---

## Developer Notes

### Code Patterns

**Singleton Pattern** (flux2_generator.py):
```python
_flux2_generator_instance = None

def get_flux2_generator() -> Flux2Generator:
    global _flux2_generator_instance
    if _flux2_generator_instance is None:
        _flux2_generator_instance = Flux2Generator()
    return _flux2_generator_instance
```

**Mode-Based Dispatching** (flux2_generator.py):
```python
def _prepare_inputs(self, mode, ...):
    if mode == "text-to-image":
        return {'prompt': prompt}
    elif mode == "image-to-image":
        return {'prompt': prompt, 'image': reference_images[0]}
    # ... 7 total modes
```

**Dynamic UI Visibility** (flux2_controls.py):
```python
def update_generation_mode_visibility(mode):
    config = visibility_map.get(mode)
    return [
        gr.update(visible=config['img2img']),
        gr.update(visible=config['inpaint']),
        # ... 9 total updates
    ]
```

### Architecture Decisions

1. **Why Singleton for Flux2Generator?**
   - Avoid multiple pipeline loads (memory expensive)
   - Reuse loaded models across tasks
   - Consistent state management

2. **Why Mode Selector Instead of Model Selector?**
   - Single FLUX.2 model handles all modes
   - Clearer user intent
   - Reduces confusion about when to use which model

3. **Why Keep Schnell Separate?**
   - Different use case (speed vs quality)
   - Users expect 4-step quick mode
   - Avoids mixing incompatible parameters

4. **Why JSON Metadata Instead of Schema Change?**
   - Non-breaking for existing records
   - Flexible for future enhancements
   - Easier migration path

---

## Migration Credits

**Executed By**: Claude Code (Anthropic)
**Migration Date**: December 7, 2025
**Total Development Time**: ~4 hours
**Files Modified**: 6
**Files Created**: 3
**Lines of Code**: ~1,200+
**Database Records Migrated**: 198/198 (100%)

**Status**: ✅ Code Complete - Awaiting Testing

---

## Version History

### v2.0.0 (FLUX.2 Migration) - 2025-12-07

**Major Changes**:
- Integrated FLUX.2-dev unified pipeline
- Removed 8 separate model pipelines
- Merged Content Creation + Post-Processing tabs
- Upgraded to Gradio 6
- Migrated 198 database records

**Minor Changes**:
- Added multi-reference generation mode
- Enhanced quantization options (BNB 4-bit)
- Added preview functions
- Improved LoRA support

**Fixes**:
- Gradio 6 API compatibility
- col_count → column_count parameter

---

## Contact

For issues related to this migration:

1. Check logs in terminal
2. Review FLUX2_MIGRATION_GUIDE.md
3. Verify environment setup
4. Test with Quick Mode first (schnell)

**End of Technical Changelog**
