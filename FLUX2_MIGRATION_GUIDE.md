# FLUX.2 Migration Guide

## Overview

This guide documents the complete migration from multiple FLUX.1 models (dev, krea-dev, Fill, Kontext, Depth, Canny, Redux) and Qwen-Image to the unified FLUX.2-dev pipeline.

**Migration Date**: December 7, 2025
**Status**: ✅ Code Complete - Ready for Testing
**Database Records Migrated**: 198/198 (100%)

---

## What Changed

### Models Replaced

The following 8 separate models have been **replaced** by FLUX.2-dev:

1. ❌ FLUX.1-dev → ✅ FLUX.2 (text-to-image mode)
2. ❌ FLUX.1-Krea-dev → ✅ FLUX.2 (text-to-image mode)
3. ❌ Qwen-Image → ✅ FLUX.2 (text-to-image mode)
4. ❌ FLUX Fill → ✅ FLUX.2 (inpainting/outpainting modes)
5. ❌ Kontext → ✅ FLUX.2 (image-to-image mode)
6. ❌ FLUX Depth → ✅ FLUX.2 (depth-guided mode)
7. ❌ FLUX Canny → ✅ FLUX.2 (canny-guided mode)
8. ❌ FLUX Redux → ✅ FLUX.2 (image-to-image mode)

### Models Preserved

✅ **FLUX.1-schnell** - Kept for fast 4-step generation

---

## UI Changes

### Before Migration

**Two Separate Tabs**:
- 📝 Content Creation (text-to-image)
- 🎨 Post-Processing (Fill, Depth, Canny, Redux, etc.)

### After Migration

**Single Unified Tab**:
- 🎨 **Generation** - One tab with mode selector supporting:
  - ✨ Text-to-Image
  - 🔄 Image-to-Image
  - 🎨 Inpainting
  - 📐 Outpainting
  - 🌊 Depth-Guided
  - 🖋️ Canny-Guided
  - 🔀 Multi-Reference (NEW - FLUX.2 exclusive)
  - ⚡ Quick Mode (FLUX.1-schnell 4-step)

### Dynamic UI

The interface automatically shows/hides controls based on selected mode:

```
Text-to-Image Mode:
  ✅ Prompt, Negative Prompt, Steps, Guidance
  ❌ Reference images, mask editor, expansion controls

Inpainting Mode:
  ✅ Prompt, Negative Prompt, Mask Editor
  ❌ Expansion controls, multi-reference

Outpainting Mode:
  ✅ Prompt, Expansion sliders (Top/Bottom/Left/Right)
  ❌ Mask editor, multi-reference
```

---

## New Features

### 1. Multi-Reference Mode
Combine 2-3 reference images to guide generation (FLUX.2 exclusive):

```python
# Example: Blend style from image 1 with composition from image 2
Reference 1: Style reference (e.g., oil painting)
Reference 2: Composition reference (e.g., landscape layout)
Reference 3: Optional detail reference
```

### 2. Unified Negative Prompts
All modes now support negative prompts (previously Qwen-only):

```
Negative Prompt: "blurry, low quality, distorted, watermark"
```

### 3. Preview Functions
Real-time previews before queuing:

- 🔍 **Depth Map Preview**: See depth extraction before generation
- 🖋️ **Canny Edge Preview**: Visualize edge detection
- 📐 **Outpaint Preview**: Preview expansion layout
- 🎨 **Mask Preview**: Extract and view inpainting mask

### 4. Enhanced Quantization
- **BNB 4-bit** (pre-quantized): `diffusers/FLUX.2-dev-bnb-4bit`
- **Quanto 8-bit** (fallback): Cross-platform compatible
- **None**: Full precision (bfloat16)

---

## Technical Implementation

### New Files Created

1. **src/generator/flux2_generator.py** (~450 lines)
   - Unified generator class
   - 7 generation modes
   - LoRA support (up to 3 simultaneous)
   - Quantization integration

2. **src/ui/flux2_controls.py** (~350 lines)
   - Dynamic UI visibility logic
   - Preview generation functions
   - Queue integration

3. **migration_flux2.py** (~200 lines)
   - Database migration script
   - Automatic backup
   - Dry-run support

### Modified Files

1. **main.py**
   - Replaced lines 214-800 (Content Creation + Post-Processing)
   - New unified Generation tab
   - Backup: `main.py.backup_before_flux2`

2. **src/core/database.py**
   - Added `save_flux2_generation()` function
   - New metadata fields: `flux2_mode`, `control_type`

3. **src/core/processing_queue.py**
   - Added `TaskType.FLUX2_GENERATION`
   - Added `_execute_flux2_generation()` method
   - Helper methods for parameter extraction

4. **requirements.txt**
   - Updated: `gradio>=6.0.0` (Gradio 5 → 6 migration)

5. **src/ui/processing_tab.py**
   - Fixed: `col_count` → `column_count` (Gradio 6)

6. **src/ui/lora_management.py**
   - Fixed: `col_count` → `column_count` (Gradio 6)

---

## Database Migration

### Migration Summary

```
Total Records: 198
Successfully Migrated: 198 (100%)
Failed: 0

Backup Created: generated_images.db.backup_20251207_114757
```

### Legacy Type Mapping

```python
LEGACY_TO_FLUX2_MODE = {
    'standard': 'text-to-image',
    'flux_fill': 'inpainting' or 'outpainting',  # Auto-detected from metadata
    'kontext': 'image-to-image',
    'flux_depth': 'depth-guided',
    'flux_canny': 'canny-guided',
    'flux_redux': 'image-to-image',
    'qwen_generation': 'text-to-image',
    'controlnet': 'canny-guided'
}
```

### New Metadata Fields

All migrated records now include:

```json
{
  "flux2_mode": "inpainting",
  "migrated_from": "flux_fill",
  "migration_date": "2025-12-07T11:47:57",
  "model_alias": "flux2-dev"
}
```

---

## Usage Guide

### Text-to-Image

1. Select "✨ Text-to-Image" mode
2. Enter prompt: `"A serene mountain landscape at sunset"`
3. Optional: Add negative prompt: `"blurry, low quality"`
4. Adjust steps (default: 28) and guidance (default: 4.0)
5. Click "🎨 Add to Queue"

### Image-to-Image

1. Select "🔄 Image-to-Image" mode
2. Upload reference image
3. Enter prompt to guide variation
4. Adjust variation strength (0.1-1.0)
5. Click "🎨 Add to Queue"

### Inpainting

1. Select "🎨 Inpainting" mode
2. Upload image in mask editor
3. Draw white mask over areas to regenerate
4. Optional: Click "Preview Mask" to verify
5. Enter prompt for regenerated areas
6. Click "🎨 Add to Queue"

### Outpainting

1. Select "📐 Outpainting" mode
2. Upload base image
3. Adjust expansion sliders (Top/Bottom/Left/Right %)
4. Optional: Click "Preview Expansion" to see layout
5. Enter prompt for extended areas
6. Click "🎨 Add to Queue"

### Depth-Guided

1. Select "🌊 Depth-Guided" mode
2. Upload reference image
3. Click "🔍 Generate Depth Map" to preview
4. Enter prompt for final image
5. Click "🎨 Add to Queue"

### Canny-Guided

1. Select "🖋️ Canny-Guided" mode
2. Upload reference image
3. Adjust thresholds (low: 100, high: 200)
4. Preview edge detection updates automatically
5. Enter prompt for final image
6. Click "🎨 Add to Queue"

### Multi-Reference (NEW)

1. Select "🔀 Multi-Reference" mode
2. Upload 2-3 reference images
3. Enter prompt describing desired blend
4. FLUX.2 will combine references intelligently
5. Click "🎨 Add to Queue"

### Quick Mode (Schnell)

1. Select "⚡ Quick Mode (schnell 4-step)"
2. Enter prompt
3. Image generates in ~2-5 seconds (4 steps)
4. No guidance scale or negative prompt
5. Best for rapid prototyping

---

## LoRA Support

FLUX.2 supports up to 3 simultaneous LoRA models across ALL modes:

```
LoRA 1: Style LoRA (scale: 0.8)
LoRA 2: Character LoRA (scale: 1.0)
LoRA 3: Environment LoRA (scale: 0.6)
```

**Compatibility**: Most FLUX.1 LoRAs work with FLUX.2. Test your existing LoRAs to verify.

---

## Quantization Options

### BNB 4-bit (Recommended)

```python
quantization = "4-bit (BNB)"
# Uses: diffusers/FLUX.2-dev-bnb-4bit
# Memory: ~8-10 GB VRAM
# Speed: Fast
# Quality: Minimal degradation
```

### Quanto 8-bit (Fallback)

```python
quantization = "8-bit (Quanto)"
# Uses: black-forest-labs/FLUX.2-dev + runtime quantization
# Memory: ~12-14 GB VRAM
# Speed: Medium
# Quality: Excellent
# Note: Fallback if BNB fails on MPS
```

### None (Full Precision)

```python
quantization = "None"
# Uses: black-forest-labs/FLUX.2-dev (bfloat16)
# Memory: ~24+ GB VRAM
# Speed: Fastest (if enough VRAM)
# Quality: Best
```

---

## Performance Expectations

### Generation Times (Approximate)

| Mode | Steps | Quantization | Time (A100) | Time (MPS M2) |
|------|-------|--------------|-------------|---------------|
| Text-to-Image | 28 | 4-bit | 8-12s | 25-40s |
| Image-to-Image | 28 | 4-bit | 10-15s | 30-45s |
| Inpainting | 28 | 4-bit | 10-15s | 30-45s |
| Quick Mode | 4 | None | 2-3s | 5-8s |

### Memory Usage

| Configuration | VRAM/MPS Required |
|---------------|-------------------|
| FLUX.2 (4-bit) | 8-10 GB |
| FLUX.2 (8-bit) | 12-14 GB |
| FLUX.2 (None) | 24+ GB |
| Schnell (None) | 12-16 GB |

---

## Troubleshooting

### Issue: "Out of Memory" Error

**Solution 1**: Use quantization
```python
quantization = "4-bit (BNB)"  # or "8-bit (Quanto)"
```

**Solution 2**: Reduce resolution
```python
width = 512  # Instead of 1024
height = 512
```

**Solution 3**: Clear queue between tasks
```
Processing Tab → Clear Queue → Memory cleanup automatic
```

### Issue: BNB 4-bit Fails on Apple Silicon

**Solution**: System automatically falls back to Quanto 8-bit
```python
# This is handled automatically - no user action needed
# Check logs for: "BNB 4-bit not supported on MPS, falling back to Quanto 8-bit"
```

### Issue: LoRA Not Loading

**Solution 1**: Verify LoRA compatibility
```bash
# Check LoRA file size and format
ls -lh lora/your_lora.safetensors
```

**Solution 2**: Test with single LoRA
```
Disable LoRA 2 and 3, test with only LoRA 1
```

### Issue: Preview Functions Not Working

**Solution**: Check reference image is uploaded
```
Depth Preview requires: depth_input image uploaded
Canny Preview requires: canny_input image uploaded
Outpaint Preview requires: outpaint_image uploaded
```

---

## Rollback Procedure

If you need to revert to the old system:

### Step 1: Restore Code

```bash
# Restore main.py
cp main.py.backup_before_flux2 main.py

# Restore database
cp generated_images.db.backup_20251207_114757 generated_images.db
```

### Step 2: Downgrade Gradio

```bash
pip install "gradio>=4.0.0,<6.0.0"
```

### Step 3: Restart Application

```bash
python main.py
```

**Note**: All FLUX.2 generations will show as "flux2_generation" type in history and may not display perfectly in legacy UI.

---

## Testing Checklist

Before deploying to production, verify:

- [ ] Application starts without errors
- [ ] Generation tab renders correctly
- [ ] Mode selector switches UI panels
- [ ] Text-to-Image generates successfully
- [ ] Image-to-Image works with uploaded reference
- [ ] Inpainting mask editor functional
- [ ] Outpainting preview shows expansion correctly
- [ ] Depth preview generates depth map
- [ ] Canny preview shows edge detection
- [ ] Multi-reference accepts 2-3 images
- [ ] Quick Mode (schnell) still works
- [ ] LoRA loading/unloading works
- [ ] Quantization options function
- [ ] Queue integration successful
- [ ] Database saves FLUX.2 metadata
- [ ] History tab displays migrated records
- [ ] Performance acceptable (<30s for 28 steps)
- [ ] Memory usage within limits

---

## Known Limitations

1. **LoRA Compatibility**: Not all FLUX.1 LoRAs guaranteed to work with FLUX.2 - testing required
2. **BNB 4-bit on MPS**: May fail - automatic fallback to Quanto 8-bit
3. **Multi-Reference Mode**: New feature - optimal usage patterns still being discovered
4. **Generation Time**: FLUX.2 slightly slower than FLUX.1-dev due to larger model (32B vs 12B params)

---

## Future Enhancements

Potential improvements for future releases:

1. **FLUX.2 LoRA Training**: Adapt training pipeline for FLUX.2
2. **Preset Modes**: Save/load mode configurations
3. **Batch Multi-Reference**: Process multiple reference sets
4. **Advanced Controlnet**: Additional control modes
5. **Remote Text Encoder**: Offload to HuggingFace endpoint for memory savings

---

## Support

### Get Help

- Check logs in terminal for error messages
- Verify Python environment is activated
- Ensure dependencies installed: `pip install -r requirements.txt`
- Review this guide's Troubleshooting section

### Report Issues

If you encounter bugs:

1. Note error message from terminal
2. Document steps to reproduce
3. Include system info (OS, RAM, GPU/MPS)
4. Check database backup exists before reporting data issues

---

## Migration Credits

**Migration Date**: December 7, 2025
**Migration Tool**: Claude Code (Anthropic)
**Files Modified**: 6 core files
**Files Created**: 3 new modules
**Lines Changed**: ~1,200+
**Database Records Migrated**: 198
**Backup Created**: ✅ Automatic timestamped backup

**Testing Status**: Code complete - awaiting end-to-end validation

---

## Changelog

### Version 2.0.0 (FLUX.2 Migration)

**Added**:
- FLUX.2-dev unified pipeline support
- Multi-reference generation mode
- Dynamic UI with mode selector
- Preview functions (depth, canny, outpaint, mask)
- Enhanced quantization (BNB 4-bit + Quanto 8-bit)
- Database migration with automatic backup

**Changed**:
- Merged Content Creation + Post-Processing tabs → Generation tab
- Updated to Gradio 6
- Enhanced metadata schema with flux2_mode field

**Removed**:
- FLUX.1-dev separate pipeline (replaced by FLUX.2)
- FLUX.1-Krea-dev separate pipeline (replaced by FLUX.2)
- Qwen-Image separate pipeline (replaced by FLUX.2)
- FLUX Fill separate pipeline (replaced by FLUX.2)
- Kontext separate pipeline (replaced by FLUX.2)
- FLUX Depth separate pipeline (replaced by FLUX.2)
- FLUX Canny separate pipeline (replaced by FLUX.2)
- FLUX Redux separate pipeline (replaced by FLUX.2)

**Preserved**:
- FLUX.1-schnell (quick 4-step mode)
- All 198 historical generation records
- LoRA management system
- Processing queue system
- History gallery
- Prompt enhancer
- Background removal
- Admin tools

---

## Conclusion

The FLUX.2 migration successfully consolidates 8 separate models into a single, more capable pipeline while preserving all historical data and enhancing the user experience with a unified, mode-based interface.

**Next Steps**:
1. Activate Python environment
2. Install dependencies: `pip install -r requirements.txt`
3. Launch application: `python main.py`
4. Test all generation modes
5. Verify LoRA compatibility with your collection
6. Monitor performance and memory usage

Welcome to FLUX.2! 🎨
