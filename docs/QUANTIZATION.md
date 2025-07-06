# FluxForge Studio - Quantization Documentation

## Overview

FluxForge Studio implements advanced 8-bit quantization using `optimum.quanto` to dramatically reduce GPU memory usage while maintaining image quality. This feature enables running FLUX models on hardware with limited VRAM.

## 🎯 Key Benefits

### Memory Reduction
- **Up to 70% GPU memory savings** with 8-bit quantization
- Tested and verified on Apple Silicon (MPS) devices
- Compatible with CUDA and CPU devices

### Performance
- **Minimal quality loss** compared to full precision
- **Automatic fallback** if quantization fails
- **Cross-platform compatibility** (MPS, CUDA, CPU)

### Stability
- **Conservative implementation** using only tested qint8 quantization
- **Graceful error handling** with detailed logging
- **Production-ready** implementation without experimental features

## 🛠️ Implementation Details

### Supported Models
Quantization is available for all FLUX model variants:

- **FLUX Schnell** - Main generation model (tested ~70% memory reduction)
- **FLUX Dev** - Main generation model (same technology as Schnell)
- **FLUX Fill** - Inpainting and outpainting model
- **FLUX Kontext** - Text-based image editing model
- **FLUX Depth** - Depth-guided generation model
- **FLUX Canny** - Edge-guided generation model
- **FLUX Redux** - Image variation model (both Prior Redux and Base pipelines)

### Quantization Options

#### Available Settings
- **8-bit**: Recommended setting with proven ~70% memory reduction
- **None**: No quantization (full precision)

#### Default Configuration
- **Default**: 8-bit quantization enabled by default
- **Fallback**: Automatic fallback to full precision if quantization fails
- **Compatibility**: Only stable qint8 quantization is used

### Technical Implementation

#### Core Technology
- **Library**: `optimum.quanto` (replaces bitsandbytes for cross-platform support)
- **Quantization Type**: qint8 (8-bit integer quantization)
- **Scope**: Pipeline components (transformer, text encoders)

#### Device Compatibility
- **MPS (Apple Silicon)**: ✅ Tested and working (70% memory reduction)
- **CUDA (NVIDIA)**: ✅ Compatible (to be tested)
- **CPU**: ✅ Compatible (primarily for testing)

## 🚀 Usage

### User Interface
1. **Location**: Each FLUX tool has a "Quantization - Memory optimisation" dropdown
2. **Options**: "8-bit" (default) or "None"
3. **Placement**: Located after guidance scale controls in each tool

### Automatic Behavior
- **Default**: 8-bit quantization is enabled by default for optimal memory usage
- **Fallback**: If quantization fails, the model continues without quantization
- **Logging**: Detailed console output shows quantization status and memory savings

### Console Output Example
```
🔧 Application quantification qint8 FLUX Schnell (économie mémoire ~70%)
✅ Quantification réussie: qint8 appliquée avec succès
💾 Économie mémoire estimée: ~70%
```

## 📊 Performance Metrics

### Memory Usage (Tested on MPS)
- **Standard FLUX Schnell**: ~62.81 GB GPU memory
- **Quantized FLUX Schnell**: ~16.67 GB GPU memory
- **Memory Reduction**: 73.5%

### Generation Time Impact
- **Standard**: ~54.98s total generation time
- **Quantized**: ~69.76s total generation time (27% slower)
- **Trade-off**: Significant memory savings with acceptable time increase

### Image Quality
- **Luminosity Comparison**: Minimal difference (187.0 vs 190.1)
- **Visual Quality**: No perceptible quality loss in generated images
- **Stability**: Consistent results across multiple generations

## 🔧 Configuration

### Default Settings
The quantization selector is configured with these defaults:

```python
gr.Dropdown(
    label="Quantization - Memory optimisation",
    choices=["None", "8-bit"],
    value="8-bit",  # Default to 8-bit for optimal memory usage
    info="8-bit: ~70% memory reduction, None: no optimization"
)
```

### Integration Across Tools
Quantization is consistently available across all FLUX tools:

- **Main Generation** (Schnell/Dev)
- **FLUX Fill** (Inpainting/Outpainting)
- **FLUX Kontext** (Text-based editing)
- **FLUX Depth** (Depth-guided generation)
- **FLUX Canny** (Edge-guided generation)
- **FLUX Redux** (Image variations)

## 🐛 Troubleshooting

### Common Scenarios

#### Quantization Fails
```
⚠️  Quantification qint8 échouée: [error message]
🔄 Continuons sans quantification...
```
**Solution**: The model continues with full precision. Check GPU memory availability.

#### Unsupported Quantization
```
⚠️  Quantification 4-bit non supportée sur [device] (tests montrent erreurs)
💡 Conseil: Utilisez '8-bit' pour économie mémoire substantielle
```
**Solution**: Use 8-bit quantization instead of 4-bit (which is unstable).

#### Device Compatibility
```
⚠️  Quantification disponible pour FLUX Schnell et Dev uniquement
```
**Solution**: Quantization is available for all supported FLUX models in FluxForge Studio.

### Best Practices

#### Memory Management
1. **Enable quantization** by default for optimal memory usage
2. **Monitor console output** for quantization status
3. **Use 8-bit setting** for best balance of memory savings and stability

#### Performance Optimization
1. **Accept the time trade-off** (~27% slower) for 70% memory savings
2. **Use quantization consistently** across all tools for maximum benefit
3. **Keep quantization enabled** unless specific quality requirements demand full precision

## 🧪 Testing and Validation

### Test Environment
- **Device**: Apple Silicon (MPS)
- **Models**: FLUX Schnell, FLUX Dev
- **Quantization**: qint8 (8-bit)
- **Memory Monitoring**: Real-time GPU memory usage tracking

### Test Results
```
🎯 MEILLEUR DTYPE SUR MPS:
- float32: ✅ OK (luminosité: 187.0)
- float16: ❌ NOIR (luminosité: 0.0)  
- bfloat16: ✅ OK (luminosité: 189.0)

💾 CONSOMMATION MÉMOIRE:
- Standard: GPU: 62.81 GB
- qint8: GPU: 16.67 GB (luminosité: 190.1)
- Économie mémoire: 73.5%
- ⏱️ 27.3% plus lent que standard

🔬 Quantifications testées:
- qint8: ✅ STABLE
- qint4: ❌ ERREUR  
- qint2: ❌ ERREUR
```

### Quality Validation
- **Visual Inspection**: No perceptible quality loss
- **Luminosity Metrics**: Consistent brightness levels
- **Consistency**: Stable results across multiple generations

## 🔮 Future Considerations

### CUDA Testing
- **Status**: Compatible but not extensively tested on CUDA devices
- **Expected**: Similar memory reduction benefits
- **Validation**: Community testing needed for CUDA-specific optimizations

### Additional Optimizations
- **Memory Management**: Further optimizations possible with advanced techniques
- **Model Variants**: Potential support for additional quantization types
- **Performance**: Ongoing optimization for generation speed

## 📚 Additional Resources

### Related Documentation
- [Architecture Documentation](../architecture.md) - System architecture and quantization module
- [Setup Guide](SETUP.md) - Installation and configuration
- [Features Guide](FEATURES.md) - Complete feature overview

### Technical References
- [Optimum Quanto Documentation](https://huggingface.co/docs/optimum/quanto/index)
- [FLUX Model Documentation](https://huggingface.co/black-forest-labs)
- [Diffusers Quantization Guide](https://huggingface.co/docs/diffusers/optimization/memory)

---

**Note**: This quantization implementation prioritizes stability and cross-platform compatibility over aggressive memory reduction, ensuring reliable performance across all supported devices.