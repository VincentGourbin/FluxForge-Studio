# FluxForge Studio Setup Guide

This comprehensive guide will help you set up FluxForge Studio on your system with the latest diffusers-based architecture.

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 20GB free space for models and outputs
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)

### Recommended Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM or Apple Silicon Mac
- **RAM**: 32GB for optimal performance
- **Storage**: SSD with 50GB+ free space
- **Internet**: Stable connection for model downloads

### GPU Support
#### NVIDIA GPUs (CUDA)
- **RTX 4090**: Optimal performance with full precision
- **RTX 4080/4070**: Good performance with 8-bit quantization
- **RTX 3080/3070**: Recommended with 8-bit quantization
- **RTX 3060**: Works with 8-bit quantization
- **GTX 1660+**: Basic functionality with 8-bit quantization

#### Apple Silicon (MPS)
- **M3 Max/Ultra**: Excellent performance
- **M2 Max/Ultra**: Very good performance
- **M2 Pro**: Good performance with optimization
- **M1 Pro/Max**: Good performance
- **M1**: Basic functionality

#### AMD GPUs
- Limited support through CPU fallback
- Consider using CPU mode for AMD systems

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/VincentGourbin/FluxForge-Studio.git
cd FluxForge-Studio
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

The requirements.txt includes:
- **diffusers**: From main branch for latest FLUX features
- **optimum[quanto]**: For cross-platform 8-bit quantization
- **torch**: With MPS support for Apple Silicon
- **gradio**: Web interface framework
- **transformers**: Model loading and processing

### 4. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import gradio; print(f'Gradio: {gradio.__version__}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "from optimum.quanto import qint8; print('Quantization: OK')"
```

## 🔧 Configuration

### 1. Directory Structure
The project uses a modular src/ structure:
```
FluxForge-Studio/
├── main.py                     # Application entry point
├── src/                        # Source code modules
│   ├── core/                   # Core functionality
│   ├── generator/              # Image generation
│   ├── postprocessing/         # FLUX tools (Canny, Depth, Fill, etc.)
│   ├── enhancement/            # Prompt enhancement
│   ├── ui/                     # User interface
│   └── utils/                  # Utilities and quantization
├── lora/                       # LoRA model files
├── outputimage/                # Generated images
└── temp_images/                # Temporary storage
```

### 2. LoRA Configuration
Create and configure `lora_info.json`:
```json
[
    {
        "file_name": "example_lora.safetensors",
        "description": "Example LoRA description",
        "activation_keyword": "trigger_word"
    }
]
```

### 3. Model Downloads
FLUX.1 models are downloaded automatically via diffusers:
- **Schnell**: `black-forest-labs/FLUX.1-schnell` (~12GB)
- **Dev**: `black-forest-labs/FLUX.1-dev` (~24GB)

Additional models for postprocessing tools:
- **FLUX Canny**: `black-forest-labs/FLUX.1-Canny-dev`
- **FLUX Depth**: `black-forest-labs/FLUX.1-Depth-dev`
- **FLUX Fill**: `black-forest-labs/FLUX.1-Fill-dev`
- **FLUX Redux**: `black-forest-labs/FLUX.1-Redux-dev`

## 🔌 Ollama Setup (Optional)

### 1. Install Ollama
Visit [ollama.ai](https://ollama.ai/) and download for your platform.

### 2. Install Models
```bash
# Text-only models
ollama pull llama3.2
ollama pull mistral

# Vision models for image analysis
ollama pull llama3.2-vision
ollama pull bakllava

# Verify installation
ollama list
```

### 3. Start Ollama Service
```bash
# Ollama typically starts automatically
# Verify it's running:
ollama serve
```

## 🚀 First Run

### 1. Start Application
```bash
python main.py
```

For public sharing with authentication:
```bash
python main.py --share
# or
python main.py -s
```

### 2. Access Interface
Open your browser and navigate to:
```
http://localhost:7860
```

### 3. Test Basic Generation
1. Navigate to the **Generation** tab
2. Enter a simple prompt: "a beautiful landscape"
3. Select **schnell** model
4. Set **4 steps**
5. Enable **8-bit quantization** if needed
6. Click **Generate Image**

## 📂 Architecture Overview

### Core Components
- **src/core/config.py**: Device detection, model options, LoRA data
- **src/core/database.py**: SQLite operations for image history
- **src/generator/image_generator.py**: Main FLUX generation with diffusers
- **src/utils/quantization.py**: Cross-platform quantization with optimum.quanto

### FLUX Postprocessing Tools
- **src/postprocessing/flux_canny.py**: Canny edge ControlNet
- **src/postprocessing/flux_depth.py**: Depth-based generation
- **src/postprocessing/flux_fill.py**: Inpainting and outpainting
- **src/postprocessing/flux_redux.py**: Image-to-image with Redux
- **src/postprocessing/kontext.py**: Text-guided image editing
- **src/postprocessing/upscaler.py**: ControlNet upscaling

### Important Features
- **Model Caching**: Intelligent caching prevents unnecessary reloads
- **LoRA Compatibility**: Quantization applied AFTER LoRA loading
- **Memory Optimization**: Up to 70% memory reduction with 8-bit quantization
- **Cross-Platform**: MPS (Apple Silicon), CUDA, and CPU support

## ⚙️ Environment Configuration

### Environment Variables
```bash
# Apple Silicon optimization
export PYTORCH_ENABLE_MPS_FALLBACK=1

# CUDA configuration
export CUDA_VISIBLE_DEVICES=0

# Gradio configuration
export GRADIO_SERVER_NAME="0.0.0.0"
export GRADIO_SERVER_PORT=7860
```

### Memory Optimization
#### For Limited VRAM
- Enable **8-bit quantization** in the interface
- Reduce image dimensions (1024x1024 recommended)
- Limit number of active LoRAs

#### For Apple Silicon
```bash
# Ensure MPS support
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## 🔧 Advanced Configuration

### Custom Model Paths
To use local FLUX.1 models:
1. Place model files in a directory
2. In the Generation tab, enter the path in "Custom Model Path"
3. Generate as normal

### Quantization Settings
The application supports:
- **8-bit quantization**: Recommended for most systems (70% memory reduction)
- **4-bit quantization**: Currently disabled (experimental)
- **No quantization**: Full precision for maximum quality

### Network Access
To access from other devices, modify launch parameters in `main.py`:
```python
demo.queue().launch(
    server_name="0.0.0.0",  # Allow external access
    server_port=7860,
    share=False  # Set to True for gradio.live URL
)
```

## 🐛 Troubleshooting

### Common Issues

#### "CUDA out of memory"
**Solutions:**
1. Enable 8-bit quantization
2. Reduce image dimensions
3. Close other GPU applications
4. Restart the application

#### "Module not found: diffusers"
**Solutions:**
1. Verify virtual environment activation
2. Reinstall requirements: `pip install -r requirements.txt`
3. Check Python version compatibility

#### "Ollama connection failed"
**Solutions:**
1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Start Ollama service: `ollama serve`
3. Verify models installed: `ollama list`

#### LoRA loading failures
**Common causes:**
1. **Quantization order**: Ensure quantization is applied AFTER LoRA loading
2. **File format**: Use .safetensors files
3. **Compatibility**: Some LoRAs may not be compatible with FLUX

#### Slow generation
**Causes & Solutions:**
1. **Large images**: Reduce dimensions to 1024x1024
2. **No quantization**: Enable 8-bit quantization
3. **CPU fallback**: Ensure GPU drivers installed
4. **Multiple LoRAs**: Reduce number of active LoRAs

### Performance Optimization

#### GPU Optimization
```python
# Force specific GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

#### Memory Management
- The application automatically clears GPU cache after generation
- Model caching prevents unnecessary reloads
- Quantization can reduce memory usage by 70%

#### Model Caching
- Keep models loaded by maintaining parameter consistency
- Avoid changing quantization unnecessarily
- Use same LoRA combinations when possible

## 📊 System Monitoring

### Monitor GPU Usage
```bash
# NVIDIA
nvidia-smi

# Apple Silicon
sudo powermetrics --show-usage-summary
```

### Monitor Disk Space
```bash
# Check available space
df -h

# Monitor specific directories
du -sh outputimage/ lora/ temp_images/
```

### Monitor Memory
```bash
# System memory
free -h  # Linux
top      # macOS/Linux
```

## 🔄 Updates and Maintenance

### Updating Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Updating Models
Models update automatically via diffusers. To force update:
```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/
```

### Database Maintenance
```bash
# Backup database
cp generated_images.db generated_images.db.backup
```

## 🚨 Security Considerations

### Network Security
- Don't expose to public internet without authentication
- Use firewall rules for network access
- Consider VPN for remote access

### File Security
- Validate uploaded images
- Monitor disk usage for DoS prevention
- Regular security updates

### Model Security
- Only use trusted LoRA sources
- Verify model checksums when possible
- Scan downloaded content

## 📞 Getting Help

### Documentation
- **README.md**: Basic usage guide
- **FEATURES.md**: Detailed feature documentation
- **API.md**: Technical API reference
- **QUANTIZATION.md**: Quantization guide
- **CLAUDE.md**: Development guidance

## 🔗 Repository Information

**Project**: FluxForge Studio  
**Repository**: https://github.com/VincentGourbin/FluxForge-Studio  
**Main Directory**: FluxForge-Studio/ (after cloning)  
**Entry Point**: main.py

### Common Commands Reference
```bash
# Start application
python main.py

# Start with public sharing
python main.py --share

# Check installation
python -c "import torch, gradio, diffusers; print('OK')"

# Test quantization
python -c "from optimum.quanto import qint8; print('Quantization OK')"
```

### Support Resources
1. Check existing GitHub issues
2. Review troubleshooting section
3. Verify system requirements
4. Test with minimal configuration

## 🎯 Key Differences from Previous Versions

### Architecture Changes
- **Diffusers-based**: Migrated from mflux to diffusers library
- **Modular structure**: Organized src/ directory with logical modules
- **Quantization**: Cross-platform support with optimum.quanto
- **FLUX Tools**: Dedicated modules for Canny, Depth, Fill, Redux, Kontext

### Performance Improvements
- **LoRA Compatibility**: Quantization applied after LoRA loading
- **Memory Optimization**: Up to 70% memory reduction
- **Model Caching**: Intelligent caching system
- **Cross-Platform**: Better MPS and CUDA support

### New Features
- **Dynamic Tool System**: Modular selection of postprocessing tools
- **Enhanced Quantization**: 8-bit quantization with graceful fallback
- **Improved UI**: Better organization and user experience
- **Comprehensive Logging**: Detailed generation parameter tracking