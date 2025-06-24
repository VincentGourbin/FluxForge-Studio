# MFLUX-Gradio Setup Guide

This comprehensive guide will help you set up MFLUX-Gradio on your system.

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
- **RTX 3060**: Works with 4-bit quantization
- **GTX 1660+**: Basic functionality with 4-bit quantization

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
git clone <repository-url>
cd mflux-gradio
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

#### Alternative: Manual Installation
```bash
pip install gradio torch torchvision transformers pillow ollama mflux
```

### 4. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import gradio; print(f'Gradio: {gradio.__version__}')"
```

## 🔧 Configuration

### 1. Directory Structure
Ensure the following directories exist:
```bash
mkdir -p lora outputimage temp_images temp_train stepwise_output docs
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
FLUX.1 models will be downloaded automatically on first use:
- **Schnell**: ~12GB download
- **Dev**: ~24GB download

#### Manual Model Download (Optional)
```bash
# Download models manually if needed
# Models will be cached in ~/.cache/huggingface/
```

## 🔌 Ollama Setup (Optional)

### 1. Install Ollama
Visit [ollama.ai](https://ollama.ai/) and download for your platform.

### 2. Install Models
```bash
# Text-only models
ollama pull llama3.2
ollama pull mistral

# Vision models  
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
python mflux-gradio.py
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
5. Click **Generate Image**

## 📂 File Organization

### Project Structure
```
mflux-gradio/
├── mflux-gradio.py          # Main application
├── image_generator.py       # Core generation logic
├── database.py             # Database operations
├── config.py               # Configuration
├── train.py                # Training script
├── prompt_enhancer.py      # Prompt enhancement
├── background_remover.py   # Background removal
├── requirements.txt        # Dependencies
├── README.md              # Documentation
├── CLAUDE.md              # Claude Code guidance
├── lora_info.json         # LoRA configuration
├── lora/                  # LoRA model files
├── outputimage/           # Generated images
├── temp_images/           # Training images
├── temp_train/            # Training configs
├── stepwise_output/       # Debug outputs
├── docs/                  # Documentation
└── generated_images.db    # History database
```

### Important Paths
- **LoRA Models**: `./lora/`
- **Generated Images**: `./outputimage/`
- **Training Data**: `./temp_images/`
- **Database**: `./generated_images.db`

## ⚙️ Environment Configuration

### Environment Variables
```bash
# Optional: Set environment variables
export PYTORCH_ENABLE_MPS_FALLBACK=1  # For Apple Silicon
export CUDA_VISIBLE_DEVICES=0         # Specify GPU
export GRADIO_SERVER_NAME="0.0.0.0"   # Network access
export GRADIO_SERVER_PORT=7860        # Custom port
```

### Memory Optimization
#### For Limited VRAM
```python
# In config.py, force quantization
quantize_options = [4]  # Force 4-bit only
```

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

### Database Migration
If migrating from older versions:
```bash
python migratedatabase.py
```

### Network Access
To access from other devices:
```python
# In mflux-gradio.py, modify launch parameters:
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
1. Enable quantization (8-bit or 4-bit)
2. Reduce image dimensions
3. Close other GPU applications
4. Restart the application

#### "Module not found: mflux"
**Solutions:**
1. Verify virtual environment activation
2. Reinstall requirements: `pip install -r requirements.txt`
3. Check Python version compatibility

#### "Ollama connection failed"
**Solutions:**
1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Start Ollama service: `ollama serve`
3. Verify models installed: `ollama list`

#### Slow generation
**Causes & Solutions:**
1. **Large images**: Reduce dimensions to 1024x1024
2. **No quantization**: Enable 8-bit quantization
3. **CPU fallback**: Ensure GPU drivers installed
4. **Multiple LoRAs**: Reduce number of active LoRAs

#### Training fails
**Common causes:**
1. **Insufficient disk space**: Ensure 5GB+ free
2. **Invalid images**: Use JPEG/PNG formats
3. **Memory issues**: Reduce batch size to 1
4. **Path issues**: Check image file paths

### Performance Optimization

#### GPU Optimization
```python
# Force specific GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

#### Memory Management
```python
# Clear GPU cache periodically
import torch
torch.cuda.empty_cache()
```

#### Model Caching
- Keep models loaded by maintaining parameter consistency
- Avoid changing quantization unnecessarily
- Use same LoRA combinations when possible

### Debug Mode
Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

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
du -sh outputimage/ lora/ temp_train/
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
Models update automatically. To force update:
```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/
```

### Database Maintenance
```bash
# Backup database
cp generated_images.db generated_images.db.backup

# Clean up orphaned records
python -c "
from database import *
# Custom cleanup script
"
```

### Log Rotation
```bash
# If logs become large
truncate -s 0 application.log
```

## 🚨 Security Considerations

### Network Security
- Don't expose to public internet without authentication
- Use firewall rules for network access
- Consider VPN for remote access

### File Security
- Validate uploaded training images
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
- **CLAUDE.md**: Development guidance

### Common Commands Reference
```bash
# Start application
python mflux-gradio.py

# Train standalone
python train.py --train-config config.json

# Database migration
python migratedatabase.py

# Check installation
python -c "import torch, gradio, transformers; print('OK')"
```

### Support Resources
1. Check existing GitHub issues
2. Review troubleshooting section
3. Verify system requirements
4. Test with minimal configuration