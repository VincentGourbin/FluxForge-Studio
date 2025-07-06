# FluxForge Studio: Advanced FLUX.1 Image Generation Interface

A comprehensive web-based interface for FLUX.1 image generation with dynamic tool system, LoRA support, prompt enhancement, and background removal capabilities.

## 🚀 Features

### 🎨 Advanced Image Generation
- **FLUX.1 Models**: Support for schnell and dev variants with diffusers library
- **Memory Optimization**: 8-bit quantization with ~70% memory reduction using optimum.quanto
- **Dynamic Tool System**: Modular selection of LoRA, ControlNet, and FLUX Tools
- **LoRA Integration**: Multiple LoRA models with individual intensity control (0-1)
- **FLUX Canny**: Specialized Canny edge ControlNet with configurable thresholds
- **FLUX Tools**: Kontext for image editing and transformation
- **Background Removal**: AI-powered background removal with rembg
- **ControlNet Upscaler**: High-quality image upscaling using JasperAI's FLUX ControlNet

### 🛠️ Dynamic Tool System
- **Modular Design**: Select and combine different AI tools
- **Individual Tool Control**: Each tool has its own parameters and settings
- **Real-time Configuration**: Adjust parameters for each selected tool
- **Visual Tool Management**: Clear interface showing selected tools and their settings

### 🧠 AI-Powered Prompt Enhancement
- **Ollama Integration**: Local LLM integration for prompt optimization
- **Vision Model Support**: Analyze images to generate detailed prompts
- **Streaming Responses**: Real-time prompt generation with progressive updates

### 📊 History & Management
- **SQLite Database**: Persistent storage of all image operations
- **Complete Metadata**: Tracks generation parameters, tools used, and settings
- **Gallery View**: Visual browsing of generated images with detailed information
- **HuggingFace Cache Management**: Integrated cache viewer and cleanup with selective deletion
- **Comprehensive Logging**: Detailed console output for debugging and monitoring

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU or Apple Silicon Mac (MPS support)
- [Ollama](https://ollama.ai/) (optional, for prompt enhancement)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/VincentGourbin/FluxForge-Studio.git
cd FluxForge-Studio
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: The requirements include `diffusers` from the main development branch to support FLUX Kontext functionality. This ensures compatibility with the latest FLUX model features.

3. (Optional) Install and configure Ollama:
```bash
# Install Ollama from https://ollama.ai/
ollama pull llama3.2-vision
ollama pull llama3.2
```

## 🚀 Usage

### Starting the Application

#### Local Access
```bash
python main.py
```
Access the web interface at `http://localhost:7860`

#### Public Sharing
```bash
python main.py --share
# or
python main.py -s
```

When using `--share`, the application will:
- Generate a public shareable link (valid for 1 week)
- Create random authentication credentials for security
- Display the username/password in the terminal

**Security Note**: Save the generated credentials immediately as they won't be shown again!

### Dynamic Tool System

#### Adding Tools
1. Click **"Sélectionner des outils"** to open the tool selection modal
2. Choose a tool type: LoRA, ControlNet, or FLUX Tools
3. Select the specific tool from the dropdown
4. Click **"Ajouter l'outil"** to add it to your selection

#### Tool Types

**LoRA Models**
- Intensity control (0-1 range)
- Automatic activation keyword integration
- Multiple LoRA support (up to 3 simultaneously)

**FLUX Postprocessing Tools**
- **FLUX Canny**: Canny edge detection with configurable thresholds
- **FLUX Depth**: Depth-based image generation and control
- **FLUX Fill**: Inpainting and outpainting capabilities
- **FLUX Redux**: Image-to-image with Redux model
- **Kontext**: Text-guided image editing

**FLUX Tools**
- Kontext for image editing
- Guidance scale control
- Input image processing

#### Managing Tools
- **Individual Removal**: Use 🗑️ buttons to remove specific tools
- **Clear All**: Remove all selected tools at once
- **Parameter Adjustment**: Modify settings for each tool independently

### Image Generation Process
1. Enter your prompt
2. Select model (schnell/dev) and configure basic parameters
3. Add desired tools using the dynamic tool system
4. Adjust tool-specific parameters
5. Generate your image

## 📁 Project Structure

```
mflux-gradio/
├── main.py                     # Main application entry point
├── lora_info.json             # LoRA model metadata
├── requirements.txt           # Python dependencies
├── src/                       # Source code modules
│   ├── core/                  # Core functionality
│   │   ├── config.py          # Configuration and device setup
│   │   └── database.py        # SQLite database operations
│   ├── generator/             # Image generation
│   │   └── image_generator.py # FLUX.1 generation with diffusers
│   ├── enhancement/           # AI enhancement tools
│   │   └── prompt_enhancer.py # Ollama prompt enhancement
│   ├── postprocessing/        # FLUX postprocessing tools
│   │   ├── background_remover.py # Background removal
│   │   ├── flux_canny.py      # Canny edge ControlNet
│   │   ├── flux_depth.py      # Depth ControlNet
│   │   ├── flux_fill.py       # Inpainting/outpainting
│   │   ├── flux_redux.py      # Image-to-image with Redux
│   │   ├── kontext.py         # Text-guided image editing
│   │   └── upscaler.py        # ControlNet upscaling
│   ├── ui/                    # User interface components
│   │   ├── components.py      # Gradio UI components
│   │   └── lora_manager.py    # LoRA management interface
│   └── utils/                 # Utility functions
│       ├── canny_processing.py   # Canny edge detection
│       ├── hf_cache_manager.py   # HuggingFace cache management
│       ├── image_processing.py   # Image utilities
│       ├── mask_utils.py         # Masking utilities
│       ├── model_cache.py        # Model caching system
│       └── quantization.py       # Memory quantization
├── docs/                      # Documentation
│   ├── API.md                # API documentation
│   ├── FEATURES.md           # Detailed features guide
│   ├── QUANTIZATION.md       # Quantization guide
│   └── SETUP.md              # Setup instructions
├── lora/                     # LoRA model files (.safetensors)
├── outputimage/              # Generated images and metadata
└── temp_images/              # Temporary file storage
```

## ⚙️ Configuration

### LoRA Configuration
Edit `lora_info.json` to add new LoRA models:
```json
{
    "file_name": "your_lora.safetensors",
    "description": "Description of your LoRA",
    "activation_keyword": "trigger_word"
}
```

### Device Configuration
The application automatically detects and uses:
- **MPS** (Apple Silicon)
- **CUDA** (NVIDIA GPUs)
- **CPU** (fallback)

### Model Configuration
- FLUX.1 models loaded via diffusers library
- ControlNet models loaded on-demand
- Intelligent model caching for performance

## 🔧 Advanced Features

### FLUX Canny
- **Configurable Thresholds**: Adjust low/high thresholds for edge detection
- **Real-time Preview**: See Canny edges before generation
- **Strength Control**: Fine-tune ControlNet influence (0-1)
- **Specialized Pipeline**: Dedicated FLUX Canny implementation

### ControlNet Upscaler
- **JasperAI Model**: Uses FLUX.1-dev-Controlnet-Upscaler
- **High Quality**: Better than simple image resizing
- **Automatic Parameters**: Optimized settings for best results

### HuggingFace Cache Management
- **Visual Cache Browser**: View all cached models with detailed information
- **Selective Deletion**: Choose specific models to remove with checkboxes
- **Space Calculation**: See freed space before deletion
- **Real-time Updates**: Automatic refresh after cleanup operations

### Memory Management & Quantization
- **8-bit Quantization**: Reduce GPU memory usage by ~70% using optimum.quanto
- **Cross-Platform Support**: Works on MPS (Apple Silicon), CUDA, and CPU
- **Automatic Cleanup**: GPU/MPS memory cleaned after operations
- **Model Caching**: Prevents unnecessary model reloads
- **Device Optimization**: Adapts to available hardware

#### Quantization Features
- **Supported Models**: FLUX Schnell, Dev, and all postprocessing tools (Fill, Kontext, Depth, Canny, Redux)
- **Memory Reduction**: Up to 70% GPU memory savings with 8-bit quantization
- **Performance**: Tested on MPS devices with minimal quality loss
- **LoRA Compatibility**: Quantization applied AFTER LoRA loading for full compatibility
- **Stability**: Conservative implementation using only tested qint8 quantization
- **Fallback**: Graceful degradation if quantization fails

### Logging System
- **Detailed Parameters**: All generation settings logged
- **Tool Information**: Complete tool configuration tracking
- **Clean Output**: Essential information without debug clutter

## 🐛 Troubleshooting

### Common Issues

**Memory Issues**
- Reduce image dimensions
- Use CPU for ControlNet operations if MPS fails
- Close other GPU-intensive applications

**Tool Selection Issues**
- Ensure LoRA files are in `lora/` directory
- Check `lora_info.json` format
- Verify input images are valid for postprocessing tools
- Check that required models are downloaded for specific tools

**Generation Failures**
- Check console logs for detailed error information
- Verify model files are accessible
- Ensure sufficient disk space

### Performance Tips
- **Model Caching**: Avoid changing models frequently
- **Tool Selection**: Only select tools you need
- **Image Dimensions**: Balance quality vs. speed
- **Batch Operations**: Process multiple images efficiently

## 🎯 Key Updates

### Recent Improvements
- **Diffusers Migration**: Full migration from mflux to diffusers library
- **Dynamic Tool System**: Modular tool selection and management
- **Fixed LoRA Mapping**: Corrected tool-to-parameter mapping issues
- **Enhanced Logging**: Comprehensive parameter tracking
- **Memory Optimization**: Better GPU/MPS memory management
- **UI Improvements**: Cleaner interface with better contrast
- **ControlNet Upscaler**: Proper implementation with JasperAI model

## 📋 TODO

### Planned Features & Improvements

- [x] **Support quantisation** - Add 4-bit/8-bit model quantization for memory efficiency (8-bit implemented, to be tested on CUDA)
- [ ] **Remove Ollama dependencies** - Make prompt enhancement optional with fallback options
- [ ] **Add interface to manage LoRA** - GUI for installing, organizing, and managing LoRA models
- [ ] **Add custom model support** - Support for user-provided custom models and fine-tunes
- [ ] **Add memory optimisation of diffusers** - Implement advanced memory management techniques
- [ ] **Add batch image generation** - Queue system for generating multiple images with different prompts/parameters

### Priority
- **High**: Quantization support and memory optimization
- **Medium**: LoRA management interface, custom models, and batch generation
- **Low**: Optional Ollama dependencies

Contributions welcome! Feel free to open issues for feature requests or bug reports.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [FLUX.1](https://github.com/black-forest-labs/flux) by Black Forest Labs
- [Diffusers](https://github.com/huggingface/diffusers) by Hugging Face
- [Gradio](https://gradio.app/) - Web interface framework
- [Ollama](https://ollama.ai/) - Local LLM integration
- [JasperAI](https://huggingface.co/jasperai) - ControlNet Upscaler model

---

**Performance Note**: This application requires significant computational resources. GPU acceleration (CUDA/MPS) is highly recommended for optimal performance.