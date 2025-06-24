# MFLUX-Gradio: Advanced FLUX.1 Image Generation Interface

A comprehensive web-based interface for FLUX.1 image generation with integrated LoRA training, prompt enhancement, and background removal capabilities.

## 🚀 Features

### 🎨 Image Generation
- **FLUX.1 Model Support**: Both schnell and dev variants
- **LoRA Integration**: Dynamic loading of multiple LoRA models with configurable scales
- **ControlNet Support**: Canny edge detection for controlled generation
- **Quantization Options**: 4-bit, 8-bit, or no quantization for memory optimization
- **Model Caching**: Intelligent model caching to prevent unnecessary reloads

### 🧠 AI-Powered Prompt Enhancement
- **Ollama Integration**: Local LLM integration for prompt optimization
- **Vision Model Support**: Analyze images to generate detailed prompts
- **Multi-language Support**: Automatic translation and enhancement
- **Streaming Responses**: Real-time prompt generation with progressive updates

### 🎯 LoRA Training
- **DreamBooth Training**: Complete training pipeline for custom LoRA models
- **Real-time Monitoring**: Live tracking of training progress with plots
- **Flexible Configuration**: Customizable training parameters and validation
- **Checkpoint Management**: Automatic saving and management of training checkpoints

### 🖼️ Background Removal
- **AI-Powered**: Advanced background removal using specialized models
- **High Quality**: Preserves image quality while removing backgrounds
- **Easy Integration**: Simple drag-and-drop interface

### 📊 History & Management
- **SQLite Database**: Persistent storage of generation history and metadata
- **Detailed Metadata**: Complete parameter tracking for each generation
- **Gallery View**: Visual browsing of generated images
- **Export Support**: JSON metadata export for reproducibility

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended) or Apple Silicon Mac
- [Ollama](https://ollama.ai/) (for prompt enhancement features)

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd mflux-gradio
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install and configure Ollama for prompt enhancement:
```bash
# Install Ollama from https://ollama.ai/
# Pull your preferred models
ollama pull llama3.2-vision
ollama pull llama3.2
```

## 🚀 Usage

### Starting the Application
```bash
python mflux-gradio.py
```

The application will launch a web interface accessible at `http://localhost:7860`

### Basic Image Generation
1. Navigate to the **Generation** tab
2. Enter your prompt in the text field
3. Configure parameters (model, steps, dimensions, etc.)
4. (Optional) Select LoRA models and adjust their scales
5. Click **Generate Image**

### Using LoRA Models
1. Place your `.safetensors` LoRA files in the `lora/` directory
2. Update `lora_info.json` with model descriptions and activation keywords
3. Select desired LoRA models in the Generation tab
4. Adjust scales (0.0-1.0) for each selected LoRA

### Training Custom LoRA
1. Navigate to the **Train** tab
2. Upload training images (multiple files supported)
3. Add descriptions for each image
4. Configure training parameters
5. Click **Start Training**
6. Monitor progress through generated plots and checkpoints

### Prompt Enhancement
1. Navigate to the **Prompt Enhancer** tab
2. Select an Ollama model
3. Enter your basic prompt or upload an image (for vision models)
4. Click **Enhance Prompt** for AI-powered optimization

## 📁 Project Structure

```
mflux-gradio/
├── mflux-gradio.py          # Main Gradio application
├── image_generator.py       # FLUX.1 image generation logic
├── database.py             # SQLite database operations
├── config.py               # Configuration and settings
├── train.py                # LoRA training functionality
├── prompt_enhancer.py      # Ollama prompt enhancement
├── background_remover.py   # Background removal functionality
├── lora_info.json          # LoRA model metadata
├── lora/                   # LoRA model files (.safetensors)
├── outputimage/            # Generated images
├── temp_images/            # Temporary training images
├── temp_train/             # Training configurations
└── stepwise_output/        # Intermediate generation outputs
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

### Model Paths
- Local FLUX.1 models can be specified in the Generation tab
- LoRA files should be placed in the `lora/` directory
- Training outputs are saved in `temp_train/current_train/`

## 🔧 Advanced Features

### ControlNet Usage
1. Upload a reference image in the ControlNet section
2. Adjust strength (0.0-1.0) to control influence
3. Enable Canny edge visualization if desired
4. Generate with your prompt and ControlNet guidance

### Training Parameters
- **Epochs**: Number of training cycles
- **Batch Size**: Training batch size (memory dependent)
- **Learning Rate**: Training learning rate (default: 1e-4)
- **Validation Frequency**: How often to generate validation images

### Quantization Options
- **No Quantization**: Full precision (highest quality, most memory)
- **8-bit**: Balanced quality and memory usage
- **4-bit**: Lowest memory usage (slight quality reduction)

## 🐛 Troubleshooting

### Common Issues
1. **CUDA out of memory**: Try enabling quantization or reducing image dimensions
2. **Ollama connection issues**: Ensure Ollama is running and models are installed
3. **LoRA not loading**: Check file paths and `lora_info.json` format
4. **Training fails**: Verify sufficient disk space and image format compatibility

### Performance Optimization
- Use quantization for lower memory usage
- Enable model caching for faster subsequent generations
- Consider image dimensions vs. generation time trade-offs

## 📊 Database Schema

The SQLite database stores:
- Generation timestamps and parameters
- Prompt and model information
- LoRA configurations used
- File paths and metadata
- ControlNet settings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [FLUX.1](https://github.com/black-forest-labs/flux) by Black Forest Labs
- [mflux](https://github.com/filipstrand/mflux) - FLUX.1 implementation
- [Gradio](https://gradio.app/) - Web interface framework
- [Ollama](https://ollama.ai/) - Local LLM integration

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing issues in the repository
3. Create a new issue with detailed information

---

**Note**: This application requires significant computational resources. GPU acceleration is highly recommended for optimal performance.