# MFLUX-Gradio Features Documentation

This document provides detailed information about all features available in MFLUX-Gradio.

## 📑 Table of Contents
- [Image Generation](#image-generation)
- [LoRA Support](#lora-support)
- [ControlNet Integration](#controlnet-integration)
- [Prompt Enhancement](#prompt-enhancement)
- [Background Removal](#background-removal)
- [Training Pipeline](#training-pipeline)
- [History Management](#history-management)

## 🎨 Image Generation

### FLUX.1 Model Support
MFLUX-Gradio supports both FLUX.1 model variants:

#### **Schnell Model**
- **Purpose**: Fast generation optimized for speed
- **Steps**: Typically 1-4 steps
- **Quality**: Good quality with faster generation times
- **Use Case**: Quick prototyping, batch generation, real-time applications

#### **Dev Model** 
- **Purpose**: High-quality generation with more control
- **Steps**: Typically 20-50 steps
- **Quality**: Superior quality with more detail
- **Use Case**: Final artwork, detailed compositions, professional use
- **Additional Features**: 
  - Guidance scale control (adjustable CFG)
  - Better prompt adherence
  - Enhanced detail generation

### Quantization Options

#### **No Quantization (Full Precision)**
- **Memory**: ~24GB VRAM
- **Quality**: Maximum quality
- **Speed**: Fastest inference (with sufficient VRAM)
- **Recommended**: High-end GPUs (RTX 4090, A100, etc.)

#### **8-bit Quantization**
- **Memory**: ~12GB VRAM
- **Quality**: Minimal quality loss
- **Speed**: Slightly slower than full precision
- **Recommended**: Mid-range to high-end GPUs (RTX 3080, 4080, etc.)

#### **4-bit Quantization**
- **Memory**: ~6GB VRAM
- **Quality**: Noticeable but acceptable quality reduction
- **Speed**: Slower inference
- **Recommended**: Lower-end GPUs, Apple Silicon Macs

### Image Dimensions
- **Supported Range**: 256px to 4096px (64px increments)
- **Optimal Sizes**: 1024x1024, 1152x896, 896x1152
- **Performance**: Larger images require more VRAM and processing time
- **Aspect Ratios**: Any ratio supported, but square and 4:3/3:4 work best

## 🔧 LoRA Support

### What are LoRAs?
LoRA (Low-Rank Adaptation) models are lightweight fine-tuned models that add specific styles, concepts, or characters to image generation without requiring full model retraining.

### Configuration
LoRA information is now stored in the SQLite database and managed through the **LoRA Management** tab in the interface. Legacy `lora_info.json` files are automatically migrated on first run.

Database schema:
- **file_name**: LoRA model filename (e.g., "example_lora.safetensors")
- **description**: Human-readable description of the style/effect
- **activation_keyword**: Trigger word automatically added to prompts

### Features
- **Multiple LoRA Support**: Load multiple LoRAs simultaneously
- **Scale Control**: Adjust influence of each LoRA (0.0-1.0)
- **Automatic Keyword Insertion**: Activation keywords automatically added to prompts
- **Dynamic Loading**: LoRAs loaded on-demand based on selection
- **Parameter Isolation**: Queue tasks capture independent LoRA snapshots, preventing UI state interference
- **Database Management**: Complete GUI for LoRA upload, editing, and organization
- **Model Caching**: Intelligent caching prevents unnecessary reloads

### Usage Tips
1. **Start with Scale 1.0**: Begin with full strength and adjust down if needed
2. **Keyword Placement**: Keywords are automatically prepended to your prompt
3. **Multiple LoRAs**: When using multiple LoRAs, reduce individual scales (0.6-0.8)
4. **Style Consistency**: Use similar style LoRAs together for best results

## 🎯 ControlNet Integration

### Canny Edge Detection
Currently supports Canny edge detection for controlled image generation.

### Features
- **Edge Detection**: Automatically extracts edges from reference images
- **Strength Control**: Adjust how closely the generated image follows the control image (0.0-1.0)
- **Visualization**: Option to save the processed Canny edge image
- **Model Integration**: Seamlessly works with both FLUX.1 variants and LoRAs

### Workflow
1. **Upload Reference Image**: Provide an image for edge detection
2. **Adjust Strength**: 
   - 0.1-0.3: Subtle guidance
   - 0.4-0.6: Moderate control
   - 0.7-1.0: Strong adherence to edges
3. **Generate**: The model uses edge information to guide generation

### Use Cases
- **Composition Control**: Maintain specific layouts or structures
- **Pose Transfer**: Transfer poses from reference images
- **Architectural Planning**: Control building and structure layouts
- **Character Positioning**: Precise character placement and poses

## 🧠 Prompt Enhancement

### Ollama Integration
Leverages local LLM models through Ollama for intelligent prompt enhancement.

### Supported Model Types

#### **Text-Only Models**
- **Function**: Enhance and expand text prompts
- **Examples**: Llama 3.2, Mistral, CodeLlama
- **Output**: Detailed, optimized prompts for FLUX.1

#### **Vision Models**
- **Function**: Analyze images and generate descriptive prompts
- **Examples**: Llama 3.2-Vision, LLaVA
- **Input**: Images + optional text descriptions
- **Output**: Comprehensive image analysis and FLUX-optimized prompts

### Features
- **Streaming Responses**: Real-time prompt generation
- **FLUX Optimization**: Prompts specifically tailored for FLUX.1 models
- **Style Analysis**: Detailed breakdown of artistic styles and techniques
- **Multi-language Support**: Automatic translation to English for optimal results

### Enhancement Process
1. **Style Analysis**: Identifies artistic styles, lighting, composition
2. **Detail Enhancement**: Adds specific, descriptive elements
3. **FLUX Optimization**: Structures prompt for optimal FLUX.1 results
4. **Quality Assurance**: Ensures prompt clarity and effectiveness

## 🖼️ Background Removal

### RMBG-2.0 Integration
Uses the state-of-the-art RMBG-2.0 model for high-quality background removal.

### Features
- **AI-Powered**: Advanced segmentation model
- **High Quality**: Preserves fine details and edge quality
- **Transparent Output**: Results include proper alpha channel
- **Batch Processing**: Process multiple images efficiently

### Technical Details
- **Model**: RMBG-2.0 from Bria AI
- **Input Size**: Optimized for 1024x1024 processing
- **Output Format**: PNG with transparency
- **Device Support**: GPU acceleration when available

### Use Cases
- **Product Photography**: Remove backgrounds for clean product shots
- **Portrait Processing**: Create professional headshots
- **Composite Creation**: Prepare elements for complex compositions
- **Web Assets**: Generate clean images for websites and presentations

## 🎓 Training Pipeline

### DreamBooth Training
Complete LoRA training pipeline using DreamBooth methodology.

### Features
- **Web Interface**: User-friendly training setup
- **Real-time Monitoring**: Live tracking of training progress
- **Flexible Configuration**: Customizable training parameters
- **Checkpoint Management**: Automatic saving and validation

### Training Parameters

#### **Basic Parameters**
- **Epochs**: Number of complete training cycles (default: 100)
- **Batch Size**: Training batch size (memory dependent, default: 1)
- **Learning Rate**: Training speed (default: 1e-4)
- **Steps**: Inference steps for generated images (default: 20)

#### **Advanced Parameters**
- **Plot Frequency**: How often to generate training plots (default: 1)
- **Generate Image Frequency**: Validation image generation frequency (default: 20)
- **Validation Prompt**: Prompt used for validation images
- **Quantization**: Memory optimization during training

### Training Workflow
1. **Data Preparation**: Upload images and add descriptions
2. **Parameter Configuration**: Set training parameters
3. **Training Execution**: Monitor progress with real-time feedback
4. **Validation**: Automatic validation image generation
5. **Checkpoint Export**: Download trained LoRA models

### Data Requirements
- **Images**: 5-20 high-quality images recommended
- **Descriptions**: Detailed, consistent descriptions for each image
- **Quality**: High resolution, good lighting, clear subjects
- **Variety**: Different poses, angles, lighting conditions

## 📊 History Management

### Database Integration
Comprehensive tracking of all generated images and their parameters.

### Features
- **Complete Metadata**: All generation parameters stored
- **Visual Gallery**: Grid-based image browsing
- **Search and Filter**: Find specific generations
- **Export Capability**: JSON metadata export for reproducibility

### Stored Information
- **Generation Parameters**: All FLUX.1 settings
- **LoRA Configuration**: Applied LoRAs and their scales
- **ControlNet Settings**: Control image and strength settings
- **Timestamps**: Complete generation history
- **File Paths**: Organized file management

### Gallery Features
- **Responsive Layout**: Adaptive grid based on screen size
- **Image Details**: Click to view full generation parameters
- **Batch Operations**: Delete multiple images
- **Metadata Export**: Export settings for reproduction

## 🔧 Configuration and Customization

### Environment Setup
- **MPS Support**: Optimized for Apple Silicon
- **CUDA Support**: NVIDIA GPU acceleration
- **CPU Fallback**: Runs on CPU when GPU unavailable
- **Memory Management**: Intelligent VRAM usage optimization

### File Organization
```
mflux-gradio/
├── lora/                    # LoRA model files
├── outputimage/             # Generated images
├── temp_images/             # Training image cache
├── temp_train/              # Training configurations
├── stepwise_output/         # Debug/intermediate outputs
└── generated_images.db      # History database
```

### Advanced Configuration
- **Model Paths**: Custom FLUX.1 model locations
- **LoRA Management**: Dynamic LoRA discovery and loading
- **Database Customization**: SQLite database configuration
- **Ollama Integration**: Local LLM model management

## 🔍 Troubleshooting and Optimization

### Performance Optimization
1. **Use Quantization**: Reduce memory usage with 8-bit/4-bit quantization
2. **Optimize Dimensions**: Use standard sizes (1024x1024) for best performance
3. **LoRA Management**: Avoid loading too many LoRAs simultaneously
4. **Model Caching**: Let the system cache models for subsequent generations

### Common Issues
1. **CUDA Out of Memory**: Enable quantization or reduce image dimensions
2. **Slow Generation**: Check quantization settings and available VRAM
3. **LoRA Not Loading**: Use the LoRA Management tab to verify database entries and file paths
4. **Training Failures**: Ensure sufficient disk space and proper image formats

### Best Practices
1. **Regular Cleanup**: Periodically clean up temporary files
2. **Backup Training Data**: Keep copies of important training datasets
3. **Monitor Resources**: Watch VRAM and disk space usage
4. **Update Models**: Keep LoRA and base models up to date