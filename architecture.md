# FluxForge Studio Architecture Documentation

## Overview

FluxForge Studio is a professional AI image generation platform built with a modular architecture and clear separation of concerns. The application uses the `src/` directory pattern with specialized subdirectories for different functional areas, providing a scalable and maintainable codebase.

## Project Structure

```
fluxforge-studio/
├── main.py                    # Main application entry point (FluxForge Studio)
├── src/                       # Main source code directory
│   ├── __init__.py           # Package initialization and metadata
│   ├── core/                 # Core system components
│   │   ├── config.py         # Configuration and device management
│   │   └── database.py       # Image history and metadata management
│   ├── generator/            # Image generation engine
│   │   └── image_generator.py # FLUX.1 model management
│   ├── postprocessing/       # Post-processing tools and filters
│   │   ├── flux_fill.py      # Inpainting/Outpainting
│   │   ├── flux_canny.py     # Canny edge-guided generation
│   │   ├── flux_depth.py     # Depth-guided generation  
│   │   ├── flux_redux.py     # Image variation/refinement
│   │   ├── kontext.py        # Text-based image editing
│   │   ├── background_remover.py # Background removal
│   │   └── upscaler.py       # Image upscaling
│   ├── enhancement/          # Prompt and image enhancement
│   │   └── prompt_enhancer.py # Ollama integration
│   ├── ui/                   # User interface components
│   │   ├── components.py     # Reusable UI components
│   │   └── lora_manager.py   # LoRA management interface
│   └── utils/                # Utility functions and helpers
│       ├── image_processing.py # Image manipulation utilities
│       ├── mask_utils.py     # Mask creation and processing
│       ├── canny_processing.py # Canny edge detection
│       └── model_cache.py    # Model caching and management
├── outputimage/              # Generated images output directory
├── lora/                     # LoRA model files storage
├── temp_images/              # Temporary image processing
├── temp_train/               # Training temporary files
└── generated_images.db       # SQLite database for image history
```

## Module Documentation

### 📁 `src/core/` - Core System Components

**Purpose**: Fundamental application components that other modules depend on.

#### Files:
- **`config.py`**: Central configuration management
  - Device detection (MPS/CUDA/CPU)
  - Model paths and settings
  - LoRA data loading and management
  - Application constants and defaults

- **`database.py`**: Database operations and image history
  - SQLite database initialization and schema
  - Image metadata storage and retrieval
  - Gallery synchronization with filesystem
  - History management functions

#### Key Features:
- Automatic device detection for Apple Silicon (MPS), NVIDIA (CUDA), or CPU
- LoRA model discovery and metadata loading from `lora_info.json`
- Comprehensive image history with generation parameters
- Thread-safe database operations

---

### 📁 `src/generator/` - Image Generation Engine

**Purpose**: Core FLUX.1 model management and image generation pipeline.

#### Files:
- **`image_generator.py`**: Main image generation class
  - FLUX.1 model loading and caching (schnell, dev variants)
  - LoRA integration and dynamic loading
  - ControlNet support with Canny edge detection
  - Memory management and optimization
  - Quantization support (4-bit, 8-bit)

#### Key Features:
- Model caching to prevent unnecessary reloads
- Dynamic LoRA loading with configurable intensities
- ControlNet integration for guided generation
- Automatic memory cleanup and garbage collection
- Support for multiple FLUX.1 model variants

---

### 📁 `src/postprocessing/` - Post-Processing Tools

**Purpose**: Advanced image manipulation and editing tools.

#### Files:
- **`flux_fill.py`**: FLUX.1-Fill-dev integration
  - Inpainting with mask drawing interface
  - Outpainting with percentage-based expansion
  - Automatic mask generation and preview
  - LoRA support for Fill operations

- **`flux_canny.py`**: FLUX.1-Canny-dev LoRA integration
  - Real-time Canny edge detection with adjustable thresholds
  - Edge-guided image generation with precise control
  - LoRA support for style enhancement
  - Debug-free production-ready implementation

- **`flux_depth.py`**: FLUX.1-Depth-dev LoRA integration
  - Automatic depth map generation using Depth Anything model
  - Depth-guided image generation with structure preservation
  - LoRA support for artistic style application
  - Memory-efficient processing with device optimization

- **`flux_redux.py`**: FLUX.1-Redux-dev integration
  - Image variation and refinement capabilities
  - FluxPriorReduxPipeline for advanced image processing
  - Configurable variation strength and guidance
  - No text prompts needed - pure image-to-image variation

- **`kontext.py`**: Text-based image editing
  - FLUX.1-Kontext-dev model integration
  - Natural language image transformations
  - LoRA support for style modifications

- **`background_remover.py`**: Background removal functionality
  - Specialized model loading for background removal
  - High-quality background extraction
  - Transparent output support

- **`upscaler.py`**: Image resolution enhancement
  - Multiple upscaling algorithms
  - Configurable scaling factors
  - Quality preservation during upscaling

#### Key Features:
- Complete FLUX.1 model suite integration (Fill, Depth, Canny, Redux, Kontext)
- Advanced post-processing pipeline with real-time previews
- LoRA integration across all tools for style enhancement
- Memory-efficient processing with automatic cleanup
- Production-ready implementations without debug clutter

---

### 📁 `src/enhancement/` - Enhancement Tools

**Purpose**: AI-powered content improvement and optimization.

#### Files:
- **`prompt_enhancer.py`**: Ollama integration for prompt enhancement
  - Multi-model support (text and vision models)
  - Context-aware prompt optimization
  - Image analysis for vision models
  - Dynamic model selection interface

#### Key Features:
- Integration with locally running Ollama instance
- Support for both text-only and vision models
- Intelligent prompt expansion and improvement
- Error handling for offline scenarios

---

### 📁 `src/ui/` - User Interface Components

**Purpose**: Reusable Gradio interface components and layouts.

#### Files:
- **`components.py`**: Standard UI component factory
  - Generation parameter controls (steps, guidance, etc.)
  - Image dimension controls with validation
  - Standardized buttons and inputs
  - Tool selection modals and interfaces

- **`lora_manager.py`**: Comprehensive LoRA management interface
  - Dynamic LoRA selection with modal interface
  - HTML display with descriptions and keywords
  - Individual intensity sliders (0.0-1.0 range)
  - Add/remove functionality with real-time updates
  - Unified interface across all tools

#### Key Features:
- Consistent design patterns across the application
- Reusable components reduce code duplication
- Standardized parameter ranges and validation
- Advanced LoRA management with visual feedback

---

### 📁 `src/utils/` - Utility Functions

**Purpose**: Common operations and helper functions used across modules.

#### Files:
- **`mask_utils.py`**: Mask creation and manipulation
  - Automatic mask extraction from ImageEditor data
  - Outpainting mask generation with expansion percentages
  - Mask validation and processing utilities
  - Image difference detection for mask creation

- **`canny_processing.py`**: Canny edge detection utilities
  - Configurable threshold edge detection
  - RGB output compatible with FLUX models
  - Preview generation for real-time feedback
  - Optimized for FLUX.1-Canny-dev LoRA

- **`image_processing.py`**: General image manipulation
  - Format conversions and validation
  - Memory management helpers
  - File I/O utilities with error handling
  - Common image transformations

- **`model_cache.py`**: Model caching and management
  - HuggingFace model cache monitoring
  - Cache status reporting for all FLUX models
  - Pre-download utilities for model optimization
  - Storage and bandwidth optimization tools

#### Key Features:
- Optimized algorithms for common image operations
- Robust error handling and validation
- Memory-efficient processing
- Cross-platform compatibility

---

## Dependencies and Relationships

### Dependency Flow:
```
main.py
├── src/core/               (foundational)
├── src/generator/          (depends on core)
├── src/postprocessing/     (depends on core, generator, utils)
├── src/enhancement/        (depends on core)
├── src/ui/                 (depends on core, provides interfaces)
└── src/utils/              (utility functions, minimal dependencies)
```

### Key Relationships:
1. **Core modules** provide foundation for all other components
2. **Generator** is used by main interface and post-processing tools
3. **UI components** are used throughout to maintain consistent interfaces
4. **Utils** provide common functionality without creating circular dependencies

## Usage Patterns

### Starting the Application:
```bash
python main.py
```

### Adding New Post-Processing Tools:
1. Create new module in `src/postprocessing/`
2. Implement processing function with standard signature
3. Add UI components using `src/ui/` modules
4. Update main.py to include new tool in interface

### Adding New LoRA Support:
1. LoRA management is centralized in `src/ui/lora_manager.py`
2. Use `create_lora_manager_interface()` for consistent interface
3. Set up events with `setup_lora_events()`
4. Processing functions should accept LoRA state parameters

### Extending UI Components:
1. Add new components to `src/ui/components.py`
2. Follow existing naming conventions (`create_*_component`)
3. Include proper documentation and parameter validation
4. Test across different tools for consistency

## Configuration Management

### LoRA Configuration:
- LoRA models stored in `lora/` directory
- Metadata managed through `lora_info.json`
- Dynamic discovery and loading through `src/core/config.py`

### Model Configuration:
- FLUX.1 models auto-downloaded or specify local paths
- Device selection: MPS > GPU > CPU fallback
- Quantization options: 4-bit, 8-bit, or no quantization

### Database Configuration:
- SQLite database: `generated_images.db`
- Automatic schema creation and migration
- Image metadata includes all generation parameters

## Development Guidelines

### Code Organization:
- Keep modules focused on single responsibilities
- Use clear, descriptive function and variable names
- Include comprehensive docstrings for all functions
- Follow consistent error handling patterns

### Adding New Features:
1. Determine appropriate module location based on functionality
2. Create unit tests for core functionality
3. Update documentation when adding new capabilities
4. Ensure memory cleanup in processing functions

### Performance Considerations:
- Use model caching to prevent unnecessary reloads
- Implement proper memory cleanup after GPU operations
- Utilize lazy loading for optional dependencies
- Monitor memory usage in resource-intensive operations

## FluxForge Studio Features

The professional platform provides comprehensive FLUX.1 model integration:

### Core Capabilities:
- **Complete FLUX.1 Suite**: Support for dev, schnell, Fill, Depth, Canny, Redux, and Kontext models
- **Advanced Post-Processing**: Real-time previews and professional-grade image manipulation
- **LoRA Integration**: Dynamic loading and management across all tools
- **Memory Optimization**: Efficient processing with automatic cleanup
- **Professional UI**: Clean, intuitive interface designed for content creators

### Architecture Benefits:
- **Maintainability**: Clear separation makes code easier to understand and modify
- **Reusability**: UI components and utilities can be shared across features
- **Testing**: Individual modules can be tested independently
- **Scalability**: New features can be added without affecting existing functionality
- **Organization**: Related functionality is grouped logically
- **Production Ready**: Debug-free, optimized code for professional use

### Platform Compatibility:
- **Cross-Platform**: Runs on Apple Silicon (MPS), NVIDIA (CUDA), and CPU
- **Model Flexibility**: Supports HuggingFace models and local file paths
- **Data Preservation**: Compatible with existing LoRA files and image history
- **Database Migration**: Automatic schema updates with data preservation

## Troubleshooting

### Common Issues:
1. **Import Errors**: Ensure `src/` directory is in Python path
2. **Model Loading**: Check device compatibility and available memory
3. **LoRA Issues**: Verify `lora_info.json` format and file paths
4. **Database Errors**: Check write permissions for `generated_images.db`

### Debug Mode:
- Use `python -v main.py` for verbose import information
- Check console output for detailed error messages
- Monitor memory usage during generation operations

---

## Conclusion

FluxForge Studio represents a professional-grade AI image generation platform built on a solid modular architecture. The clear separation of concerns, comprehensive FLUX.1 model integration, and production-ready codebase make it an ideal foundation for content creators and developers alike.

The platform successfully combines power with usability, offering advanced features like multiple FLUX.1 variants, real-time previews, LoRA management, and professional post-processing tools, all wrapped in an intuitive interface.

For specific implementation details, refer to the comprehensive docstrings within each module and the example usage patterns in `main.py`.