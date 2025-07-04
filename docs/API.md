# MFLUX-Gradio API Documentation

This document provides technical documentation for the MFLUX-Gradio API and internal functions.

## 📚 Module Overview

### Core Modules

#### `main.py`
Main Gradio application with web interface.

#### `image_generator.py` 
Core image generation logic with model management.

#### `database.py`
SQLite database operations for history management.

#### `config.py`
Configuration management and device setup.

#### `train.py`
Standalone LoRA training functionality.

#### `prompt_enhancer.py`
Ollama integration for prompt enhancement.

#### `background_remover.py`
AI-powered background removal using RMBG-2.0.

## 🏗️ Core Classes

### ImageGenerator

Main class for handling FLUX.1 image generation with advanced caching and LoRA support.

```python
class ImageGenerator:
    def __init__(self):
        """Initialize the ImageGenerator with model caching capabilities."""
```

#### Attributes
- `current_model_alias`: Currently loaded model name
- `current_quantize`: Active quantization level
- `current_path`: Path to current model
- `current_lora_paths`: List of loaded LoRA paths
- `current_lora_scales`: LoRA scale values
- `flux_model`: Cached model instance
- `lora_data`: LoRA configuration data

#### Methods

##### `generate_image()`
```python
def generate_image(
    self,
    prompt: str,
    model_alias: str,
    quantize: str,
    steps: int,
    seed: int,
    metadata: bool,
    guidance: float,
    height: int,
    width: int,
    path: str,
    controlnet_image_path: Optional[str],
    controlnet_strength: float,
    controlnet_save_canny: bool,
    progress=gr.Progress(),
    *args
) -> PIL.Image:
```

**Parameters:**
- `prompt`: Text description for image generation
- `model_alias`: FLUX.1 model variant ("schnell" or "dev")
- `quantize`: Quantization level ("4", "8", or None)
- `steps`: Number of inference steps
- `seed`: Random seed (0 for random)
- `metadata`: Whether to export JSON metadata
- `guidance`: CFG guidance scale (dev model only)
- `height`: Image height in pixels
- `width`: Image width in pixels
- `path`: Custom model path (optional)
- `controlnet_image_path`: ControlNet reference image
- `controlnet_strength`: ControlNet influence (0.0-1.0)
- `controlnet_save_canny`: Save Canny edge visualization
- `*args`: LoRA selection and scale values

**Returns:**
- `PIL.Image`: Generated image

**Raises:**
- Model loading errors
- CUDA out of memory errors
- Invalid parameter errors

##### `update_guidance_visibility()`
```python
def update_guidance_visibility(self, model_alias: str) -> gr.update:
```

Controls visibility of guidance scale parameter based on model type.

## 📊 Database Functions

### `init_db()`
```python
def init_db() -> None:
```
Initialize SQLite database with required table structure.

### `save_image_info()`
```python
def save_image_info(details: tuple) -> None:
```
Save image generation metadata to database.

**Parameters:**
- `details`: Tuple containing all generation parameters

### `load_history()`
```python
def load_history() -> List[str]:
```
Load existing image history, filtering for existing files.

**Returns:**
- List of valid image file paths

### `get_image_details()`
```python
def get_image_details(index: int) -> str:
```
Retrieve detailed information for a specific image.

**Parameters:**
- `index`: Image index in history (0-based)

**Returns:**
- Formatted string with all generation parameters

### `delete_image()`
```python
def delete_image(selected_image_index: int) -> Tuple[bool, str]:
```
Delete image file and database record.

**Parameters:**
- `selected_image_index`: Index of image to delete

**Returns:**
- Tuple of (success: bool, message: str)

## 🎨 Background Removal Functions

### `load_background_removal_model()`
```python
def load_background_removal_model() -> AutoModelForImageSegmentation:
```
Load and initialize RMBG-2.0 model for background removal.

**Returns:**
- Configured segmentation model

### `remove_background()`
```python
def remove_background(
    input_image: PIL.Image, 
    modelbgrm: AutoModelForImageSegmentation
) -> PIL.Image:
```
Remove background from image using AI segmentation.

**Parameters:**
- `input_image`: Source image
- `modelbgrm`: Loaded segmentation model

**Returns:**
- Image with transparent background

## 🧠 Prompt Enhancement Functions

### `enhance_prompt()`
```python
def enhance_prompt(
    selected_model: str, 
    input_text: str, 
    input_image: Optional[str]
) -> Iterator[str]:
```
Enhance prompts using Ollama models with streaming responses.

**Parameters:**
- `selected_model`: Ollama model name
- `input_text`: Text to enhance
- `input_image`: Image for vision models (optional)

**Yields:**
- Progressive enhancement text

### `update_image_input_visibility()`
```python
def update_image_input_visibility(selected_model: str) -> gr.update:
```
Control image input visibility based on model capabilities.

### `update_button_label()`
```python
def update_button_label(selected_model: str) -> gr.update:
```
Update button text based on model type (vision vs text-only).

## 🎓 Training Functions

### `prepare_training_json_and_start()`
```python
def prepare_training_json_and_start(
    files: List[str],
    descriptions: List[str],
    seed: int,
    steps: int,
    guidance: float,
    quantize: str,
    width: int,
    height: int,
    num_epochs: int,
    batch_size: int,
    plot_frequency: int,
    generate_image_frequency: int,
    validation_prompt: str,
    pdf_output: gr.Files,
    zip_output: gr.Files
) -> str:
```
Orchestrate complete LoRA training process.

**Parameters:**
- `files`: Training image files
- `descriptions`: Image descriptions
- `seed`: Training seed
- `steps`: Inference steps
- `guidance`: Guidance scale
- `quantize`: Quantization option
- `width/height`: Image dimensions
- `num_epochs`: Training epochs
- `batch_size`: Training batch size
- `plot_frequency`: Plot generation frequency
- `generate_image_frequency`: Validation frequency
- `validation_prompt`: Validation text
- `pdf_output/zip_output`: Output file components

**Returns:**
- Training status message

### Training Configuration Structure
```json
{
    "model": "dev",
    "seed": 42,
    "quantize": null,
    "steps": 20,
    "guidance": 3.0,
    "width": 512,
    "height": 512,
    "training_loop": {
        "num_epochs": 100,
        "batch_size": 1
    },
    "optimizer": {
        "name": "AdamW",
        "learning_rate": 1e-4
    },
    "save": {
        "checkpoint_frequency": 10,
        "output_path": "./temp_train/current_train"
    },
    "instrumentation": {
        "plot_frequency": 1,
        "generate_image_frequency": 20,
        "validation_prompt": "photo of sks dog"
    },
    "lora_layers": {
        "single_transformer_blocks": {
            "block_range": {
                "start": 0,
                "end": 38
            },
            "layer_types": [
                "proj_out",
                "proj_mlp", 
                "attn.to_q",
                "attn.to_k",
                "attn.to_v"
            ],
            "lora_rank": 4
        }
    },
    "examples": {
        "path": "../../temp_images",
        "images": [
            {
                "image": "filename.jpg",
                "prompt": "description"
            }
        ]
    }
}
```

## ⚙️ Configuration Management

### Device Configuration
```python
# Automatic device detection
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
```

### LoRA Configuration
LoRA metadata structure in `lora_info.json`:
```json
[
    {
        "file_name": "model.safetensors",
        "description": "Model description",
        "activation_keyword": "trigger_word"
    }
]
```

### Model Options
```python
model_options = ["schnell", "dev"]
quantize_options = [4, 8, None]
```

## 🔧 Utility Functions

### File Management
```python
# Output directory structure
output_dir = Path("outputimage")
output_filename = output_dir / f"{timestamp}_{seed}.png"
```

### Model Caching Logic
```python
# Cache validation
needs_reload = (
    model_alias != current_model_alias or
    quantize != current_quantize or
    path != current_path or
    lora_paths_list != current_lora_paths or
    lora_scales_list != current_lora_scales or
    flux_model is None or
    use_controlnet != (current_model_type == 'controlnet')
)
```

## 🚨 Error Handling

### Common Exceptions
- `torch.cuda.OutOfMemoryError`: VRAM exhaustion
- `FileNotFoundError`: Missing model or LoRA files
- `ValueError`: Invalid parameter values
- `RuntimeError`: Model loading failures

### Error Recovery
```python
try:
    # Model operations
    pass
except torch.cuda.OutOfMemoryError:
    # Suggest quantization
    pass
except Exception as e:
    # Log error and provide user feedback
    pass
```

## 🔄 Event Handling

### Gradio Events
```python
# Button click events
btn.click(
    fn=function_name,
    inputs=[input_components],
    outputs=[output_components],
    show_progress=True
)

# Change events
component.change(
    fn=update_function,
    inputs=[trigger_component],
    outputs=[target_component]
)

# Select events
gallery.select(
    fn=selection_handler,
    inputs=[state_component],
    outputs=[info_component, state_component]
)
```

## 📱 Interface Components

### Input Components
- `gr.Textbox`: Text input fields
- `gr.Number`: Numeric inputs
- `gr.Slider`: Range selectors
- `gr.Dropdown`: Selection menus
- `gr.Checkbox`: Boolean options
- `gr.File`: File uploads
- `gr.Image`: Image inputs

### Output Components
- `gr.Image`: Image display
- `gr.Gallery`: Image galleries
- `gr.Markdown`: Formatted text
- `gr.Files`: File downloads

### Layout Components
- `gr.Blocks`: Main container
- `gr.Tab`: Tab organization
- `gr.Row`: Horizontal layout
- `gr.Column`: Vertical layout
- `gr.Accordion`: Collapsible sections
- `gr.Group`: Logical grouping

## 🔐 Security Considerations

### File Upload Validation
- File type restrictions
- Size limitations
- Path traversal prevention

### Model Loading Safety
- Trusted model sources only
- Sandboxed execution environment
- Resource usage monitoring

### Database Security
- Parameterized queries
- Input sanitization
- Connection management

## 🚀 Performance Optimization

### Model Caching
- Intelligent cache invalidation
- Memory-efficient loading
- GPU memory management

### Database Optimization
- Indexed queries
- Connection pooling
- Efficient data structures

### Resource Management
- Automatic cleanup
- Memory monitoring
- Thread safety