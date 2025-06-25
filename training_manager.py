"""
Training Management Module for mflux-gradio application.

This module handles all LoRA training functionality including:
- Training file management and gallery updates
- Training parameter configuration and validation
- DreamBooth training execution with monitoring
- Real-time training output monitoring (plots and checkpoints)

Author: Vincent
License: MIT
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import gradio as gr
from mflux.dreambooth.dreambooth import DreamBooth
from mflux.dreambooth.dreambooth_initializer import DreamBoothInitializer
from mflux.error.exceptions import StopTrainingException

# Global constants
TEMP_IMAGES_DIR = "temp_images"
TEMP_TRAIN_DIR = "temp_train"

def update_gallery_and_descriptions(files) -> Tuple[List[str], List[str], str]:
    """
    Update the training gallery and initialize descriptions for uploaded files.
    
    This function processes uploaded training files and prepares them for display
    in the training interface gallery. It creates placeholder descriptions for
    each uploaded image.
    
    Args:
        files: List of uploaded file objects from Gradio file component
        
    Returns:
        Tuple[List[str], List[str], str]: A tuple containing:
            - images (List[str]): List of file paths for gallery display
            - descriptions (List[str]): List of empty description strings
            - status_message (str): Status message for user feedback
            
    Note:
        Returns empty lists and error message if no files are provided.
    """
    if not files:
        return [], [], "Aucun fichier chargé."
    
    images = [file.name for file in files]
    descriptions = [""] * len(images)  # Initialize empty descriptions
    return images, descriptions, ""


def update_description(index: str, new_description: str, descriptions: List[str]) -> List[str]:
    """
    Update the description for a specific image in the training set.
    
    This function modifies the description list by updating the description
    at the specified index with the new provided description.
    
    Args:
        index (str): String representation of the image index to update
        new_description (str): New description text for the selected image
        descriptions (List[str]): Current list of descriptions
        
    Returns:
        List[str]: Updated list of descriptions with the new description at the specified index
        
    Note:
        Converts string index to integer for list access.
    """
    descriptions[int(index)] = new_description
    return descriptions


def select_train_image(evt: gr.SelectData, descriptions: List[str]) -> Tuple[int, str]:
    """
    Select an image in the training gallery and load its description.
    
    Args:
        evt (gr.SelectData): Gradio selection event containing the selected image index
        descriptions (List[str]): Current list of descriptions
        
    Returns:
        Tuple[int, str]: Selected image index and its current description
    """
    index = evt.index
    current_description = descriptions[index] if index < len(descriptions) else ""
    return index, current_description


def get_training_outputs(output_path: str) -> Tuple[List[str], List[str]]:
    """
    Get all training output files from the specified directory.
    
    Args:
        output_path (str): Base path for training outputs
        
    Returns:
        Tuple[List[str], List[str]]: Lists of PDF plot files and ZIP checkpoint files
    """
    plot_dir = os.path.join(output_path, "_validation/plots")
    checkpoint_dir = os.path.join(output_path, "_checkpoints")

    # Ensure directories exist
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Get all PDF and ZIP files
    pdf_files = [str(pdf) for pdf in Path(plot_dir).glob("*.pdf")]
    zip_files = [str(zip_file) for zip_file in Path(checkpoint_dir).glob("*.zip")]
    
    return pdf_files, zip_files


def get_default_training_config() -> Dict[str, Any]:
    """
    Get the default training configuration with optimal settings.
    
    Returns:
        Dict[str, Any]: Complete default configuration for LoRA training
    """
    return {
        "model": "dev",
        "seed": 42,
        "quantize": None,
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
            "output_path": ""
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
            "path": "../../" + TEMP_IMAGES_DIR,
            "images": []
        }
    }


def build_training_config(
    default_config: Dict[str, Any],
    seed: Optional[int],
    steps: Optional[int],
    guidance: Optional[float],
    quantize: Optional[str],
    width: Optional[int],
    height: Optional[int],
    num_epochs: Optional[int],
    batch_size: Optional[int],
    plot_frequency: Optional[int],
    generate_image_frequency: Optional[int],
    validation_prompt: Optional[str]
) -> Dict[str, Any]:
    """
    Build training configuration by merging user parameters with defaults.
    
    Args:
        default_config (Dict[str, Any]): Base configuration with default values
        **kwargs: User-specified training parameters
        
    Returns:
        Dict[str, Any]: Complete training configuration with user overrides applied
    """
    config = default_config.copy()
    
    # Update top-level parameters with user values or defaults
    if seed is not None:
        config["seed"] = seed
    if steps is not None:
        config["steps"] = steps
    if guidance is not None:
        config["guidance"] = guidance
    if width is not None:
        config["width"] = width
    if height is not None:
        config["height"] = height
    
    # Handle quantization setting
    if quantize == "No quantization":
        config["quantize"] = None
    elif quantize:
        config["quantize"] = int(quantize)
    
    # Update training loop parameters
    if num_epochs is not None:
        config["training_loop"]["num_epochs"] = num_epochs
    if batch_size is not None:
        config["training_loop"]["batch_size"] = batch_size
    
    # Update instrumentation parameters
    if plot_frequency is not None:
        config["instrumentation"]["plot_frequency"] = plot_frequency
    if generate_image_frequency is not None:
        config["instrumentation"]["generate_image_frequency"] = generate_image_frequency
    if validation_prompt:
        config["instrumentation"]["validation_prompt"] = validation_prompt
    
    return config


def process_training_files(files, descriptions: List[str], config: Dict[str, Any]) -> None:
    """
    Process training files by copying them to temp directory and updating config.
    
    Args:
        files: List of uploaded file objects
        descriptions (List[str]): Corresponding descriptions for each file
        config (Dict[str, Any]): Training configuration to update with file information
    """
    for file, description in zip(files, descriptions):
        # Copy file to temporary training directory
        temp_file_path = os.path.join(TEMP_IMAGES_DIR, Path(file.name).name)
        shutil.copy(file.name, temp_file_path)
        
        # Add file and description to training config
        config["examples"]["images"].append({
            "image": Path(temp_file_path).name,
            "prompt": description
        })


def execute_training(json_path: str) -> Tuple[str, List[str], List[str]]:
    """
    Execute the LoRA training process with comprehensive error handling.
    
    Args:
        json_path (str): Path to the training configuration JSON file
        
    Returns:
        Tuple[str, List[str], List[str]]: Training status message, PDF files, ZIP files
    """
    training_spec = None
    training_state = None
    
    try:
        # Initialize DreamBooth training components
        flux, runtime_config, training_spec, training_state = DreamBoothInitializer.initialize(
            config_path=json_path,
            checkpoint_path=None
        )

        # Execute training process
        DreamBooth.train(
            flux=flux,
            runtime_config=runtime_config,
            training_spec=training_spec,
            training_state=training_state
        )

        # Get training outputs after completion
        pdf_files, zip_files = get_training_outputs(training_spec.saver.output_path)
        
        message = f"Entraînement terminé avec succès !\nConfiguration JSON sauvegardée dans {json_path}"
        return message, pdf_files, zip_files
        
    except StopTrainingException as stop_exc:
        # Handle early training termination gracefully
        if training_state and training_spec:
            training_state.save(training_spec)
            # Try to get partial outputs
            try:
                pdf_files, zip_files = get_training_outputs(training_spec.saver.output_path)
            except:
                pdf_files, zip_files = [], []
        else:
            pdf_files, zip_files = [], []
        message = f"Entraînement arrêté prématurément : {str(stop_exc)}"
        return message, pdf_files, zip_files
        
    except Exception as e:
        # Handle unexpected errors during training
        message = f"Erreur inattendue lors de l'entraînement : {str(e)}"
        return message, [], []


def prepare_training_json_and_start(
    files,
    descriptions: List[str],
    seed: Optional[int],
    steps: Optional[int],
    guidance: Optional[float],
    quantize: Optional[str],
    width: Optional[int],
    height: Optional[int],
    num_epochs: Optional[int],
    batch_size: Optional[int],
    plot_frequency: Optional[int],
    generate_image_frequency: Optional[int],
    validation_prompt: Optional[str],
    pdf_output: gr.Files,
    zip_output: gr.Files
) -> Tuple[str, List[str], List[str]]:
    """
    Prepare training configuration and start LoRA training process.
    
    This comprehensive function handles the entire LoRA training workflow:
    1. Validates input files and descriptions
    2. Creates training configuration JSON with user parameters
    3. Copies training images to temporary directory
    4. Initializes DreamBooth training components
    5. Executes the training process with error handling
    6. Returns training outputs
    
    Args:
        files: List of uploaded training image files
        descriptions (List[str]): Corresponding descriptions for each image
        seed (Optional[int]): Random seed for reproducible training
        steps (Optional[int]): Number of inference steps during training
        guidance (Optional[float]): Guidance scale for image generation
        quantize (Optional[str]): Quantization setting ("No quantization", "4", "8")
        width (Optional[int]): Generated image width in pixels
        height (Optional[int]): Generated image height in pixels
        num_epochs (Optional[int]): Number of training epochs
        batch_size (Optional[int]): Training batch size
        plot_frequency (Optional[int]): Frequency of plot generation during training
        generate_image_frequency (Optional[int]): Frequency of image generation during training
        validation_prompt (Optional[str]): Prompt used for validation image generation
        pdf_output (gr.Files): Gradio component for displaying training plots
        zip_output (gr.Files): Gradio component for displaying model checkpoints
        
    Returns:
        Tuple[str, List[str], List[str]]: Status message, PDF files, ZIP files
        
    Raises:
        Various exceptions related to file operations, model initialization, or training process
        
    Note:
        Creates a comprehensive training configuration with sensible defaults.
        Handles training interruptions gracefully with state saving.
    """
    # Validate input files and descriptions
    if not files:
        return "Aucun fichier téléchargé pour l'entraînement.", [], []

    if len(files) != len(descriptions):
        return "Le nombre de descriptions ne correspond pas au nombre de fichiers.", [], []

    # Define comprehensive default training configuration
    default_config = get_default_training_config()
    
    # Create training configuration with user parameters
    training_config = build_training_config(
        default_config, seed, steps, guidance, quantize, width, height,
        num_epochs, batch_size, plot_frequency, generate_image_frequency, validation_prompt
    )

    # Process training files and update configuration
    process_training_files(files, descriptions, training_config)

    # Setup training directory and save configuration
    train_dir = os.path.join(TEMP_TRAIN_DIR, "current_train")
    os.makedirs(train_dir, exist_ok=True)
    json_path = os.path.join(train_dir, "train.json")
    training_config["save"]["output_path"] = train_dir

    with open(json_path, "w") as json_file:
        json.dump(training_config, json_file, indent=4)

    # Execute training process with comprehensive error handling
    return execute_training(json_path)