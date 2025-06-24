"""
mflux-gradio: A comprehensive web interface for MFLUX image generation with advanced features.

This module provides a Gradio-based web interface for the MFLUX image generation system,
including features for:
- Image generation with customizable parameters and LoRA support
- Prompt enhancement using Ollama integration
- Background removal from images
- Image history management with gallery view
- LoRA training interface with comprehensive parameter control

The interface is organized into 5 main tabs:
1. Generation: Main image generation with model selection and parameter tuning
2. Prompt Enhancer: AI-powered prompt improvement using Ollama models
3. Background Remover: Remove backgrounds from uploaded images
4. History: View and manage previously generated images
5. Train: LoRA training interface for custom model fine-tuning

Author: MFLUX Team
License: MIT
"""

import os
import json
import shutil
import threading
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import gradio as gr

# Set PyTorch MPS fallback for Apple Silicon compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Import mflux components
from mflux.dreambooth.dreambooth import DreamBooth
from mflux.dreambooth.dreambooth_initializer import DreamBoothInitializer
from mflux.error.exceptions import StopTrainingException

# Import local modules
from database import init_db, load_history, get_image_details, delete_image
from background_remover import load_background_removal_model, remove_background
from prompt_enhancer import (
    enhance_prompt, 
    update_image_input_visibility, 
    update_button_label, 
    model_names
)
from image_generator import ImageGenerator

# Global constants
TEMP_IMAGES_DIR = "temp_images"
TEMP_TRAIN_DIR = "temp_train"

# Initialize application components
init_db()
modelbgrm = load_background_removal_model()
image_generator = ImageGenerator()

# Create temporary directories for image processing and training
os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)
os.makedirs(TEMP_TRAIN_DIR, exist_ok=True)

# ==============================================================================
# UTILITY FUNCTIONS FOR IMAGE HISTORY MANAGEMENT
# ==============================================================================

def show_image_details(evt: gr.SelectData, selected_image_index: Optional[int]) -> Tuple[str, int]:
    """
    Display detailed information for a selected image from the history gallery.
    
    This function is triggered when a user clicks on an image in the history gallery.
    It retrieves and formats the image metadata including generation parameters,
    creation timestamp, and file information.
    
    Args:
        evt (gr.SelectData): Gradio selection event containing the selected image index
        selected_image_index (Optional[int]): Currently selected image index (unused but maintained for compatibility)
        
    Returns:
        Tuple[str, int]: A tuple containing:
            - details_text (str): Formatted string with image details and metadata
            - index (int): The index of the selected image for state management
    """
    index = evt.index
    details_text = get_image_details(index)
    return details_text, index

def delete_selected_image(selected_image_index: Optional[int]) -> Tuple[str, gr.update, Optional[int]]:
    """
    Delete a selected image from the history and refresh the gallery.
    
    This function handles the deletion of images from both the database and filesystem.
    It provides user feedback and automatically refreshes the gallery view after
    successful deletion.
    
    Args:
        selected_image_index (Optional[int]): Index of the image to delete from history
        
    Returns:
        Tuple[str, gr.update, Optional[int]]: A tuple containing:
            - message (str): Success or error message for user feedback
            - gallery_update (gr.update): Updated gallery component or no-change update
            - new_index (Optional[int]): Reset index (None) or unchanged index
            
    Note:
        If no image is selected, returns an error message without making changes.
        On successful deletion, resets the selected index and refreshes the gallery.
    """
    if selected_image_index is None:
        return "Veuillez sélectionner une image à supprimer.", gr.update(), selected_image_index

    success, message = delete_image(selected_image_index)

    if success:
        # Reset selected index and refresh gallery
        selected_image_index = None
        images = load_history()
        return message, images, selected_image_index
    else:
        return message, gr.update(), selected_image_index

def refresh_history() -> List:
    """
    Refresh and load the complete image history from the database.
    
    This function retrieves all generated images from the database and returns
    them in a format suitable for display in the Gradio gallery component.
    
    Returns:
        List: List of image data for gallery display
    """
    return load_history()


# ==============================================================================
# BACKGROUND REMOVAL FUNCTIONS
# ==============================================================================

def remove_background_wrapper(input_image) -> Any:
    """
    Wrapper function for background removal functionality.
    
    This wrapper provides a clean interface between the Gradio component
    and the background removal model, handling the image processing pipeline.
    
    Args:
        input_image: Input image from Gradio component (PIL Image format)
        
    Returns:
        Any: Processed image with background removed
        
    Note:
        Uses the globally loaded background removal model (modelbgrm) for processing.
    """
    return remove_background(input_image, modelbgrm)


# ==============================================================================
# TRAINING INTERFACE UTILITY FUNCTIONS
# ==============================================================================

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

# ==============================================================================
# TRAINING MONITORING AND EXECUTION FUNCTIONS
# ==============================================================================

def monitor_training_directory(output_path: str, pdf_output: gr.Files, zip_output: gr.Files) -> None:
    """
    Monitor training output directories for new files and update the interface in real-time.
    
    This function continuously monitors the training output directories for new PDF plots
    and ZIP checkpoint files, updating the Gradio interface components when new files
    are detected. It runs in a separate thread to avoid blocking the main interface.
    
    Args:
        output_path (str): Base path for training outputs
        pdf_output (gr.Files): Gradio Files component for displaying PDF plots
        zip_output (gr.Files): Gradio Files component for displaying ZIP checkpoints
        
    Note:
        This function runs in an infinite loop with a 2-second polling interval.
        It creates the necessary subdirectories if they don't exist.
        
    Directory Structure:
        - {output_path}/_validation/plots/: Contains training plot PDFs
        - {output_path}/_checkpoints/: Contains model checkpoint ZIP files
    """
    plot_dir = os.path.join(output_path, "_validation/plots")
    checkpoint_dir = os.path.join(output_path, "_checkpoints")

    # Ensure directories exist
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Track previously seen files to detect new ones
    seen_pdfs = set()
    seen_zips = set()

    while True:
        # Monitor PDF files in validation plots directory
        current_pdfs = set(Path(plot_dir).glob("*.pdf"))
        new_pdfs = current_pdfs - seen_pdfs
        seen_pdfs.update(new_pdfs)

        # Monitor ZIP files in checkpoints directory
        current_zips = set(Path(checkpoint_dir).glob("*.zip"))
        new_zips = current_zips - seen_zips
        seen_zips.update(new_zips)

        # Update Gradio interface with new files
        if new_pdfs:
            pdf_output.update([str(pdf) for pdf in new_pdfs])
        if new_zips:
            zip_output.update([str(zip_file) for zip_file in new_zips])

        # Sleep to prevent excessive CPU usage
        time.sleep(2)

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
) -> str:
    """
    Prepare training configuration and start LoRA training process.
    
    This comprehensive function handles the entire LoRA training workflow:
    1. Validates input files and descriptions
    2. Creates training configuration JSON with user parameters
    3. Copies training images to temporary directory
    4. Initializes DreamBooth training components
    5. Starts monitoring thread for training outputs
    6. Executes the training process with error handling
    
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
        str: Status message indicating training success, early stop, or error details
        
    Raises:
        Various exceptions related to file operations, model initialization, or training process
        
    Note:
        Creates a comprehensive training configuration with sensible defaults.
        Runs monitoring in a separate thread to track training progress.
        Handles training interruptions gracefully with state saving.
    """
    # Validate input files and descriptions
    if not files:
        return "Aucun fichier téléchargé pour l'entraînement."

    if len(files) != len(descriptions):
        return "Le nombre de descriptions ne correspond pas au nombre de fichiers."

    # Define comprehensive default training configuration
    default_config = _get_default_training_config()
    
    # Create training configuration with user parameters
    training_config = _build_training_config(
        default_config, seed, steps, guidance, quantize, width, height,
        num_epochs, batch_size, plot_frequency, generate_image_frequency, validation_prompt
    )

    # Process training files and update configuration
    _process_training_files(files, descriptions, training_config)

    # Setup training directory and save configuration
    train_dir = os.path.join(TEMP_TRAIN_DIR, "current_train")
    os.makedirs(train_dir, exist_ok=True)
    json_path = os.path.join(train_dir, "train.json")
    training_config["save"]["output_path"] = train_dir

    with open(json_path, "w") as json_file:
        json.dump(training_config, json_file, indent=4)

    # Execute training process with comprehensive error handling
    return _execute_training(json_path, pdf_output, zip_output)


def _get_default_training_config() -> Dict[str, Any]:
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


def _build_training_config(
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
        **kwargs: User-specified training parameters (see prepare_training_json_and_start)
        
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


def _process_training_files(files, descriptions: List[str], config: Dict[str, Any]) -> None:
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


def _execute_training(json_path: str, pdf_output: gr.Files, zip_output: gr.Files) -> str:
    """
    Execute the LoRA training process with comprehensive error handling.
    
    Args:
        json_path (str): Path to the training configuration JSON file
        pdf_output (gr.Files): Gradio component for training plots
        zip_output (gr.Files): Gradio component for model checkpoints
        
    Returns:
        str: Training status message (success, early stop, or error)
    """
    try:
        # Initialize DreamBooth training components
        flux, runtime_config, training_spec, training_state = DreamBoothInitializer.initialize(
            config_path=json_path,
            checkpoint_path=None
        )

        # Start monitoring thread for training outputs
        monitor_thread = threading.Thread(
            target=monitor_training_directory,
            args=(training_spec.saver.output_path, pdf_output, zip_output)
        )
        monitor_thread.daemon = True  # Allow main thread to exit
        monitor_thread.start()

        # Execute training process
        DreamBooth.train(
            flux=flux,
            runtime_config=runtime_config,
            training_spec=training_spec,
            training_state=training_state
        )

        return f"Entraînement terminé avec succès !\nConfiguration JSON sauvegardée dans {json_path}"
        
    except StopTrainingException as stop_exc:
        # Handle early training termination gracefully
        training_state.save(training_spec)
        return f"Entraînement arrêté prématurément : {str(stop_exc)}"
        
    except Exception as e:
        # Handle unexpected errors during training
        return f"Erreur inattendue lors de l'entraînement : {str(e)}"
    

# ==============================================================================
# GRADIO INTERFACE DEFINITION
# ==============================================================================

with gr.Blocks(title="MFLUX Image Generator") as demo:
    gr.Markdown("# Générateur d'images MFLUX")
    gr.Markdown("Interface complète pour la génération d'images, l'amélioration de prompts, la suppression d'arrière-plan et l'entraînement de modèles LoRA.")

    # ==============================================================================
    # TAB 1: IMAGE GENERATION
    # ==============================================================================
    with gr.Tab("Génération"):
        gr.Markdown("## Génération d'images avec paramètres avancés")
        
        # Main prompt input
        prompt = gr.Textbox(
            label="Prompt (description de l'image)", 
            value="Luxury food photograph", 
            lines=2,
            placeholder="Décrivez l'image que vous souhaitez générer..."
        )

        gr.Markdown("### Paramètres principaux")
        
        # Primary generation parameters
        with gr.Row():
            model_alias = gr.Dropdown(
                label="Modèle - Choisissez le modèle de génération", 
                choices=image_generator.model_options, 
                value="schnell"
            )
            quantize = gr.Dropdown(
                label="Quantize - Niveau de quantification pour l'optimisation mémoire", 
                choices=[str(q) for q in image_generator.quantize_options], 
                value="8"
            )
            steps = gr.Number(
                label="Nombre d'étapes d'inférence - Plus d'étapes = meilleure qualité mais plus lent", 
                value=4, 
                precision=0, 
                minimum=1
            )
            seed = gr.Number(
                label="Seed - 0 = aléatoire, autre valeur = reproductible", 
                value=0, 
                precision=0
            )
            metadata = gr.Checkbox(
                label="Exporter les métadonnées - Inclure les paramètres de génération dans l'image", 
                value=True
            )

        # Guidance parameter (conditionally visible)
        with gr.Row():
            guidance = gr.Number(
                label="Guidance scale - Contrôle l'adhérence au prompt", 
                value=3.5, 
                visible=False
            )

        # Update guidance visibility based on model selection
        model_alias.change(
            fn=image_generator.update_guidance_visibility, 
            inputs=model_alias, 
            outputs=guidance
        )

        # Image dimensions
        with gr.Row():
            height = gr.Slider(
                label="Hauteur - Hauteur de l'image en pixels", 
                minimum=256, 
                maximum=4096, 
                step=64, 
                value=1024
            )
            width = gr.Slider(
                label="Largeur - Largeur de l'image en pixels", 
                minimum=256, 
                maximum=4096, 
                step=64, 
                value=1024
            )

        gr.Markdown("### Sélection des LoRA")
        
        # LoRA selection interface
        with gr.Accordion("Modèles LoRA disponibles", open=False):
            lora_checkboxes = []
            lora_scales = []
            for lora in image_generator.lora_data:
                # Display LoRA information
                gr.Markdown(f"**Description**: {lora['description']}")
                gr.Markdown(f"**Mot-clé d'activation**: `{lora['activation_keyword']}`")
                with gr.Row():
                    checkbox = gr.Checkbox(
                        label=f"Utiliser {lora['file_name']}", 
                        value=False, 
                        scale=1
                    )
                    scale_input = gr.Slider(
                        label="Intensité - Intensité de l'effet LoRA", 
                        minimum=0.0, 
                        maximum=1.0, 
                        value=1.0, 
                        step=0.1, 
                        scale=2
                    )
                    lora_checkboxes.append(checkbox)
                    lora_scales.append(scale_input)

        # ControlNet configuration
        with gr.Accordion("ControlNet", open=False):
            gr.Markdown("Contrôlez la génération avec une image de référence")
            with gr.Group():
                controlnet_image_path = gr.File(
                    label="Image ControlNet - Image de référence pour guider la génération"
                )
                controlnet_strength = gr.Slider(
                    label="Force ControlNet - Influence de l'image de contrôle", 
                    minimum=0.0, 
                    maximum=1.0, 
                    step=0.1, 
                    value=0.4
                )
                controlnet_save_canny = gr.Checkbox(
                    label="Sauvegarder l'image Canny (fil de fer) - Exporter l'image de détection des contours", 
                    value=False
                )

        # Local model path (advanced option)
        with gr.Accordion("Options avancées", open=False):
            path = gr.Textbox(
                label="Chemin vers un modèle local - Optionnel: utilisez un modèle local au lieu des modèles prédéfinis", 
                value="",
                placeholder="/chemin/vers/modele/local"
            )

        # Generation button and output
        btn = gr.Button("🎨 Générer l'image", variant="primary", size="lg")
        output_image = gr.Image(label="Image générée", height=400)

        # Configure input list for generation function
        inputs = [
            prompt, model_alias, quantize, steps, seed, metadata, guidance,
            height, width, path, controlnet_image_path, controlnet_strength,
            controlnet_save_canny
        ] + lora_checkboxes + lora_scales

        # Connect generation button to image generator
        btn.click(
            fn=image_generator.generate_image,
            inputs=inputs,
            outputs=output_image,
            show_progress=True
        )

    # ==============================================================================
    # TAB 2: PROMPT ENHANCER
    # ==============================================================================
    with gr.Tab("Prompt Enhancer"):
        gr.Markdown("## Amélioration du prompt à l'aide d'Ollama")
        gr.Markdown("Utilisez les modèles Ollama pour améliorer et détailler vos prompts de génération d'images.")

        with gr.Row():
            selected_model = gr.Dropdown(
                label="Sélectionnez un modèle Ollama - Modèle d'IA pour l'amélioration de prompts",
                choices=model_names,
                value=model_names[0] if model_names else None
            )
            input_text = gr.Textbox(
                label="Texte à traiter - Prompt simple que vous souhaitez améliorer",
                placeholder="Saisissez votre prompt basique ici...",
                lines=3
            )
        
        # Image input (conditionally visible based on model)
        input_image = gr.Image(
            label="Image à fournir au modèle (si requis) - Certains modèles peuvent analyser des images",
            type='filepath',
            visible=False
        )

        # Update image visibility based on model selection
        selected_model.change(
            fn=update_image_input_visibility,
            inputs=selected_model,
            outputs=input_image
        )

        # Enhancement button with dynamic label
        enhance_button = gr.Button("🚀 Améliorer le prompt", variant="primary")

        # Update button label based on model selection
        selected_model.change(
            fn=update_button_label,
            inputs=selected_model,
            outputs=enhance_button
        )
        
        # Enhanced output display
        enhanced_output = gr.Markdown(
            label="Prompt amélioré",
            value="Le prompt amélioré apparaîtra ici...",
            elem_classes=["enhanced-output"]
        )

        # Connect enhancement functionality
        enhance_button.click(
            fn=enhance_prompt,
            inputs=[selected_model, input_text, input_image],
            outputs=enhanced_output,
            show_progress=True
        )

    # ==============================================================================
    # TAB 3: BACKGROUND REMOVER
    # ==============================================================================
    with gr.Tab("Background Remover"):
        gr.Markdown("## Supprimer l'arrière-plan d'une image")
        gr.Markdown("Retirez automatiquement l'arrière-plan de vos images en utilisant un modèle de segmentation avancé.")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Image d'entrée - Téléchargez l'image dont vous voulez retirer l'arrière-plan", 
                    type='pil',
                    height=400
                )
                remove_bg_button = gr.Button(
                    "🖼️ Supprimer l'arrière-plan", 
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column():
                output_image = gr.Image(
                    label="Image sans arrière-plan - Résultat avec arrière-plan transparent",
                    height=400
                )
                
                gr.Markdown("""
                ### 💡 Conseils d'utilisation:
                - Utilisez des images avec des sujets bien définis
                - Évitez les arrière-plans trop complexes
                - L'image de sortie aura un fond transparent (PNG)
                """)
            
        # Connect background removal function
        remove_bg_button.click(
            fn=remove_background_wrapper,
            inputs=input_image,
            outputs=output_image,
            show_progress=True
        )
            
    # ==============================================================================
    # TAB 4: HISTORY MANAGEMENT
    # ==============================================================================
    with gr.Tab("Historique"):
        gr.Markdown("## Historique des images générées")
        gr.Markdown("Consultez, gérez et supprimez vos images générées précédemment.")
        
        with gr.Row():
            with gr.Column(scale=2):
                history_gallery = gr.Gallery(
                    label="Galerie d'images - Cliquez sur une image pour voir ses détails",
                    columns=4,
                    height=500,
                    object_fit="contain",
                    allow_preview=True
                )
                
                with gr.Row():
                    refresh_history_button = gr.Button(
                        "🔄 Rafraîchir l'historique", 
                        variant="secondary"
                    )
                    delete_button = gr.Button(
                        "🗑️ Supprimer l'image sélectionnée", 
                        variant="stop"
                    )
                    
            with gr.Column(scale=1):
                history_info = gr.Textbox(
                    label="Informations sur l'image sélectionnée - Métadonnées et paramètres de génération",
                    lines=20,
                    interactive=False,
                    placeholder="Sélectionnez une image pour voir ses détails..."
                )

        # State management for selected image
        selected_image_index = gr.State(None)

        # Load gallery on interface startup
        demo.load(fn=refresh_history, inputs=[], outputs=history_gallery)

        # Connect event handlers
        history_gallery.select(
            fn=show_image_details, 
            inputs=selected_image_index, 
            outputs=[history_info, selected_image_index]
        )
        refresh_history_button.click(
            fn=refresh_history, 
            inputs=[], 
            outputs=history_gallery
        )
        delete_button.click(
            fn=delete_selected_image, 
            inputs=selected_image_index, 
            outputs=[history_info, history_gallery, selected_image_index]
        )

    # Onglet pour l'entraînement
    with gr.Tab("Train"):
        gr.Markdown("## Entraîner un modèle avec vos fichiers")
        gr.Markdown("Téléchargez plusieurs images ci-dessous et ajoutez une description pour chacune.")

        # Composants pour les fichiers, gallery et descriptions
        train_files = gr.Files(label="Fichiers d'entraînement", type="filepath")
        gallery = gr.Gallery(label="Images téléchargées", columns=3, height="auto", object_fit="contain")
        description_textbox = gr.Textbox(
            label="Description pour l'image sélectionnée",
            placeholder="Entrez une description ici...",
            lines=2
        )
        descriptions_state = gr.State([])  # Stocke les descriptions associées
        selected_image_index = gr.Number(label="Index de l'image sélectionnée", value=0, visible=False)

        # Helper function for image selection in training interface
        def select_train_image(evt: gr.SelectData, descriptions):
            """Select an image in the training gallery and load its description."""
            index = evt.index
            current_description = descriptions[index] if index < len(descriptions) else ""
            return index, current_description

        # Connect file upload to gallery update
        train_files.change(
            fn=update_gallery_and_descriptions,
            inputs=train_files,
            outputs=[gallery, descriptions_state, description_textbox]
        )

        # Connect description update
        description_textbox.change(
            fn=update_description,
            inputs=[selected_image_index, description_textbox, descriptions_state],
            outputs=descriptions_state
        )

        # Connect image selection
        gallery.select(
            fn=select_train_image,
            inputs=descriptions_state,
            outputs=[selected_image_index, description_textbox]
        )

        # Paramètres d'entraînement
        gr.Markdown("### Paramètres généraux")
        seed = gr.Number(label="Seed", value=42, precision=0)
        steps = gr.Number(label="Steps", value=20, precision=0)
        guidance = gr.Number(label="Guidance", value=3.0)
        quantize_dropdown = gr.Dropdown(label="Quantize", choices=["No quantization", "4", "8"], value="4")
        width = gr.Number(label="Width", value=512, precision=0)
        height = gr.Number(label="Height", value=512, precision=0)

        gr.Markdown("### Boucle d'entraînement")
        num_epochs = gr.Number(label="Nombre d'époques", value=100, precision=0)
        batch_size = gr.Number(label="Taille de lot", value=1, precision=0)

        gr.Markdown("### Instrumentation")
        plot_frequency = gr.Number(label="Plot frequency", value=1, precision=0)
        generate_image_frequency = gr.Number(label="Generate image frequency", value=20, precision=0)
        validation_prompt = gr.Textbox(label="Validation prompt", value="photo of sks dog")

        # Bouton pour lancer l'entraînement
        train_button = gr.Button("Lancer l'entraînement")
        pdf_output = gr.Files(label="Plots PDF générés", type="filepath", interactive=False)
        zip_output = gr.Files(label="Checkpoints ZIP générés", type="filepath", interactive=False)
        train_output = gr.Markdown(label="Compte-rendu de l’entraînement")

        train_button.click(
            fn=prepare_training_json_and_start,
            inputs=[
                train_files, descriptions_state, seed, steps, guidance,
                quantize_dropdown, width, height, num_epochs, batch_size,
                plot_frequency, generate_image_frequency, validation_prompt, pdf_output, zip_output
            ],
            outputs=train_output,
            show_progress=True,
        )
# ==============================================================================
# APPLICATION LAUNCH
# ==============================================================================

if __name__ == "__main__":
    # Launch the Gradio interface with optimal settings
    demo.queue(
        max_size=20  # Maximum queue size
    ).launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,  # Default Gradio port
        share=False,  # Set to True to create a public Gradio link
        debug=False,  # Set to True for development debugging
        show_error=True,  # Show detailed error messages
        quiet=False  # Set to True to reduce console output
    )