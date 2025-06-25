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

# Import mflux components (training imports moved to training_manager.py)

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
from training_manager import (
    update_gallery_and_descriptions,
    update_description,
    select_train_image,
    prepare_training_json_and_start
)

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


# Training functions have been moved to training_manager.py for better organization

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
            enable_stepwise = gr.Checkbox(
                label="🎬 Stepwise (Temps Réel) - Affiche chaque étape de génération", 
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

        # Generation button and outputs
        btn = gr.Button("🎨 Générer l'image", variant="primary", size="lg")
        
        # Status display
        status_display = gr.Markdown("**Statut:** Prêt à générer")
        
        # Single image output that shows both stepwise and final
        output_image = gr.Image(
            label="🖼️ Image (Étapes + Finale)", 
            height=500
        )

        # Wrapper function to handle both stepwise and normal generation
        def generate_wrapper(*inputs_args):
            try:
                # Extract enable_stepwise from inputs
                prompt, model_alias, quantize, steps, seed, metadata, guidance, height, width, path, controlnet_image_path, controlnet_strength, controlnet_save_canny, enable_stepwise_val = inputs_args[:14]
                lora_args = inputs_args[14:]
                
                if enable_stepwise_val:
                    # Use streaming generator for stepwise - yield each step
                    for current_image, final_image, status in image_generator.generate_image_with_stepwise_streaming(
                        prompt,
                        model_alias,
                        quantize,
                        steps,
                        seed,
                        metadata,
                        guidance,
                        height,
                        width,
                        path,
                        controlnet_image_path,
                        controlnet_strength,
                        controlnet_save_canny,
                        True,  # enable_stepwise
                        *lora_args
                    ):
                        # Yield current step image or final image
                        display_image = final_image if final_image is not None else current_image
                        yield display_image, status
                else:
                    # Standard generation
                    yield None, "🚀 Génération en cours..."
                    final_image = image_generator.generate_image(
                        prompt,
                        model_alias,
                        quantize,
                        steps,
                        seed,
                        metadata,
                        guidance,
                        height,
                        width,
                        path,
                        controlnet_image_path,
                        controlnet_strength,
                        controlnet_save_canny,
                        False,  # enable_stepwise
                        gr.Progress(),  # progress parameter
                        *lora_args
                    )
                    yield final_image, "✅ Génération terminée !"
                    
            except Exception as e:
                yield None, f"❌ Erreur: {str(e)}"

        # Configure input list for generation function
        inputs = [
            prompt, model_alias, quantize, steps, seed, metadata, guidance,
            height, width, path, controlnet_image_path, controlnet_strength,
            controlnet_save_canny, enable_stepwise
        ] + lora_checkboxes + lora_scales

        # Connect generation button to wrapper function
        btn.click(
            fn=generate_wrapper,
            inputs=inputs,
            outputs=[output_image, status_display],
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
        # Set initial visibility based on first model
        initial_image_visible = False
        if model_names and model_names[0]:
            from prompt_enhancer import models_info
            first_model_capabilities = models_info.get(model_names[0], [])
            initial_image_visible = 'vision' in first_model_capabilities
            
        input_image = gr.Image(
            label="Image à fournir au modèle (si requis) - Certains modèles peuvent analyser des images",
            type='filepath',
            visible=initial_image_visible
        )

        # Update image visibility based on model selection
        selected_model.change(
            fn=update_image_input_visibility,
            inputs=selected_model,
            outputs=input_image
        )

        # Enhancement button with dynamic label
        # Set initial button label based on first model
        initial_button_label = "🚀 Améliorer le prompt"
        if model_names and model_names[0]:
            first_model_capabilities = models_info.get(model_names[0], [])
            if 'vision' in first_model_capabilities:
                initial_button_label = "🚀 Analyser l'image"
                
        enhance_button = gr.Button(initial_button_label, variant="primary")

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

        # Image selection function is now imported from training_manager

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

        # Wrapper function to handle training and update file outputs
        def training_wrapper(*args):
            """Wrapper to handle training outputs and update file components."""
            # Call the training function
            message, pdf_files, zip_files = prepare_training_json_and_start(*args)
            
            # Return all outputs for Gradio
            return message, pdf_files, zip_files

        train_button.click(
            fn=training_wrapper,
            inputs=[
                train_files, descriptions_state, seed, steps, guidance,
                quantize_dropdown, width, height, num_epochs, batch_size,
                plot_frequency, generate_image_frequency, validation_prompt, pdf_output, zip_output
            ],
            outputs=[train_output, pdf_output, zip_output],
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