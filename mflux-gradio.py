import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import gradio as gr
from pathlib import Path
import os

# Importer les autres scripts

from database import init_db, load_history, get_image_details, delete_image
from background_remover import load_background_removal_model, remove_background
from prompt_enhancer import enhance_prompt, update_image_input_visibility, update_button_label, models_info, model_names
from image_generator import ImageGenerator

# Initialiser la base de données
init_db()

# Chargement du modèle de suppression d'arrière-plan
modelbgrm = load_background_removal_model()



# Créer une instance de la classe ImageGenerator
image_generator = ImageGenerator()

# Les autres fonctions restent inchangées
def show_image_details(evt: gr.SelectData, selected_image_index):
    index = evt.index
    selected_image_index = index

    details_text = get_image_details(index)
    return details_text, selected_image_index

def delete_selected_image(selected_image_index):
    if selected_image_index is None:
        return gr.update(value="Veuillez sélectionner une image à supprimer."), gr.update(), selected_image_index

    success, message = delete_image(selected_image_index)

    if success:
        # Réinitialiser l'index sélectionné
        selected_image_index = None

        # Rafraîchir la galerie
        images = load_history()

        # Retourner le message de succès, la galerie mise à jour et l'index réinitialisé
        return message, images, selected_image_index
    else:
        return message, gr.update(), selected_image_index

def refresh_history():
    images = load_history()
    return images

def remove_background_wrapper(input_image):
    return remove_background(input_image, modelbgrm)

# Définir l'interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Générateur d'images mflux")

    with gr.Tab("Génération"):
        prompt = gr.Textbox(label="Prompt (description de l'image)", value="Luxury food photograph", lines=2)

        gr.Markdown("## Paramètres principaux")

        with gr.Row():
            model_alias = gr.Dropdown(label="Modèle", choices=image_generator.model_options, value="schnell")
            quantize = gr.Dropdown(label="Quantize", choices=[str(q) for q in image_generator.quantize_options], value="8")
            steps = gr.Number(label="Nombre d'étapes d'inférence", value=4, precision=0, minimum=1)
            seed = gr.Number(label="Seed", value=0, precision=0)
            metadata = gr.Checkbox(label="Exporter les métadonnées", value=True)

        with gr.Row():
            guidance = gr.Number(label="Guidance scale", value=3.5, visible=False)

        model_alias.change(fn=image_generator.update_guidance_visibility, inputs=model_alias, outputs=guidance)

        with gr.Row():
            height = gr.Slider(label="Hauteur", minimum=256, maximum=4096, step=64, value=1024)
            width = gr.Slider(label="Largeur", minimum=256, maximum=4096, step=64, value=1024)

        gr.Markdown("## Sélection des LoRA")

        with gr.Accordion(""):
            lora_checkboxes = []
            lora_scales = []
            for lora in image_generator.lora_data:
                # Afficher la description et le mot-clé d'activation
                gr.Markdown(f"**Description**: {lora['description']}")
                gr.Markdown(f"**Mot-clé d'activation**: {lora['activation_keyword']}")
                with gr.Row():
                    checkbox = gr.Checkbox(label=f"Utiliser {lora['file_name']}", value=False, scale=1)
                    scale_input = gr.Slider(label=f"Scale", minimum=0.0, maximum=1.0, value=1.0, step=0.1, scale=2)
                    lora_checkboxes.append(checkbox)
                    lora_scales.append(scale_input)

        with gr.Accordion("ControlNet"):
            with gr.Group():
                controlnet_image_path = gr.File(label="Image ControlNet")
                controlnet_strength = gr.Slider(label="ControlNet Strength", minimum=0.0, maximum=1.0, step=0.1, value=0.4)
                controlnet_save_canny = gr.Checkbox(label="Sauvegarder l'image Canny (fil de fer)", value=False)

        with gr.Row():
            path = gr.Textbox(label="Chemin vers un modèle local", value="")

        btn = gr.Button("Générer l'image")
        output_image = gr.Image(label="Image générée")

        # Définir les entrées
        inputs = [
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
        ] + lora_checkboxes + lora_scales

        # Configuration du clic sur le bouton avec mise à jour en direct
        btn.click(
            fn=image_generator.generate_image,
            inputs=inputs,
            outputs=output_image,
            show_progress=True
        )

    with gr.Tab("Prompt Enhancer"):
        gr.Markdown("## Amélioration du prompt à l'aide d'Ollama")

        with gr.Row():
            selected_model = gr.Dropdown(
                label="Sélectionnez un modèle Ollama",
                choices=model_names,
                value=model_names[0] if model_names else None
            )
            input_text = gr.Textbox(
                label="Texte à traiter",
                placeholder="Saisissez votre texte ici..."
            )
        
        # Zone de dépôt d'image, initialement cachée
        input_image = gr.Image(
            label="Image à fournir au modèle (si requis)",
            type='filepath',
            visible=False
        )

        # Mettre à jour la visibilité de la zone d'image en fonction du modèle sélectionné
        selected_model.change(
            fn=update_image_input_visibility,
            inputs=selected_model,
            outputs=input_image
        )

        # Bouton avec label dynamique
        enhance_button = gr.Button("Améliorer le prompt")

        # Mettre à jour le label du bouton en fonction du modèle sélectionné
        selected_model.change(
            fn=update_button_label,
            inputs=selected_model,
            outputs=enhance_button
        )
        enhanced_output = gr.Markdown(
            label="Texte détaillé"
        )

        enhance_button.click(
            fn=enhance_prompt,
            inputs=[selected_model, input_text, input_image],
            outputs=enhanced_output,
            show_progress=True
        )

    with gr.Tab("Background Remover"):
        gr.Markdown("## Supprimer l'arrière-plan d'une image")
        with gr.Column():
            input_image = gr.Image(label="Image d'entrée", type='pil')
            remove_bg_button = gr.Button("Remove background")
            output_image = gr.Image(label="Image sans arrière-plan")
            
        remove_bg_button.click(
            fn=remove_background_wrapper,
            inputs=input_image,
            outputs=output_image
        )
            
    with gr.Tab("Historique"):
        with gr.Column():
            history_gallery = gr.Gallery(
                label="Historique des images générées",
                columns=4,
                height='auto',
                object_fit="contain"
            )
            refresh_history_button = gr.Button("Rafraîchir l'historique")
            delete_button = gr.Button("Supprimer l'image sélectionnée")
            history_info = gr.Textbox(
                label="Informations sur l'image sélectionnée",
                lines=15,
                interactive=False
            )

        selected_image_index = gr.State(None)

        # Utiliser l'événement .load() sur l'objet Blocks
        demo.load(fn=refresh_history, inputs=[], outputs=history_gallery)

        # Connecter les événements
        history_gallery.select(fn=show_image_details, inputs=selected_image_index, outputs=[history_info, selected_image_index])
        refresh_history_button.click(fn=refresh_history, inputs=[], outputs=history_gallery)
        delete_button.click(fn=delete_selected_image, inputs=selected_image_index, outputs=[history_info, history_gallery, selected_image_index])

demo.queue().launch()