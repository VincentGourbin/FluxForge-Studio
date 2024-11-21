import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from PIL import Image
import gradio as gr
from mflux import Flux1, Flux1Controlnet, Config, ConfigControlnet, ModelConfig
import datetime
from pathlib import Path
import os
import random
import json
from PIL import Image
# Importer les autres scripts
import config  # Assurez-vous d'importer le module config
from database import init_db, save_image_info, load_history, get_image_details, delete_image
from background_remover import load_background_removal_model, remove_background
from prompt_enhancer import enhance_prompt, update_image_input_visibility, update_button_label, models_info, model_names

# Initialiser la base de données
init_db()

# Chargement du modèle de suppression d'arrière-plan
modelbgrm = load_background_removal_model()

def generate_image(
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
    progress=gr.Progress(),
    *args
):
    # Accéder aux variables via le module config
    # Exemple : config.current_model_alias

    # Création du dossier pour les images intermédiaires
    stepwise_dir = Path("stepwise_output")
    stepwise_dir.mkdir(parents=True, exist_ok=True)
    
    # Nettoyer le dossier des images précédentes
    for file in stepwise_dir.glob("*.png"):
        try:
            os.remove(file)
        except:
            pass

    # Traitement des paramètres
    if seed is None or int(seed) == 0:
        seed = random.randint(1, 2**32 - 1)
    else:
        seed = int(seed)

    height = int(height)
    width = int(width)
    steps = max(1, int(steps))  # Assurer que steps >= 1
    guidance = float(guidance) if guidance else 3.5
    if quantize == 'None' or quantize is None:
        quantize = None
    else:
        quantize = int(quantize)
    controlnet_strength = float(controlnet_strength)
    controlnet_save_canny = bool(controlnet_save_canny)

    # Traitement des sélections et des scales pour les LoRA
    num_lora = len(config.lora_data)
    lora_checkbox_values = args[:num_lora]
    lora_scale_values = args[num_lora:2*num_lora]

    lora_paths_list = []
    lora_scales_list = []
    for idx, (selected, scale) in enumerate(zip(lora_checkbox_values, lora_scale_values)):
        if selected:
            lora_info = config.lora_data[idx]
            lora_file = lora_info['file_name']
            lora_path = os.path.join(config.lora_directory, lora_file)
            lora_paths_list.append(lora_path)
            lora_scales_list.append(float(scale))
            # Ajouter le mot-clé d'activation au prompt
            prompt = f"{lora_info['activation_keyword']}, {prompt}"

    # Déterminer si ControlNet est utilisé
    use_controlnet = controlnet_image_path is not None

    # Vérifier si le modèle doit être rechargé
    if (model_alias != config.current_model_alias) or (quantize != config.current_quantize) or (path != config.current_path) or \
        (lora_paths_list != config.current_lora_paths) or (lora_scales_list != config.current_lora_scales) or (config.flux_model is None) or \
        (use_controlnet != (config.current_model_type == 'controlnet')):

        # Charger la configuration du modèle
        if path:
            model_config = ModelConfig.from_pretrained(path=path)
        else:
            model_config = ModelConfig.from_alias(alias=model_alias)

        # Charger le modèle approprié
        if use_controlnet:
            flux_model = Flux1Controlnet(
                model_config=model_config,
                quantize=quantize,
                local_path=path,
                lora_paths=lora_paths_list,
                lora_scales=lora_scales_list,
            )
            current_model_type = 'controlnet'
        else:
            flux_model = Flux1(
                model_config=model_config,
                quantize=quantize,
                local_path=path,
                lora_paths=lora_paths_list,
                lora_scales=lora_scales_list,
            )
            current_model_type = 'standard'

        # Mettre à jour les paramètres du modèle courant
        config.current_model_alias = model_alias
        config.current_quantize = quantize
        config.current_path = path
        config.current_lora_paths = lora_paths_list
        config.current_lora_scales = lora_scales_list
        config.current_model_type = current_model_type
        config.flux_model = flux_model
    else:
        # Si le modèle n'a pas changé, utilisez le modèle existant
        flux_model = config.flux_model
        current_model_type = config.current_model_type

    # Construire la configuration
    if use_controlnet:
        config_obj = ConfigControlnet(
            num_inference_steps=steps,
            guidance=guidance,
            height=height,
            width=width,
            controlnet_strength=controlnet_strength,
        )
    else:
        config_obj = Config(
            num_inference_steps=steps,
            guidance=guidance,
            height=height,
            width=width,
        )

    # Obtenir la date et l'heure actuelles
    timestamp = datetime.datetime.now()
    timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')

    # Adapter le nom du fichier de sortie pour éviter les caractères invalides
    output_dir = Path("outputimage")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{seed}.png"

    # Générer l'image
    if use_controlnet:
        image = flux_model.generate_image(
            seed=seed,
            prompt=prompt,
            output=str(output_filename),
            controlnet_image_path=controlnet_image_path.name,
            controlnet_save_canny=controlnet_save_canny,
            config=config_obj,
            stepwise_output_dir=stepwise_dir
        )
    else:
        image = flux_model.generate_image(
            seed=seed,
            prompt=prompt,
            config=config_obj,
            stepwise_output_dir=stepwise_dir
        )
        
    image.save(path=str(output_filename), export_json_metadata=metadata)

    # Enregistrer les paramètres et le chemin de l'image dans la base de données
    save_image_info((
        timestamp_str,
        seed,
        prompt,
        model_alias,
        quantize,
        steps,
        guidance,
        height,
        width,
        path,
        controlnet_image_path.name if controlnet_image_path else None,
        controlnet_strength,
        controlnet_save_canny,
        json.dumps(lora_paths_list),  # Stocker les listes en tant que JSON
        json.dumps(lora_scales_list),
        str(output_filename)
    ))

    # Retourner l'image pour affichage dans Gradio
    return image.image  # Retourner l'image PIL


def update_guidance_visibility(model_alias):
    if model_alias == "dev":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)
    
# Fonction pour afficher les détails de l'image sélectionnée
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
# Fonction pour mettre à jour l'historique
def refresh_history():
    images = load_history()
    return images

def remove_background_wrapper(input_image):
    return remove_background(input_image, modelbgrm)

# Définir l'interface Gradio
model_options = ["schnell", "dev"]
quantize_options = [4, 8, None]

with gr.Blocks() as demo:
    gr.Markdown("# Générateur d'images mflux")

    with gr.Tab("Génération"):
        prompt = gr.Textbox(label="Prompt (description de l'image)", value="Luxury food photograph", lines=2)

        gr.Markdown("## Paramètres principaux")

        with gr.Row():
            model_alias = gr.Dropdown(label="Modèle", choices=config.model_options, value="schnell")
            quantize = gr.Dropdown(label="Quantize", choices=[str(q) for q in config.quantize_options], value="8")
            steps = gr.Number(label="Nombre d'étapes d'inférence", value=4, precision=0, minimum=1)
            seed = gr.Number(label="Seed", value=0, precision=0)
            metadata = gr.Checkbox(label="Exporter les métadonnées", value=True)

        with gr.Row():
            guidance = gr.Number(label="Guidance scale", value=3.5, visible=False)

        model_alias.change(fn=update_guidance_visibility, inputs=model_alias, outputs=guidance)

        with gr.Row():
            height = gr.Slider(label="Hauteur", minimum=256, maximum=4096, step=64, value=1024)
            width = gr.Slider(label="Largeur", minimum=256, maximum=4096, step=64, value=1024)

        gr.Markdown("## Sélection des LoRA")

        with gr.Accordion(""):
            lora_checkboxes = []
            lora_scales = []
            for lora in config.lora_data:
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
            fn=generate_image,
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
#demo.queue().launch(auth=("digitallab", "n6L64eMd2PxA5m"),share=True)