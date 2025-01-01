import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import gradio as gr
from pathlib import Path
import time
import os
import json
import shutil
from mflux.dreambooth.dreambooth import DreamBooth
from mflux.dreambooth.dreambooth_initializer import DreamBoothInitializer
from mflux.error.exceptions import StopTrainingException
import threading

# Importer les autres scripts
from database import init_db, load_history, get_image_details, delete_image
from background_remover import load_background_removal_model, remove_background
from prompt_enhancer import enhance_prompt, update_image_input_visibility, update_button_label, models_info, model_names
from image_generator import ImageGenerator

# Initialiser la base de données
init_db()

# Chargement du modèle de suppression d'arrière-plan
modelbgrm = load_background_removal_model()

# Créer un répertoire temporaire pour les images
TEMP_IMAGES_DIR = "temp_images"
TEMP_TRAIN_DIR = "temp_train"
os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)
os.makedirs(TEMP_TRAIN_DIR, exist_ok=True)

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



# Fonction pour mettre à jour la Gallery et initialiser les descriptions
def update_gallery_and_descriptions(files):
    if not files:
        return [], [], "Aucun fichier chargé."
    images = [file.name for file in files]
    descriptions = [""] * len(images)  # Initialise des descriptions vides
    return images, descriptions, ""

# Fonction pour associer une description à l'élément sélectionné
def update_description(index, new_description, descriptions):
    descriptions[int(index)] = new_description
    return descriptions

def monitor_training_directory(output_path, pdf_output, zip_output):
    """
    Surveille les nouveaux fichiers dans les sous-dossiers `_validation/plots` (PDF)
    et `_checkpoints` (ZIP) du répertoire `output_path` et les met à jour en temps réel.
    """
    plot_dir = os.path.join(output_path, "_validation/plots")
    checkpoint_dir = os.path.join(output_path, "_checkpoints")

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    seen_pdfs = set()
    seen_zips = set()

    while True:
        # Vérifie la présence des fichiers PDF
        current_pdfs = set(Path(plot_dir).glob("*.pdf"))
        new_pdfs = current_pdfs - seen_pdfs
        seen_pdfs.update(new_pdfs)

        # Vérifie la présence des fichiers ZIP
        current_zips = set(Path(checkpoint_dir).glob("*.zip"))
        new_zips = current_zips - seen_zips
        seen_zips.update(new_zips)

        # Met à jour les fichiers visibles dans l'interface Gradio
        if new_pdfs:
            pdf_output.update([str(pdf) for pdf in new_pdfs])
        if new_zips:
            zip_output.update([str(zip_file) for zip_file in new_zips])

        time.sleep(2)  # Délai pour limiter l'utilisation du CPU

def prepare_training_json_and_start(
    files,
    descriptions,
    seed,
    steps,
    guidance,
    quantize,
    width,
    height,
    num_epochs,
    batch_size,
    plot_frequency,
    generate_image_frequency,
    validation_prompt,
    pdf_output,
    zip_output
):
    if not files:
        return "Aucun fichier téléchargé pour l'entraînement."

    if len(files) != len(descriptions):
        return "Le nombre de descriptions ne correspond pas au nombre de fichiers."

    # Valeurs par défaut pour les paramètres absents
    default_json = {
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
            "path": "../../"+TEMP_IMAGES_DIR,
            "images": []
        }
    }

    # Mise à jour des paramètres avec les valeurs de l'interface
    training_json = default_json.copy()
    training_json.update({
        **({"seed": seed} if seed is not None else {}),
        **({"steps": steps} if steps is not None else {"steps": default_json["steps"]}),
        **({"guidance": guidance} if guidance is not None else {"guidance": default_json["guidance"]}),
        **({"quantize": None} if quantize == "No quantization" else ({"quantize": int(quantize)} if quantize else {})),
        **({"width": width} if width is not None else {"width": default_json["width"]}),
        **({"height": height} if height is not None else {"height": default_json["height"]}),
        "training_loop": {
            **({"num_epochs": num_epochs} if num_epochs is not None else {"num_epochs": default_json["training_loop"]["num_epochs"]}),
            **({"batch_size": batch_size} if batch_size is not None else {"batch_size": default_json["training_loop"]["batch_size"]}),
        },
        "instrumentation": {
            **({"plot_frequency": plot_frequency} if plot_frequency is not None else {"plot_frequency": default_json["instrumentation"]["plot_frequency"]}),
            **({"generate_image_frequency": generate_image_frequency} if generate_image_frequency is not None else {"generate_image_frequency": default_json["instrumentation"]["generate_image_frequency"]}),
            **({"validation_prompt": validation_prompt} if validation_prompt else {"validation_prompt": default_json["instrumentation"]["validation_prompt"]}),
        },
    })

    # Déplacer les fichiers dans le répertoire temporaire et ajouter au JSON
    for file, description in zip(files, descriptions):
        temp_file_path = os.path.join(TEMP_IMAGES_DIR, Path(file.name).name)
        shutil.copy(file.name, temp_file_path)
        training_json["examples"]["images"].append({
            "image": Path(temp_file_path).name,
            "prompt": description
        })

    # Sauvegarder le fichier JSON dans un répertoire fixe
    train_dir = os.path.join(TEMP_TRAIN_DIR, "current_train")
    os.makedirs(train_dir, exist_ok=True)
    json_path = os.path.join(train_dir, "train.json")
    training_json["save"]["output_path"] = train_dir

    with open(json_path, "w") as json_file:
        json.dump(training_json, json_file, indent=4)

    # Lancer l'entraînement avec DreamBooth
    try:
        flux, runtime_config, training_spec, training_state = DreamBoothInitializer.initialize(
            config_path=json_path,
            checkpoint_path=None
        )


        monitor_thread = threading.Thread(
            target=monitor_training_directory, args=(training_spec.saver.output_path, pdf_output, zip_output)
        )
        monitor_thread.start()

        DreamBooth.train(
            flux=flux,
            runtime_config=runtime_config,
            training_spec=training_spec,
            training_state=training_state
        )

        monitor_thread.join()
        return f"Entraînement terminé avec succès !\nConfiguration JSON sauvegardée dans {json_path}"
    except StopTrainingException as stop_exc:
        training_state.save(training_spec)
        return f"Entraînement arrêté prématurément : {str(stop_exc)}"
    except Exception as e:
        return f"Erreur inattendue lors de l'entraînement : {str(e)}"
    

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

    # Onglet pour l'entraînement
    with gr.Tab("Train"):
        gr.Markdown("## Entraîner un modèle avec vos fichiers")
        gr.Markdown("Téléchargez plusieurs images ci-dessous et ajoutez une description pour chacune.")

        # Composants pour les fichiers, gallery et descriptions
        train_files = gr.Files(label="Fichiers d'entraînement", file_count="multiple", type="filepath")
        gallery = gr.Gallery(label="Images téléchargées", columns=3, height="auto", object_fit="contain")
        description_textbox = gr.Textbox(
            label="Description pour l'image sélectionnée",
            placeholder="Entrez une description ici...",
            lines=2
        )
        descriptions_state = gr.State([])  # Stocke les descriptions associées
        selected_image_index = gr.Number(label="Index de l'image sélectionnée", value=0, visible=False)

        # Mise à jour dynamique de la Gallery et des descriptions
        def update_gallery_and_descriptions(files):
            if not files:
                return [], [], "Aucun fichier chargé."
            images = [file.name for file in files]
            descriptions = [""] * len(images)  # Initialise des descriptions vides
            return images, descriptions, ""

        train_files.change(
            fn=update_gallery_and_descriptions,
            inputs=train_files,
            outputs=[gallery, descriptions_state, description_textbox]
        )

        # Mise à jour de la description pour l'image sélectionnée
        def update_description(index, new_description, descriptions):
            descriptions[int(index)] = new_description
            return descriptions

        description_textbox.change(
            fn=update_description,
            inputs=[selected_image_index, description_textbox, descriptions_state],
            outputs=descriptions_state
        )

        # Sélection d'une image dans la Gallery
        def select_image(evt: gr.SelectData, descriptions):
            index = evt.index
            current_description = descriptions[index]
            return index, current_description

        gallery.select(
            fn=select_image,
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
demo.queue().launch()