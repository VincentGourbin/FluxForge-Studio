import gradio as gr
from mflux import Flux1, Flux1Controlnet, Config, ConfigControlnet, ModelConfig
import datetime
from pathlib import Path
import os
import random
import json
import sqlite3

# Variables globales pour stocker le modèle chargé
current_model_alias = None
current_quantize = None
current_path = None
current_lora_paths = None
current_lora_scales = None
current_model_type = None  # 'standard' ou 'controlnet'
flux_model = None
lora_data = []  # Stockera les informations des LoRA

# Chemin vers le dossier contenant les fichiers LoRA
lora_directory = 'lora'

# Chemin vers le fichier JSON contenant les informations des LoRA
lora_json_file = 'lora_info.json'

# Lire les informations des LoRA depuis le fichier JSON
with open(lora_json_file, 'r') as f:
    lora_data = json.load(f)

# Chemin vers la base de données SQLite
db_path = 'generated_images.db'

# Fonction pour initialiser la base de données
def init_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Création de la table pour stocker les images et les paramètres
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            seed INTEGER,
            prompt TEXT,
            model_alias TEXT,
            quantize INTEGER,
            steps INTEGER,
            guidance REAL,
            height INTEGER,
            width INTEGER,
            path TEXT,
            controlnet_image_path TEXT,
            controlnet_strength REAL,
            controlnet_save_canny BOOLEAN,
            lora_paths TEXT,
            lora_scales TEXT,
            output_filename TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Appel de la fonction pour initialiser la base de données
init_db()

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
    *args  # Ceci capturera les valeurs des cases à cocher et des scales pour les LoRA
):
    global current_model_alias, current_quantize, current_path, current_lora_paths, current_lora_scales, flux_model, current_model_type

    # Traitement du seed
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
    num_lora = len(lora_data)
    lora_checkbox_values = args[:num_lora]
    lora_scale_values = args[num_lora:2*num_lora]

    lora_paths_list = []
    lora_scales_list = []
    for idx, (selected, scale) in enumerate(zip(lora_checkbox_values, lora_scale_values)):
        if selected:
            lora_info = lora_data[idx]
            lora_file = lora_info['file_name']
            lora_path = os.path.join(lora_directory, lora_file)
            lora_paths_list.append(lora_path)
            lora_scales_list.append(float(scale))
            # Ajouter le mot-clé d'activation au prompt
            prompt += f" {lora_info['activation_keyword']}"

    # Déterminer si ControlNet est utilisé
    use_controlnet = controlnet_image_path is not None

    # Vérifier si le modèle doit être rechargé
    if (model_alias != current_model_alias) or (quantize != current_quantize) or (path != current_path) or \
        (lora_paths_list != current_lora_paths) or (lora_scales_list != current_lora_scales) or (flux_model is None) or \
        (use_controlnet != (current_model_type == 'controlnet')):

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
        current_model_alias = model_alias
        current_quantize = quantize
        current_path = path
        current_lora_paths = lora_paths_list
        current_lora_scales = lora_scales_list

    # Construire la configuration
    if use_controlnet:
        config = ConfigControlnet(
            num_inference_steps=steps,
            guidance=guidance,
            height=height,
            width=width,
            controlnet_strength=controlnet_strength,
        )
    else:
        config = Config(
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
            config=config,
        )
    else:
        image = flux_model.generate_image(
            seed=seed,
            prompt=prompt,
            config=config,
        )
        # Sauvegarder l'image avec les métadonnées si demandé
        image.save(path=str(output_filename), export_json_metadata=metadata)

    # Enregistrer les paramètres et le chemin de l'image dans la base de données
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO images (
            timestamp, seed, prompt, model_alias, quantize, steps, guidance,
            height, width, path, controlnet_image_path, controlnet_strength,
            controlnet_save_canny, lora_paths, lora_scales, output_filename
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
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
    conn.commit()
    conn.close()

    # Retourner l'image pour affichage dans Gradio
    return image.image  # Retourner l'image PIL

def update_guidance_visibility(model_alias):
    if model_alias == "dev":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

# Fonction pour charger l'historique depuis la base de données
def load_history():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT id, output_filename FROM images ORDER BY timestamp DESC')
    records = cursor.fetchall()
    conn.close()

    images = []
    for record in records:
        image_id, output_filename = record
        if os.path.exists(output_filename):
            images.append(output_filename)
    return images

def show_image_details(evt: gr.SelectData, selected_image_index):
    index = evt.index
    selected_image_index = index

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM images ORDER BY timestamp DESC')
    records = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    conn.close()

    if 0 <= index < len(records):
        record = records[index]
        details = dict(zip(columns, record))
        # Formater les détails pour affichage
        details_text = '\n'.join([f"{key}: {value}" for key, value in details.items()])
        return details_text, selected_image_index
    else:
        return "Aucune information disponible pour cette image.", selected_image_index

def delete_selected_image(selected_image_index):
    if selected_image_index is None:
        return gr.update(value="Veuillez sélectionner une image à supprimer."), gr.update(), selected_image_index

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM images ORDER BY timestamp DESC')
    records = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    if 0 <= selected_image_index < len(records):
        record = records[selected_image_index]
        image_id = record[columns.index('id')]
        output_filename = record[columns.index('output_filename')]

        # Supprimer le fichier image du système de fichiers
        if os.path.exists(output_filename):
            os.remove(output_filename)

        # Supprimer l'enregistrement de la base de données
        cursor.execute('DELETE FROM images WHERE id = ?', (image_id,))
        conn.commit()
        conn.close()

        # Réinitialiser l'index sélectionné
        selected_image_index = None

        # Rafraîchir la galerie
        images = load_history()

        # Retourner le message de succès, la galerie mise à jour et l'index réinitialisé
        return "Image supprimée avec succès.", images, selected_image_index
    else:
        conn.close()
        return "Aucune image correspondante trouvée.", gr.update(), selected_image_index

# Fonction pour mettre à jour l'historique
def refresh_history():
    images = load_history()
    return images


# Définir l'interface Gradio
model_options = ["schnell", "dev"]
quantize_options = [4, 8, None]

with gr.Blocks() as demo:
    gr.Markdown("# Générateur d'images mflux")

    with gr.Tab("Génération"):
        prompt = gr.Textbox(label="Prompt (description de l'image)", value="Luxury food photograph", lines=2)

        gr.Markdown("## Paramètres principaux")

        with gr.Row():
            model_alias = gr.Dropdown(label="Modèle", choices=model_options, value="schnell")
            quantize = gr.Dropdown(label="Quantize", choices=[str(q) for q in quantize_options], value="8")
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
            for lora in lora_data:
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

        btn.click(fn=generate_image, inputs=inputs, outputs=output_image)

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

demo.launch()