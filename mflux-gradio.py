import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import gradio as gr
from mflux import Flux1, Flux1Controlnet, Config, ConfigControlnet, ModelConfig
import datetime
from pathlib import Path
import os
import random
import json
import sqlite3
import time
from PIL import Image
import asyncio
import ollama
from ollama import AsyncClient
import threading
import queue


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
    progress=gr.Progress(),
    *args
):
    global current_model_alias, current_quantize, current_path, current_lora_paths, current_lora_scales, flux_model, current_model_type

    # Création du dossier pour les images intermédiaires
    stepwise_dir = Path("stepwise_output")
    stepwise_dir.mkdir(parents=True, exist_ok=True)
    
    # Nettoyer le dossier des images précédentes
    for file in stepwise_dir.glob("*.png"):
        try:
            os.remove(file)
        except:
            pass

    # Traitement des paramètres (garder le code existant pour les paramètres)
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
            prompt = f"{lora_info['activation_keyword']}, {prompt}"


    print(prompt)
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
            stepwise_output_dir=stepwise_dir
        )
    else:
        image = flux_model.generate_image(
            seed=seed,
            prompt=prompt,
            config=config,
            stepwise_output_dir=stepwise_dir
        )
        
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

def enhance_prompt(selected_model, input_text, input_image):
    families = models_info.get(selected_model, [])
    if not selected_model or ((len(families) > 1 or 'clip' in families) and not input_image) or (not input_text and not input_image):
        yield "Veuillez sélectionner un modèle et saisir du texte ou insérer une image."
        return

    # Gérer les modèles qui acceptent une image en entrée
    if (len(families) > 1 or 'clip' in families):
        if input_image is not None:
            # Étape 1 : Analyser l'image et générer une description
            analysis_prompt = """
            You are a prompt creation assistant for FLUX, an AI image generation model. Your mission is to help the user craft a detailed and optimized prompt by following these steps:

            Analyze this picture and make a description in order to generate a similar image with my FLUX.1 diffusion model.

            Be very explicit on the different graphics styles.

            1. **Understanding the images's details**:
                - Be very explicit on the different graphics styles (photorealist, comics, manga, drawing .....).
                - Be very explicit on the different color schemes

            2. **Enhancing Details**:
                - Enrich the basic idea with vivid, specific, and descriptive elements.
                - Include factors such as lighting, mood, style, perspective, and specific objects or elements the user wants in the scene.

            3. **Formatting the Prompt**:
                - Structure the enriched description into a clear, precise, and effective prompt.
                - Ensure the prompt is tailored for high-quality output from the FLUX model, considering its strengths (e.g., photorealistic details, fine anatomy, or artistic styles).

            4. **Translations (if necessary)**:
                - If the user provides a request in another language, translate it into English for the prompt and transcribe it back into their language for clarity.

            Use this process to compose a detailed and coherent prompt. Ensure the final prompt is clear and complete, and write your response in English.

            Ensure that the final part is a synthesized version of the prompt that I can use in FLUX         
            """

            async def analyze_image():
                client = AsyncClient()

                if 'mllama' in families:
                    # Préparer les messages avec l'image
                    messages = [{
                        'role': 'user',
                        'content': analysis_prompt,
                        'images': [input_image]
                    }]
                else:
                    # Préparer les messages avec l'image
                    messages = [{
                        'role': 'user',
                        'content': analysis_prompt,
                        'image': input_image
                    }]

                # Initialize content accumulator
                content = ""
                # Async iteration over the streamed response
                async for part in await client.chat(model=selected_model, messages=messages, stream=True):
                    delta = part['message']['content']
                    content += delta
                    yield content  # Yield the accumulated content

            # Fonction pour exécuter le traitement asynchrone
            async def process_prompts():
                try:
                    # Analyser l'image et obtenir la description
                    async for generated_description in analyze_image():
                        output_queue.put(generated_description)
                except Exception as e:
                    output_queue.put(f"An error occurred: {e}")
                finally:
                    output_queue.put(None)  # Indiquer la fin du processus

            # Utiliser une file d'attente pour communiquer entre l'async et le sync
            output_queue = queue.Queue()

            def run_async_loop(loop, coro):
                asyncio.set_event_loop(loop)
                loop.run_until_complete(coro)

            # Démarrer la fonction asynchrone dans un thread séparé
            new_loop = asyncio.new_event_loop()
            t = threading.Thread(target=run_async_loop, args=(new_loop, process_prompts()))
            t.start()

            while True:
                output = output_queue.get()
                if output is None:
                    break
                yield output

        else:
            yield "Veuillez fournir une image pour ce modèle."
            return
    else:
        # Modèles sans image en entrée
        guide_instructions = """
        You are a prompt creation assistant for FLUX, an AI image generation model. Your mission is to help the user craft a detailed and optimized prompt by following these steps:

        1. **Understanding the User's Needs**:
            - The user provides a basic idea, concept, or description.
            - Analyze their input to determine essential details and nuances.

        2. **Enhancing Details**:
            - Enrich the basic idea with vivid, specific, and descriptive elements.
            - Include factors such as lighting, mood, style, perspective, and specific objects or elements the user wants in the scene.

        3. **Formatting the Prompt**:
            - Structure the enriched description into a clear, precise, and effective prompt.
            - Ensure the prompt is tailored for high-quality output from the FLUX model, considering its strengths (e.g., photorealistic details, fine anatomy, or artistic styles).

        4. **Translations (if necessary)**:
            - If the user provides a request in another language, translate it into English for the prompt and transcribe it back into their language for clarity.

        Use this process to compose a detailed and coherent prompt. Ensure the final prompt is clear and complete, and write your response in English.

        Ensure that the final part is a synthesized version of the prompt.
        """

        prompt_for_llm = f"{guide_instructions}\n\nUser input: \"{input_text}\"\n\nGenerated prompt:"

        async def run_chat():
            client = AsyncClient()
            message = {'role': 'user', 'content': prompt_for_llm}
            content = ""
            # Attendre la coroutine client.chat() avant de l'utiliser dans async for
            async for part in await client.chat(model=selected_model, messages=[message], stream=True):
                delta = part['message']['content']
                content += delta
                yield content

        output_queue = queue.Queue()

        def run_async_loop(loop, coro):
            asyncio.set_event_loop(loop)
            loop.run_until_complete(coro)

        async def async_runner():
            try:
                async for output in run_chat():
                    output_queue.put(output)
            finally:
                output_queue.put(None)

        new_loop = asyncio.new_event_loop()
        t = threading.Thread(target=run_async_loop, args=(new_loop, async_runner()))
        t.start()

        while True:
            output = output_queue.get()
            if output is None:
                break
            yield output

# Fonction pour synthétiser le prompt
def synthesize_prompt(enhanced_text, selected_model):
    if not enhanced_text or not selected_model:
        yield "Veuillez fournir un texte à synthétiser et sélectionner un modèle."
        return

    # Prompt pour la synthétisation
    synthesis_prompt = f"As an assistant, synthesize this prompt generation in 5 lines:\n\n{enhanced_text}"

    async def run_chat():
        client = AsyncClient()
        message = {'role': 'user', 'content': synthesis_prompt}
        content = ""
        async for part in await client.chat(model=selected_model, messages=[message], stream=True):
            delta = part['message']['content']
            content += delta
            yield content

    output_queue = queue.Queue()

    def run_async_loop(loop, coro):
        asyncio.set_event_loop(loop)
        loop.run_until_complete(coro)

    async def async_runner():
        try:
            async for output in run_chat():
                output_queue.put(output)
        finally:
            output_queue.put(None)

    new_loop = asyncio.new_event_loop()
    t = threading.Thread(target=run_async_loop, args=(new_loop, async_runner()))
    t.start()

    while True:
        output = output_queue.get()
        if output is None:
            break
        yield output



# Fonction pour mettre à jour la visibilité de la zone de dépôt d'image
def update_image_input_visibility(selected_model):
    families = models_info.get(selected_model, [])
    if len(families) > 1 or 'clip' in families:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def update_button_label(selected_model):
    families = models_info.get(selected_model, [])
    if len(families) > 1 or 'clip' in families:
        # Modèle qui accepte une image en entrée
        return gr.update(value="Analyser l'image")
    else:
        return gr.update(value="Améliorer le prompt")

def remove_background(input_image):
    image = input_image.convert("RGB")
    input_images = transform_image(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        preds = modelbgrm(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)
    
    return image

# Définir l'interface Gradio
model_options = ["schnell", "dev"]
quantize_options = [4, 8, None]

# Obtenir la liste des modèles disponibles avec ollama
try:
    ollama_models = ollama.list()
    models_info = {}
    for model in ollama_models['models']:
        name = model['name']
        families = model['details'].get('families', [])
        models_info[name] = families
    model_names = list(models_info.keys())
except Exception as e:
    models_info = {}
    model_names = []
    print(f"Erreur lors de la récupération des modèles ollama : {e}")


modelbgrm = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
torch.set_float32_matmul_precision(['high', 'highest'][0])
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
modelbgrm.to(device)
modelbgrm.eval()

image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


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
            fn=remove_background,
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