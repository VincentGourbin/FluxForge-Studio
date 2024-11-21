# prompt_enhancer.py

import asyncio
import threading
import queue
import ollama
from ollama import AsyncClient
import gradio as gr

# Obtenir la liste des modèles disponibles avec Ollama
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
    print(f"Erreur lors de la récupération des modèles Ollama : {e}")

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

                # Accumulateur de contenu
                content = ""
                # Itération asynchrone sur la réponse en streaming
                async for part in await client.chat(model=selected_model, messages=messages, stream=True):
                    delta = part['message']['content']
                    content += delta
                    yield content  # Renvoie le contenu accumulé

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
            # Itération asynchrone sur la réponse en streaming
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