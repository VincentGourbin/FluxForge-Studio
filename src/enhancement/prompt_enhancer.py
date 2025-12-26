# prompt_enhancer.py
"""
Ollama integration module for prompt enhancement with support for vision models.

This module provides functionality to enhance prompts for FLUX image generation
using various Ollama models, including vision-capable models that can analyze
images and text-only models for prompt refinement.
"""

import asyncio
import threading
import queue
import ollama
from ollama import AsyncClient
import gradio as gr

# Global model information storage
models_info = {}
model_names = []

def _initialize_ollama_models():
    """
    Initialize and retrieve available Ollama models with their capabilities.
    
    Populates the global models_info dictionary with model names as keys
    and their family capabilities as values. Also creates a list of model names.
    
    Globals:
        models_info (dict): Dictionary mapping model names to their families
        model_names (list): List of available model names
    """
    global models_info, model_names
    
    try:
        ollama_response = ollama.list()
        models_info = {}
        
        # Handle different response formats (newer versions might return object)
        if hasattr(ollama_response, 'models'):
            models_list = ollama_response.models
        elif isinstance(ollama_response, dict) and 'models' in ollama_response:
            models_list = ollama_response['models']
        else:
            models_list = ollama_response if isinstance(ollama_response, list) else []
        
        for model in models_list:
            # Handle different model object formats
            name = None
            if hasattr(model, 'model'):  # New ollama format uses 'model' attribute
                name = model.model
            elif hasattr(model, 'name'):  # Legacy format
                name = model.name
            elif isinstance(model, dict):
                name = model.get('name', '') or model.get('model', '')
            
            if not name:
                continue
            
            # Get detailed model information using ollama.show()
            try:
                model_details = ollama.show(name)
                capabilities = model_details.get('capabilities', [])
                
                # Store capabilities instead of families
                # We're particularly interested in 'vision' capability
                models_info[name] = capabilities
                
            except Exception as e:
                # Fallback to families if show() fails
                families = []
                if hasattr(model, 'details') and hasattr(model.details, 'families'):
                    families = model.details.families
                models_info[name] = families
        
        model_names = list(models_info.keys())
    except Exception as e:
        models_info = {}
        model_names = []
        print(f"Error retrieving Ollama models: {e}")

def _run_async_loop(loop, coro):
    """
    Helper function to run async coroutines in a separate thread.
    
    Args:
        loop (asyncio.AbstractEventLoop): The event loop to use
        coro (coroutine): The coroutine to execute
    """
    asyncio.set_event_loop(loop)
    loop.run_until_complete(coro)

def _async_to_sync_bridge(async_generator_func):
    """
    Bridge async generators to sync generators using threading and queues.
    
    Args:
        async_generator_func (callable): Async generator function to execute
        
    Yields:
        Any: Values yielded by the async generator
    """
    output_queue = queue.Queue()
    
    async def async_runner():
        try:
            async for output in async_generator_func():
                output_queue.put(output)
        except Exception as e:
            output_queue.put(f"An error occurred: {e}")
        finally:
            output_queue.put(None)  # Signal completion
    
    # Run async function in separate thread
    new_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=_run_async_loop, args=(new_loop, async_runner()))
    thread.start()
    
    # Yield results as they become available
    while True:
        output = output_queue.get()
        if output is None:
            break
        yield output

# Initialize models on module import
_initialize_ollama_models()

def enhance_prompt(selected_model, input_text, input_image):
    """
    Enhance prompts for FLUX image generation using Ollama models.
    
    This function handles both vision-capable models (that can analyze images)
    and text-only models for prompt enhancement. It validates inputs and
    routes to appropriate processing based on model capabilities.
    
    Args:
        selected_model (str): Name of the Ollama model to use
        input_text (str): Text input for prompt enhancement
        input_image (str|None): Path to image file for vision models
        
    Yields:
        str: Enhanced prompt text or error messages as they are generated
        
    Returns:
        None: Function is a generator that yields results
        
    Raises:
        Yields error messages instead of raising exceptions
    """
    # Get model capabilities
    families = models_info.get(selected_model, [])
    
    # Check if model supports vision based on capabilities
    is_vision_model = 'vision' in families  # families now contains capabilities
    
    # Validate inputs based on model requirements
    if not selected_model:
        yield "Please select a model."
        return
    
    # For vision models, require an image
    if is_vision_model and not input_image:
        yield "Please provide an image for this vision model."
        return
    
    # For text models, require text input
    if not is_vision_model and not input_text:
        yield "Please enter text to enhance."
        return

    # Handle vision-capable models (models that accept image input)
    if is_vision_model:
        if input_image is not None:
            # Define the image analysis prompt for FLUX optimization
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
                """
                Analyze image using vision-capable Ollama model.
                
                Creates appropriate message format based on model family
                (mllama uses 'images' array, others use 'image' field).
                
                Yields:
                    str: Accumulated analysis content as it's generated
                """
                client = AsyncClient()

                # Format message based on model family (check if model supports specific image format)
                # Most vision models use the 'images' array, some older ones use 'image' field
                if any(keyword in selected_model.lower() for keyword in ['llama3.2-vision', 'mllama', 'vision']):
                    # Modern vision models typically use 'images' array
                    messages = [{
                        'role': 'user',
                        'content': analysis_prompt,
                        'images': [input_image]
                    }]
                else:
                    # Fallback to 'image' field for older models
                    messages = [{
                        'role': 'user',
                        'content': analysis_prompt,
                        'image': input_image
                    }]

                # Stream response and accumulate content
                content = ""
                async for part in await client.chat(model=selected_model, messages=messages, stream=True):
                    delta = part['message']['content']
                    content += delta
                    yield content

            # Use the async-to-sync bridge to handle the image analysis
            yield from _async_to_sync_bridge(analyze_image)

        else:
            yield "Please provide an image for this model."
            return
    else:
        # Handle text-only models (models without image input capability)
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

        # Construct the full prompt for the LLM
        prompt_for_llm = f"{guide_instructions}\n\nUser input: \"{input_text}\"\n\nGenerated prompt:"

        async def run_chat():
            """
            Process text-only prompt enhancement using Ollama model.
            
            Yields:
                str: Accumulated enhanced prompt content as it's generated
            """
            client = AsyncClient()
            message = {'role': 'user', 'content': prompt_for_llm}
            content = ""
            
            # Stream response and accumulate content
            async for part in await client.chat(model=selected_model, messages=[message], stream=True):
                delta = part['message']['content']
                content += delta
                yield content

        # Use the async-to-sync bridge to handle the text processing
        yield from _async_to_sync_bridge(run_chat)

def update_image_input_visibility(selected_model):
    """
    Update the visibility of image input component based on model capabilities.
    
    Vision-capable models (those with 'vision' capability)
    require image input, so the image upload component should be visible.
    
    Args:
        selected_model (str): Name of the selected Ollama model
        
    Returns:
        gr.update: Gradio update object to control component visibility
    """
    capabilities = models_info.get(selected_model, [])
    if 'vision' in capabilities:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def update_button_label(selected_model):
    """
    Update the button label based on model capabilities.
    
    Vision-capable models get "Analyser l'image" (Analyze image) label,
    while text-only models get "Améliorer le prompt" (Enhance prompt) label.
    
    Args:
        selected_model (str): Name of the selected Ollama model
        
    Returns:
        gr.update: Gradio update object to change button label
    """
    capabilities = models_info.get(selected_model, [])
    if 'vision' in capabilities:
        # Vision-capable model that accepts image input
        return gr.update(value="Analyze Image")
    else:
        # Text-only model for prompt enhancement
        return gr.update(value="Enhance Prompt")