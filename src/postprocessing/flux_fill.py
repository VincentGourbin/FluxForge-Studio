"""
FLUX Fill Post-Processing Module

Provides inpainting and outpainting capabilities using FLUX.1-Fill-dev model.
Supports both manual mask creation (inpainting) and automatic expansion (outpainting).

Features:
- Inpainting: Paint over areas to regenerate with user-drawn masks
- Outpainting: Extend image boundaries with percentage-based expansion
- LoRA support for style enhancement
- Automatic mask generation and preview
- Memory-efficient model loading

Author: MFLUX Team
"""

import os
import datetime
import random
import json
import torch
import gc
import warnings
from pathlib import Path
from PIL import Image
import numpy as np
import gradio as gr
from utils.progress_tracker import global_progress_tracker

def create_outpainting_mask(image, top_percent, bottom_percent, left_percent, right_percent):
    """
    Create expanded image and mask for outpainting operation.
    
    Args:
        image (PIL.Image): Original image to expand
        top_percent (float): Top expansion percentage (0-100)
        bottom_percent (float): Bottom expansion percentage (0-100)
        left_percent (float): Left expansion percentage (0-100)
        right_percent (float): Right expansion percentage (0-100)
        
    Returns:
        tuple: (expanded_image, mask) where mask has BLACK=keep, WHITE=fill
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Calculate expansion pixels
    orig_width, orig_height = image.size
    left_pixels = int(orig_width * left_percent / 100)
    right_pixels = int(orig_width * right_percent / 100)
    top_pixels = int(orig_height * top_percent / 100)
    bottom_pixels = int(orig_height * bottom_percent / 100)
    
    # Calculate new dimensions
    new_width = orig_width + left_pixels + right_pixels
    new_height = orig_height + top_pixels + bottom_pixels
    
    
    # Create expanded canvas (black background)
    expanded_image = Image.new('RGB', (new_width, new_height), (0, 0, 0))
    
    # Paste original image in the center
    paste_x = left_pixels
    paste_y = top_pixels
    expanded_image.paste(image, (paste_x, paste_y))
    
    # Create mask: BLACK for original image area, WHITE for expansion areas
    mask = Image.new('L', (new_width, new_height), 255)  # White background (fill areas)
    
    # Create black rectangle for original image area (keep area)
    mask_array = np.array(mask)
    mask_array[paste_y:paste_y + orig_height, paste_x:paste_x + orig_width] = 0  # Black = keep
    mask = Image.fromarray(mask_array)
    
    return expanded_image, mask

def generate_flux_fill_preview(mode, image_editor_data, outpaint_image, top_percent, bottom_percent, left_percent, right_percent):
    """
    Generate preview mask for FLUX Fill operation without running the model.
    
    Args:
        mode (str): "Inpainting" or "Outpainting"
        image_editor_data: Data from ImageEditor for inpainting
        outpaint_image: Image for outpainting
        top_percent, bottom_percent, left_percent, right_percent: Outpainting percentages
        
    Returns:
        PIL.Image: Preview mask showing BLACK=keep, WHITE=fill areas
    """
    try:
        if mode == "Inpainting":
            if image_editor_data is None:
                return None
                
            if 'background' not in image_editor_data:
                return None
                
            base_image = image_editor_data['background']
            if base_image is None:
                return None
            
            # Extract inpainting mask from ImageEditor
            mask = None
            
            # Try to get mask from composite (preferred method)
            composite = image_editor_data.get('composite')
            if composite and base_image:
                bg_array = np.array(base_image.convert('RGB'))
                comp_array = np.array(composite.convert('RGB'))
                
                # Find pixels that changed (where user drew)
                diff = np.sum(np.abs(bg_array.astype(int) - comp_array.astype(int)), axis=2)
                mask_array = (diff > 30).astype(np.uint8) * 255  # White where user drew
                
                if np.sum(mask_array) > 0:
                    mask = Image.fromarray(mask_array, mode='L')
            
            # Fallback: try to extract from layers
            if mask is None and 'layers' in image_editor_data:
                layers = image_editor_data.get('layers', [])
                if layers:
                    mask_array = np.zeros((base_image.height, base_image.width), dtype=np.uint8)
                    
                    for layer in layers:
                        if layer and hasattr(layer, 'size'):
                            layer_array = np.array(layer.convert('L'))
                            mask_array = np.maximum(mask_array, layer_array)
                    
                    if np.sum(mask_array) > 0:
                        mask = Image.fromarray(mask_array, mode='L')
            
            return mask
            
        elif mode == "Outpainting":
            if outpaint_image is None:
                return None
                
            # Create outpainting mask
            _, mask = create_outpainting_mask(
                outpaint_image, top_percent, bottom_percent, left_percent, right_percent
            )
            return mask
            
    except Exception as e:
        print(f"Error generating preview: {e}")
        return None

def process_flux_fill(fill_mode, image_editor_data, outpaint_image, prompt, steps, guidance_scale, quantization,
                     top_percent, bottom_percent, left_percent, right_percent, 
                     flux_fill_selected_lora_state, flux_fill_lora_strength_1, 
                     flux_fill_lora_strength_2, flux_fill_lora_strength_3, image_generator):
    """
    Process inpainting or outpainting using FLUX.1-Fill-dev model.
    
    Args:
        fill_mode: "Inpainting" or "Outpainting"
        image_editor_data: Data from ImageEditor for inpainting
        outpaint_image: Image for outpainting
        prompt: Text prompt for generation
        steps: Number of inference steps
        guidance_scale: Guidance scale for generation
        quantization: "None", "8-bit", or "Auto" for memory optimization
        top_percent, bottom_percent, left_percent, right_percent: Outpainting percentages
        flux_fill_selected_lora_state: List of selected LoRA models
        flux_fill_lora_strength_1/2/3: LoRA strength values
        image_generator: Reference to ImageGenerator instance
        
    Returns:
        PIL.Image: Generated result
    """
    try:
        from core.database import save_image_info
        
        print("=" * 80)
        print("🎨 FLUX FILL PROCESSING - PARAMETERS")
        print("=" * 80)
        print(f"📐 Mode: {fill_mode}")
        print(f"📝 Prompt: {prompt}")
        print(f"🔄 Steps: {steps}")
        print(f"🎚️ Guidance Scale: {guidance_scale}")
        
        if not prompt or prompt.strip() == "":
            print("❌ Please provide a prompt for the fill operation")
            return None
        
        # Prepare image and mask based on mode
        if fill_mode == "Inpainting":
            if image_editor_data is None:
                print("❌ No image editor data provided for inpainting")
                return None
                
            # Extract base image and mask from ImageEditor
            if 'background' not in image_editor_data:
                print("❌ No background image found in editor data")
                return None
                
            base_image = image_editor_data['background'].convert('RGB')
            
            # Extract inpainting mask: BLACK base + WHITE where user drew
            mask = None
            
            # First try to get mask from composite (preferred method)
            composite = image_editor_data.get('composite')
            if composite and base_image:
                bg_array = np.array(base_image.convert('RGB'))
                comp_array = np.array(composite.convert('RGB'))
                
                # Find pixels that changed (where user drew)
                diff = np.sum(np.abs(bg_array.astype(int) - comp_array.astype(int)), axis=2)
                mask_array = (diff > 30).astype(np.uint8) * 255  # White where user drew
                
                if np.sum(mask_array) > 0:
                    mask = Image.fromarray(mask_array, mode='L')
            
            # Fallback: try to extract from layers
            if mask is None and 'layers' in image_editor_data:
                layers = image_editor_data.get('layers', [])
                if layers:
                    mask_array = np.zeros((base_image.height, base_image.width), dtype=np.uint8)
                    
                    for layer in layers:
                        if layer and hasattr(layer, 'size'):
                            layer_array = np.array(layer.convert('L'))
                            mask_array = np.maximum(mask_array, layer_array)
                    
                    if np.sum(mask_array) > 0:
                        mask = Image.fromarray(mask_array, mode='L')
            
            if mask is None:
                print("❌ No valid mask found for inpainting - please draw on the image")
                return None
            
            # Prepare for FLUX Fill
            input_image = base_image
            fill_mask = mask
            
        elif fill_mode == "Outpainting":
            if outpaint_image is None:
                print("❌ No image provided for outpainting")
                return None
                
            # Create expanded image and mask
            expanded_image, mask = create_outpainting_mask(
                outpaint_image.convert('RGB'), top_percent, bottom_percent, left_percent, right_percent
            )
            
            # Prepare for FLUX Fill
            input_image = expanded_image
            fill_mask = mask
            
        else:
            print("❌ Invalid fill mode")
            return None
        
        # Load FLUX.1-Fill-dev pipeline
        print("🔄 Loading FLUX.1-Fill-dev model...")
        try:
            from diffusers import FluxFillPipeline
        except ImportError:
            print("❌ FluxFillPipeline not available in this diffusers version")
            print("Please update diffusers: pip install -U diffusers")
            return None
        
        # Determine device and dtype
        device = image_generator.device
        dtype = image_generator.dtype
        
        print(f"🔧 Using device: {device}, dtype: {dtype}")
        
        # Initialize FLUX Fill pipeline
        flux_fill_pipeline = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            torch_dtype=dtype,
            use_safetensors=True
        )
        
        # Move pipeline to appropriate device
        flux_fill_pipeline = flux_fill_pipeline.to(device)
        
        # Enable memory efficient attention
        flux_fill_pipeline.enable_attention_slicing()
        
        # IMPORTANT: Quantization sera appliquée APRÈS le chargement des LoRA
        
        # Process selected LoRA from the state
        selected_loras = []
        adapter_names = []
        adapter_weights = []
        lora_strengths = [flux_fill_lora_strength_1, flux_fill_lora_strength_2, flux_fill_lora_strength_3]
        
        if flux_fill_selected_lora_state:
            for i, selected_lora in enumerate(flux_fill_selected_lora_state):
                if i < len(lora_strengths):
                    lora_file = selected_lora['name']
                    lora_path = os.path.join(image_generator.lora_directory, lora_file)
                    
                    if os.path.exists(lora_path):
                        strength = lora_strengths[i] if lora_strengths[i] is not None else 0.8
                        selected_loras.append((lora_path, selected_lora, float(strength)))
                        
                        # Prepend activation keyword to prompt for better LoRA activation
                        prompt = f"{selected_lora['activation_keyword']}, {prompt}"
        
        # Load selected LoRA models
        if selected_loras:
            print(f"🎨 Loading {len(selected_loras)} LoRA model(s)...")
            try:
                for lora_path, lora_info, lora_scale in selected_loras:
                    lora_dir = os.path.dirname(lora_path)
                    lora_filename = os.path.basename(lora_path)
                    adapter_name = os.path.splitext(lora_filename)[0].replace('.', '_')
                    
                    # Load LoRA with warning suppression and error handling
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message=".*CLIPTextModel.*")
                            warnings.filterwarnings("ignore", message=".*No LoRA keys associated to CLIPTextModel.*")
                            warnings.filterwarnings("ignore", message=".*Already found a.*peft_config.*attribute.*")
                            flux_fill_pipeline.load_lora_weights(
                                lora_dir, 
                                weight_name=lora_filename,
                                adapter_name=adapter_name
                            )
                            print(f"✅ LoRA Fill chargé: {lora_info['name']} (weight: {lora_scale})")
                    except KeyError as e:
                        print(f"❌ LoRA incompatible ignoré: {lora_info['name']}")
                        print(f"   Erreur: Paramètre manquant {e}")
                        print(f"   💡 Ce LoRA n'est pas compatible avec FLUX Fill")
                        continue  # Skip this LoRA
                    except Exception as e:
                        print(f"❌ Erreur chargement LoRA Fill: {lora_info['name']}")
                        print(f"   Erreur: {e}")
                        continue  # Skip this LoRA
                    
                    adapter_names.append(adapter_name)
                    adapter_weights.append(lora_scale)
                
                # Set all adapter weights
                if adapter_names:
                    flux_fill_pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
                    print(f"📝 Modified prompt: {prompt}")
                    
            except Exception as e:
                print(f"⚠️ Failed to load LoRA: {e}")
        else:
            print("ℹ️ No LoRA models selected")
        
        # Apply quantization AFTER LoRA loading to avoid parameter name conflicts
        if quantization and quantization != "None":
            from utils.quantization import quantize_pipeline_components
            
            # Apply same quantization logic as main models
            if quantization in ["8-bit", "Auto"]:
                print(f"🔧 Application quantification qint8 FLUX Fill APRÈS LoRA")
                success, error = quantize_pipeline_components(flux_fill_pipeline, device, prefer_4bit=False, verbose=True)
                if not success:
                    print(f"⚠️  Quantification qint8 échouée: {error}")
                    print("🔄 Continuons sans quantification...")
            elif quantization == "4-bit":
                print(f"⚠️  Quantification 4-bit non supportée sur {device} (tests montrent erreurs)")
                print("💡 Conseil: Utilisez '8-bit' pour économie mémoire")
                print("🔄 Continuons sans quantification...")
            else:
                print(f"⚠️  Quantification {quantization} non supportée")
                print("🔄 Continuons sans quantification...")
        
        print("🎨 Generating with FLUX.1-Fill-dev...")
        
        # Generate seed for reproducibility
        seed = random.randint(1, 2**32 - 1)
        generator = torch.Generator(device=device).manual_seed(seed)
        
        print(f"🎲 Using seed: {seed}")
        
        # Apply progress tracking for FLUX Fill
        print(f"🎨 Starting FLUX Fill generation with progress tracking...")
        
        # Reset and start progress tracking
        global_progress_tracker.reset()
        global_progress_tracker.apply_tqdm_patches()
        
        try:
            # Run FLUX Fill generation
            result = flux_fill_pipeline(
                prompt=prompt,
                image=input_image,
                mask_image=fill_mask,
                height=input_image.height,
                width=input_image.width,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                max_sequence_length=512,
                generator=generator
            )
        finally:
            # Always restore patches after generation
            global_progress_tracker.remove_tqdm_patches()
            print(f"✅ FLUX Fill generation completed with progress tracking")
        
        result_image = result.images[0]
        
        print("✅ FLUX Fill generation completed!")
        
        # Clean up memory
        del flux_fill_pipeline
        if device == 'cuda':
            torch.cuda.empty_cache()
        elif device == 'mps':
            torch.mps.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Save result
        if result_image:
            timestamp = datetime.datetime.now()
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            
            output_dir = Path("outputimage")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_filename = output_dir / f"flux_fill_{fill_mode.lower()}_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
            
            result_image.save(str(output_filename))
            
            # Prepare LoRA data for database storage
            lora_paths_for_db = []
            lora_scales_for_db = []
            if selected_loras:
                for lora_path, lora_info, lora_scale in selected_loras:
                    lora_paths_for_db.append(lora_path)
                    lora_scales_for_db.append(lora_scale)
            
            # Save to database
            from core.database import save_flux_fill_generation
            save_flux_fill_generation(
                timestamp_str,
                seed,
                prompt,
                fill_mode,
                steps,
                guidance_scale,
                result_image.height,
                result_image.width,
                lora_paths_for_db,
                lora_scales_for_db,
                str(output_filename)
            )
            
            print(f"💾 Image saved: {output_filename}")
            print(f"📊 Saved to database: {timestamp_str}")
            print("✅ FLUX FILL PROCESSING COMPLETED!")
            print("=" * 80)
        
        return result_image
        
    except Exception as e:
        print(f"❌ Error during FLUX Fill processing: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return None

def update_flux_fill_controls_visibility(processing_type):
    """
    Update visibility of post-processing controls based on selected type.
    
    Args:
        processing_type (str): Selected processing type
        
    Returns:
        tuple: Visibility updates for different control groups
    """
    # Handle None case explicitly
    if processing_type is None or processing_type == "None":
        return (
            gr.update(visible=False),  # flux_fill_group
            gr.update(visible=False),  # kontext_group
            gr.update(visible=False),  # flux_depth_group
            gr.update(visible=False),  # flux_canny_group
            gr.update(visible=False),  # flux_redux_group
            gr.update(visible=False),  # bg_removal_group
            gr.update(visible=False)   # upscaling_group
        )
    
    is_flux_fill = "FLUX Fill" in processing_type and processing_type != "None"
    is_kontext = "Kontext" in processing_type
    is_flux_depth = "FLUX Depth" in processing_type
    is_flux_canny = "FLUX Canny" in processing_type
    is_flux_redux = "FLUX Redux" in processing_type
    is_bg_removal = "Background Removal" in processing_type
    is_upscaling = "Upscaling" in processing_type
    
    return (
        gr.update(visible=is_flux_fill),     # flux_fill_group
        gr.update(visible=is_kontext),       # kontext_group
        gr.update(visible=is_flux_depth),    # flux_depth_group
        gr.update(visible=is_flux_canny),    # flux_canny_group
        gr.update(visible=is_flux_redux),    # flux_redux_group
        gr.update(visible=is_bg_removal),    # bg_removal_group
        gr.update(visible=is_upscaling)      # upscaling_group
    )

def update_flux_fill_mode_visibility(fill_mode):
    """
    Update visibility of inpainting/outpainting controls based on mode selection.
    
    Args:
        fill_mode (str): Selected fill mode ("Inpainting" or "Outpainting")
        
    Returns:
        tuple: Updates for (inpainting_group, outpainting_group) visibility
    """
    is_inpainting = fill_mode == "Inpainting"
    is_outpainting = fill_mode == "Outpainting"
    
    return (
        gr.update(visible=is_inpainting),   # inpainting_group
        gr.update(visible=is_outpainting)   # outpainting_group
    )

def generate_flux_fill_preview(fill_mode, image_editor_data, outpaint_image, top_percent, bottom_percent, left_percent, right_percent):
    """
    Generate preview of the mask for inpainting or outpainting.
    Shows the actual mask that will be used: BLACK=keep, WHITE=fill
    
    Returns:
        PIL.Image: Preview mask showing areas to be filled
    """
    try:
        if fill_mode == "Inpainting":
            if image_editor_data is None:
                return None
            
            # Extract background and layers from ImageEditor
            background = image_editor_data.get('background')
            layers = image_editor_data.get('layers', [])
            
            if not background:
                return None
                
            if not layers and not image_editor_data.get('composite'):
                # No drawing yet, return black mask (all areas to keep)
                return Image.new('L', background.size, color=0)
            
            # Create inpainting mask: BLACK base + WHITE where user drew
            # Start with black rectangle (areas to keep)
            mask_array = np.zeros((background.height, background.width), dtype=np.uint8)
            
            # Process layers to extract drawn areas
            for layer in layers:
                if layer and hasattr(layer, 'size'):
                    # Convert layer to mask
                    layer_array = np.array(layer.convert('L'))
                    # Add drawn areas to mask (white where drawn)
                    mask_array = np.maximum(mask_array, layer_array)
            
            # If there's a composite, use it preferentially
            composite = image_editor_data.get('composite')
            if composite:
                # Get difference between composite and background to find drawn areas
                bg_array = np.array(background.convert('RGB'))
                comp_array = np.array(composite.convert('RGB'))
                
                # Find pixels that changed (where user drew)
                diff = np.sum(np.abs(bg_array.astype(int) - comp_array.astype(int)), axis=2)
                mask_array = (diff > 30).astype(np.uint8) * 255  # White where user drew
            
            if np.sum(mask_array) == 0:
                # No drawing detected, return black mask (all areas to keep)
                black_mask = Image.new('L', background.size, color=0)
                return black_mask
            
            # Return the actual mask that will be passed to the model
            # BLACK = areas to keep, WHITE = areas to regenerate
            mask_image = Image.fromarray(mask_array, mode='L')
            return mask_image
            
        elif fill_mode == "Outpainting":
            if outpaint_image is None:
                return None
                
            expanded_image, mask = create_outpainting_mask(
                outpaint_image, top_percent, bottom_percent, left_percent, right_percent
            )
            
            if expanded_image and mask:
                # Return the actual mask that will be passed to the model
                # BLACK = areas to keep (original image), WHITE = areas to fill (expansion)
                return mask
            
            return None
            
    except Exception as e:
        return None