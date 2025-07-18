"""
FLUX Canny Post-Processing Module

Canny edge-guided image generation using FLUX.1-Canny-dev model.
Generates high-quality images guided by Canny edge maps with configurable detection parameters.

Features:
- Real-time Canny edge detection with adjustable thresholds
- FLUX.1-Canny-dev pipeline for edge-guided generation
- LoRA support for style enhancement
- Memory-efficient processing with device detection
- Preview generation for edge detection tuning

Author: MFLUX Team
"""

import os
import datetime
import random
import torch
import gc
import warnings
from pathlib import Path
from PIL import Image
import numpy as np
from utils.progress_tracker import global_progress_tracker

def generate_canny_preview(input_image, low_threshold=100, high_threshold=200):
    """
    Generate Canny edge preview for real-time UI feedback.
    
    Args:
        input_image (PIL.Image): Input image to process
        low_threshold (int): Lower threshold for edge detection
        high_threshold (int): Higher threshold for edge detection
        
    Returns:
        PIL.Image: Canny edge preview image or None if processing fails
    """
    try:
        if input_image is None:
            return None
        
        # Import canny processing utilities
        from utils.canny_processing import preprocess_canny
        
        # Generate Canny edges using existing utility
        canny_result = preprocess_canny(input_image, low_threshold, high_threshold)
        
        return canny_result
        
    except Exception as e:
        print(f"❌ Error generating Canny preview: {e}")
        return None

def process_flux_canny(input_image, prompt, steps, guidance_scale, quantization, low_threshold, high_threshold,
                      canny_selected_lora_state, canny_lora_strength_1, 
                      canny_lora_strength_2, canny_lora_strength_3, image_generator):
    """
    Process Canny edge-guided image generation using FLUX.1-Canny-dev model.
    
    Args:
        input_image (PIL.Image): Input image to extract edges from
        prompt (str): Text prompt for generation
        steps (int): Number of inference steps
        guidance_scale (float): Guidance scale for generation
        quantization: "None", "8-bit", or "Auto" for memory optimization
        low_threshold (int): Lower Canny threshold
        high_threshold (int): Higher Canny threshold
        canny_selected_lora_state: List of selected LoRA models
        canny_lora_strength_1/2/3: LoRA strength values
        image_generator: Reference to ImageGenerator instance
        
    Returns:
        PIL.Image: Generated result image
    """
    try:
        from core.database import save_flux_canny_generation
        
        print("=" * 80)
        print("🖋️ FLUX CANNY PROCESSING - PARAMETERS")
        print("=" * 80)
        print(f"📝 Prompt: {prompt}")
        print(f"🔄 Steps: {steps}")
        print(f"🎚️ Guidance Scale: {guidance_scale}")
        print(f"📉 Low Threshold: {low_threshold}")
        print(f"📈 High Threshold: {high_threshold}")
        
        if not prompt or prompt.strip() == "":
            print("❌ Please provide a prompt for the Canny generation")
            return None
            
        if input_image is None:
            print("❌ Please provide an input image")
            return None
        
        # Step 1: Generate Canny edge map
        print("🔍 Step 1: Generating Canny edge map...")
        from utils.canny_processing import preprocess_canny
        
        # Ensure input image is RGB for consistent processing
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        
        canny_image = preprocess_canny(input_image, low_threshold, high_threshold)
        if canny_image is None:
            print("❌ Failed to generate Canny edge map")
            return None
        
        print(f"✅ Canny edge map generated! Size: {canny_image.width}x{canny_image.height}")
        print(f"🔍 Canny format: {canny_image.mode} (should be RGB for LoRA compatibility)")
        
        
        # Step 2: Load FLUX.1-Canny pipeline
        print("🔄 Step 2: Loading FLUX.1-Canny model...")
        try:
            from diffusers import FluxControlPipeline
        except ImportError:
            print("❌ FluxControlPipeline not available in this diffusers version")
            print("Please update diffusers: pip install -U diffusers")
            return None
        
        # Determine device and dtype
        device = image_generator.device
        dtype = image_generator.dtype
        
        print(f"🔧 Using device: {device}, dtype: {dtype}")
        
        # Initialize FLUX Canny pipeline using LoRA (recommended approach)
        try:
            print("🔄 Loading FLUX.1-dev with Canny LoRA...")
            
            # Use FluxControlPipeline with Canny LoRA (as per official docs)
            flux_canny_pipeline = FluxControlPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=dtype,
                use_safetensors=True
            )
            
            # Move pipeline to device
            flux_canny_pipeline = flux_canny_pipeline.to(device)
            
            # For MPS compatibility, try to force VAE to CPU if convolution issues occur
            if device == "mps":
                print("⚠️  MPS detected: Will use CPU fallback for VAE if convolution errors occur")
            
            # Load Canny LoRA BEFORE quantization (important for compatibility)
            print("🔄 Loading Canny LoRA weights...")
            flux_canny_pipeline.load_lora_weights("black-forest-labs/FLUX.1-Canny-dev-lora", adapter_name="canny")
            flux_canny_pipeline.set_adapters("canny", 0.85)
            print("✅ Loaded FLUX.1-dev with Canny LoRA successfully")
            print(f"🎛️ Canny LoRA adapter active at 0.85 strength")
            
            # Apply quantization AFTER loading LoRA for compatibility
            if quantization and quantization != "None":
                from utils.quantization import quantize_pipeline_components
                
                # Apply same quantization logic as main models
                if quantization in ["8-bit", "Auto"]:
                    print(f"🔧 Application quantification qint8 FLUX Canny")
                    success, error = quantize_pipeline_components(flux_canny_pipeline, device, prefer_4bit=False, verbose=True)
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
        except Exception as e:
            print(f"❌ Failed to load FLUX.1-dev with Canny LoRA: {e}")
            print("💡 Make sure FLUX.1-dev and FLUX.1-Canny-dev-lora are available")
            import traceback
            traceback.print_exc()
            return None
        
        # Enable memory efficient attention
        flux_canny_pipeline.enable_attention_slicing()
        print(f"✅ Pipeline configured and ready on {device}")
        
        # Process selected LoRA from the state
        selected_loras = []
        adapter_names = []
        adapter_weights = []
        lora_strengths = [canny_lora_strength_1, canny_lora_strength_2, canny_lora_strength_3]
        
        if canny_selected_lora_state:
            for i, selected_lora in enumerate(canny_selected_lora_state):
                if i < len(lora_strengths):
                    lora_file = selected_lora['name']
                    lora_path = os.path.join(image_generator.lora_directory, lora_file)
                    
                    if os.path.exists(lora_path):
                        strength = lora_strengths[i] if lora_strengths[i] is not None else 0.8
                        selected_loras.append((lora_path, selected_lora, float(strength)))
                        
                        # Prepend activation keyword to prompt
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
                            flux_canny_pipeline.load_lora_weights(
                                lora_dir, 
                                weight_name=lora_filename,
                                adapter_name=adapter_name
                            )
                        
                        adapter_names.append(adapter_name)
                        adapter_weights.append(lora_scale)
                        print(f"✅ LoRA Canny chargé: {lora_info['name']} (weight: {lora_scale})")
                    except KeyError as e:
                        print(f"❌ LoRA incompatible ignoré: {lora_info['name']}")
                        print(f"   Erreur: Paramètre manquant {e}")
                        print(f"   💡 Ce LoRA n'est pas compatible avec FLUX Canny")
                        continue  # Skip this LoRA
                    except Exception as e:
                        print(f"❌ Erreur chargement LoRA Canny: {lora_info['name']}")
                        print(f"   Erreur: {e}")
                        continue  # Skip this LoRA
                
                # Set all adapter weights INCLUDING the Canny adapter
                if adapter_names:
                    # IMPORTANT: Keep the Canny adapter active with user LoRA
                    all_adapter_names = ["canny"] + adapter_names
                    all_adapter_weights = [0.85] + adapter_weights
                    flux_canny_pipeline.set_adapters(all_adapter_names, adapter_weights=all_adapter_weights)
                    print(f"📝 Modified prompt: {prompt}")
                    print(f"🎛️ Active adapters: {all_adapter_names} with weights: {all_adapter_weights}")
                else:
                    # No user LoRA, but keep Canny adapter active
                    flux_canny_pipeline.set_adapters("canny", 0.85)
                    print(f"🎛️ Active adapters: ['canny'] with weight: 0.85")
                    
            except Exception as e:
                print(f"⚠️ Failed to load LoRA: {e}")
        else:
            print("ℹ️ No LoRA models selected")
        
        print("🎨 Step 3: Generating with FLUX.1-Canny-dev...")
        print(f"🔍 FINAL PROMPT USED: '{prompt}'")
        print(f"📐 Image dimensions: {input_image.width}x{input_image.height}")
        print(f"🖋️ Canny control image: {canny_image.size} ({canny_image.mode})")
        print(f"🎚️ Final guidance scale: {guidance_scale}")
        print(f"🔄 Final steps: {steps}")
        
        # Generate seed for reproducibility
        seed = random.randint(1, 2**32 - 1)
        generator = torch.Generator(device=device).manual_seed(seed)
        
        print(f"🎲 Using seed: {seed}")
        
        # Apply progress tracking for FLUX Canny
        print("🚀 Starting FLUX Canny generation with progress tracking...")
        
        # Reset and start progress tracking
        global_progress_tracker.reset()
        global_progress_tracker.apply_tqdm_patches()
        
        try:
            # Run FLUX Canny generation using FluxControlPipeline
            # Handle MPS convolution errors gracefully
            try:
                result = flux_canny_pipeline(
                    prompt=prompt,
                    control_image=canny_image,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    height=input_image.height,
                    width=input_image.width,
                    generator=generator
                )
            except NotImplementedError as e:
                if "convolution_overrideable" in str(e) and device == "mps":
                    print("⚠️  MPS convolution error detected, moving VAE to CPU...")
                    # Move VAE to CPU to avoid convolution issues
                    flux_canny_pipeline.vae = flux_canny_pipeline.vae.to("cpu")
                    
                    # Retry the generation
                    result = flux_canny_pipeline(
                        prompt=prompt,
                        control_image=canny_image,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        height=input_image.height,
                        width=input_image.width,
                        generator=generator
                    )
                else:
                    raise e
        finally:
            # Always restore patches after generation
            global_progress_tracker.remove_tqdm_patches()
            print("✅ FLUX Canny generation completed with progress tracking")
        
        result_image = result.images[0]
        
        print("✅ FLUX Canny generation completed!")
        
        # Clean up memory
        del flux_canny_pipeline
        
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
            output_filename = output_dir / f"flux_canny_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
            
            result_image.save(str(output_filename))
            
            # Prepare LoRA data for database storage
            lora_paths_for_db = []
            lora_scales_for_db = []
            if selected_loras:
                for lora_path, lora_info, lora_scale in selected_loras:
                    lora_paths_for_db.append(lora_path)
                    lora_scales_for_db.append(lora_scale)
            
            # Save to database
            save_flux_canny_generation(
                timestamp_str,
                seed,
                prompt,
                steps,
                guidance_scale,
                low_threshold,
                high_threshold,
                result_image.height,
                result_image.width,
                lora_paths_for_db,
                lora_scales_for_db,
                str(output_filename)
            )
            
            print(f"💾 Image saved: {output_filename}")
            print(f"📊 Saved to database: {timestamp_str}")
            print("✅ FLUX CANNY PROCESSING COMPLETED!")
            print("=" * 80)
        
        return result_image
        
    except Exception as e:
        print(f"❌ Error during FLUX Canny processing: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return None