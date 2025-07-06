"""
Kontext Post-Processing Module

Text-based image editing using FLUX.1-Kontext-dev model.
Allows transformation and modification of images using natural language descriptions.

Features:
- Text-based image editing and transformation
- LoRA support for style enhancement
- Memory-efficient model loading
- Comprehensive parameter control

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

def process_kontext(input_image, prompt, steps, guidance_scale, quantization, kontext_selected_lora_state, 
                   kontext_lora_strength_1, kontext_lora_strength_2, kontext_lora_strength_3, image_generator):
    """
    Process image editing using FLUX.1-Kontext-dev model.
    
    Args:
        input_image: Input image to edit
        prompt: Text prompt describing desired changes
        steps: Number of inference steps
        guidance_scale: Guidance scale for generation
        quantization: "None", "8-bit", or "Auto" for memory optimization
        kontext_selected_lora_state: List of selected LoRA models
        kontext_lora_strength_1/2/3: LoRA strength values
        image_generator: Reference to ImageGenerator instance
        
    Returns:
        PIL.Image: Edited result image
    """
    try:
        from core.database import save_image_info
        
        print("=" * 80)
        print("🎨 KONTEXT PROCESSING - PARAMETERS")
        print("=" * 80)
        print(f"📝 Prompt: {prompt}")
        print(f"🔄 Steps: {steps}")
        print(f"🎚️ Guidance Scale: {guidance_scale}")
        
        if not prompt or prompt.strip() == "":
            print("❌ Please provide a prompt for the editing operation")
            return None
            
        if input_image is None:
            print("❌ Please provide an input image")
            return None
        
        # Load FLUX.1-Kontext-dev pipeline
        print("🔄 Loading FLUX.1-Kontext-dev model...")
        try:
            from diffusers import FluxKontextPipeline
        except ImportError:
            print("❌ FluxKontextPipeline not available in this diffusers version")
            print("Please update diffusers: pip install -U diffusers")
            return None
        
        # Determine device and dtype
        device = image_generator.device
        dtype = image_generator.dtype
        
        print(f"🔧 Using device: {device}, dtype: {dtype}")
        
        # Initialize FLUX Kontext pipeline
        kontext_pipeline = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            torch_dtype=dtype,
            use_safetensors=True
        )
        
        # Move pipeline to appropriate device
        kontext_pipeline = kontext_pipeline.to(device)
        
        # Enable memory efficient attention
        kontext_pipeline.enable_attention_slicing()
        
        # IMPORTANT: Quantization sera appliquée APRÈS le chargement des LoRA
        
        # Process selected LoRA from the state
        selected_loras = []
        adapter_names = []
        adapter_weights = []
        lora_strengths = [kontext_lora_strength_1, kontext_lora_strength_2, kontext_lora_strength_3]
        
        if kontext_selected_lora_state:
            for i, selected_lora in enumerate(kontext_selected_lora_state):
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
                            kontext_pipeline.load_lora_weights(
                                lora_dir, 
                                weight_name=lora_filename,
                                adapter_name=adapter_name
                            )
                        
                        adapter_names.append(adapter_name)
                        adapter_weights.append(lora_scale)
                        print(f"✅ LoRA Kontext chargé: {lora_info['name']} (weight: {lora_scale})")
                    except KeyError as e:
                        print(f"❌ LoRA incompatible ignoré: {lora_info['name']}")
                        print(f"   Erreur: Paramètre manquant {e}")
                        print(f"   💡 Ce LoRA n'est pas compatible avec FLUX Kontext")
                        continue  # Skip this LoRA
                    except Exception as e:
                        print(f"❌ Erreur chargement LoRA Kontext: {lora_info['name']}")
                        print(f"   Erreur: {e}")
                        continue  # Skip this LoRA
                
                # Set all adapter weights
                if adapter_names:
                    kontext_pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
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
                print(f"🔧 Application quantification qint8 FLUX Kontext (économie mémoire ~70%) APRÈS LoRA")
                success, error = quantize_pipeline_components(kontext_pipeline, device, prefer_4bit=False, verbose=True)
                if not success:
                    print(f"⚠️  Quantification qint8 échouée: {error}")
                    print("🔄 Continuons sans quantification...")
            elif quantization == "4-bit":
                print(f"⚠️  Quantification 4-bit non supportée sur {device} (tests montrent erreurs)")
                print("💡 Conseil: Utilisez '8-bit' pour économie mémoire substantielle")
                print("🔄 Continuons sans quantification...")
            else:
                print(f"⚠️  Quantification {quantization} non supportée")
                print("🔄 Continuons sans quantification...")
        
        print("🎨 Generating with FLUX.1-Kontext-dev...")
        print(f"🔍 FINAL PROMPT USED: '{prompt}'")
        print(f"📐 Image dimensions: {input_image.width}x{input_image.height}")
        print(f"🎚️ Final guidance scale: {guidance_scale}")
        print(f"🔄 Final steps: {steps}")
        
        # Generate seed for reproducibility
        seed = random.randint(1, 2**32 - 1)
        generator = torch.Generator(device=device).manual_seed(seed)
        
        print(f"🎲 Using seed: {seed}")
        
        # Ensure input image is RGB
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        
        # Run Kontext generation
        result = kontext_pipeline(
            image=input_image,
            prompt=prompt,
            height=input_image.height,
            width=input_image.width,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator
        )
        
        result_image = result.images[0]
        
        print("✅ Kontext generation completed!")
        
        # Clean up memory
        del kontext_pipeline
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
            output_filename = output_dir / f"kontext_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
            
            result_image.save(str(output_filename))
            
            # Prepare LoRA data for database storage
            lora_paths_for_db = []
            lora_scales_for_db = []
            if selected_loras:
                for lora_path, lora_info, lora_scale in selected_loras:
                    lora_paths_for_db.append(lora_path)
                    lora_scales_for_db.append(lora_scale)
            
            # Save to database
            from core.database import save_kontext_generation
            save_kontext_generation(
                timestamp_str,
                seed,
                prompt,
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
            print("✅ KONTEXT PROCESSING COMPLETED!")
            print("=" * 80)
        
        return result_image
        
    except Exception as e:
        print(f"❌ Error during Kontext processing: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return None