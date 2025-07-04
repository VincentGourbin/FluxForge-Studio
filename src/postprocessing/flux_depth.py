"""
FLUX Depth Post-Processing Module

Depth-based image generation using FLUX.1-Depth-dev model.
Generates depth maps from input images and creates new images based on depth guidance.

Features:
- Automatic depth map generation using DepthAnything
- FLUX.1-Depth-dev pipeline for depth-guided generation
- LoRA support for style enhancement
- Memory-efficient processing with device detection
- Two-step process: depth generation preview + final image generation

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
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

def generate_depth_map(input_image):
    """
    Generate depth map from input image using Depth Anything via Transformers.
    
    Args:
        input_image (PIL.Image): Input image to generate depth from
        
    Returns:
        PIL.Image: Depth map as RGB image or None if processing fails
    """
    try:
        if input_image is None:
            print("❌ No input image provided for depth generation")
            return None
        
        print("=" * 80)
        print("🔍 DEPTH MAP GENERATION")
        print("=" * 80)
        print(f"📐 Input image size: {input_image.width}x{input_image.height}")
        
        # Determine device and dtype carefully for depth model compatibility
        if torch.cuda.is_available():
            device = "cuda"
            # Use float32 for better stability with depth models
            dtype = torch.float32
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            # Use float32 for MPS to avoid type mismatch issues
            dtype = torch.float32
        else:
            device = "cpu"
            dtype = torch.float32
        
        print(f"🔧 Using device: {device}, dtype: {dtype}")
        
        # Load Depth Anything model directly using transformers
        try:
            print("🔄 Loading Depth Anything model...")
            image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
            model = AutoModelForDepthEstimation.from_pretrained(
                "LiheYoung/depth-anything-large-hf",
                torch_dtype=dtype
            )
            
            # Move model to device
            model = model.to(device)
            
            print(f"✅ Depth Anything model loaded successfully on {device}!")
        except Exception as e:
            print("❌ Failed to load Depth Anything model")
            print(f"Error: {e}")
            return None
        
        # Ensure input image is RGB
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        
        print("🎨 Generating depth map...")
        
        # Prepare image for the model
        inputs = image_processor(images=input_image, return_tensors="pt")
        
        # Move inputs to device and ensure correct dtype
        inputs = {k: v.to(device, dtype=dtype) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=input_image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        
        # Convert to numpy and normalize
        depth_array = prediction.squeeze().cpu().numpy()
        
        # Normalize to 0-255 range
        depth_min = depth_array.min()
        depth_max = depth_array.max()
        depth_normalized = ((depth_array - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        
        # Convert to RGB depth map
        depth_map = Image.fromarray(depth_normalized, mode='L').convert('RGB')
        
        # Clean up memory
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
        elif device == 'mps':
            torch.mps.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        print("✅ Depth map generation completed!")
        print(f"📐 Depth map size: {depth_map.width}x{depth_map.height}")
        print("=" * 80)
        
        return depth_map
        
    except Exception as e:
        print(f"❌ Error during depth map generation: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return None

def process_flux_depth(input_image, prompt, steps, guidance_scale, 
                      depth_selected_lora_state, depth_lora_strength_1, 
                      depth_lora_strength_2, depth_lora_strength_3, image_generator):
    """
    Process depth-guided image generation using FLUX.1-Depth-dev model.
    
    Args:
        input_image (PIL.Image): Input image to extract depth from
        prompt (str): Text prompt for generation
        steps (int): Number of inference steps
        guidance_scale (float): Guidance scale for generation
        depth_selected_lora_state: List of selected LoRA models
        depth_lora_strength_1/2/3: LoRA strength values
        image_generator: Reference to ImageGenerator instance
        
    Returns:
        PIL.Image: Generated result image
    """
    try:
        from core.database import save_flux_depth_generation
        
        print("=" * 80)
        print("🌊 FLUX DEPTH PROCESSING - PARAMETERS")
        print("=" * 80)
        print(f"📝 Prompt: {prompt}")
        print(f"🔄 Steps: {steps}")
        print(f"🎚️ Guidance Scale: {guidance_scale}")
        
        if not prompt or prompt.strip() == "":
            print("❌ Please provide a prompt for the depth generation")
            return None
            
        if input_image is None:
            print("❌ Please provide an input image")
            return None
        
        # First generate depth map
        print("🔍 Step 1: Generating depth map...")
        depth_map = generate_depth_map(input_image)
        if depth_map is None:
            print("❌ Failed to generate depth map")
            return None
        
        # Load FLUX.1-Depth pipeline
        print("🔄 Step 2: Loading FLUX.1-Depth model...")
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
        
        # Use LoRA version (recommended approach)
        try:
            print("🔄 Loading FLUX.1-dev with Depth LoRA...")
            flux_depth_pipeline = FluxControlPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=dtype,
                use_safetensors=True
            )
            # Load depth LoRA (this is the preferred method)
            flux_depth_pipeline.load_lora_weights("black-forest-labs/FLUX.1-Depth-dev-lora", adapter_name="depth")
            flux_depth_pipeline.set_adapters("depth", 0.85)
            print("✅ Loaded FLUX.1-dev with Depth LoRA successfully")
        except Exception as e:
            print(f"❌ Failed to load FLUX.1-dev with Depth LoRA: {e}")
            print("💡 Make sure FLUX.1-dev and FLUX.1-Depth-dev-lora are available")
            return None
        
        # Move pipeline to appropriate device
        print(f"🔄 Moving pipeline to {device}...")
        flux_depth_pipeline = flux_depth_pipeline.to(device)
        print(f"✅ Pipeline moved to {device} successfully!")
        
        # Enable memory efficient attention
        flux_depth_pipeline.enable_attention_slicing()
        
        # Process selected LoRA from the state
        selected_loras = []
        adapter_names = []
        adapter_weights = []
        lora_strengths = [depth_lora_strength_1, depth_lora_strength_2, depth_lora_strength_3]
        
        if depth_selected_lora_state:
            for i, selected_lora in enumerate(depth_selected_lora_state):
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
                    
                    # Load LoRA with warning suppression
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*CLIPTextModel.*")
                        warnings.filterwarnings("ignore", message=".*No LoRA keys associated to CLIPTextModel.*")
                        warnings.filterwarnings("ignore", message=".*Already found a.*peft_config.*attribute.*")
                        flux_depth_pipeline.load_lora_weights(
                            lora_dir, 
                            weight_name=lora_filename,
                            adapter_name=adapter_name
                        )
                    
                    adapter_names.append(adapter_name)
                    adapter_weights.append(lora_scale)
                    print(f"✅ LoRA loaded: {lora_info['name']} (weight: {lora_scale})")
                
                # Set all adapter weights
                if adapter_names:
                    flux_depth_pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
                    print(f"📝 Modified prompt: {prompt}")
                    
            except Exception as e:
                print(f"⚠️ Failed to load LoRA: {e}")
        else:
            print("ℹ️ No LoRA models selected")
        
        print("🎨 Step 3: Generating with FLUX.1-Depth-dev...")
        print(f"🔍 FINAL PROMPT USED: '{prompt}'")
        print(f"📐 Image dimensions: {input_image.width}x{input_image.height}")
        print(f"🎚️ Final guidance scale: {guidance_scale}")
        print(f"🔄 Final steps: {steps}")
        
        # Generate seed for reproducibility
        seed = random.randint(1, 2**32 - 1)
        generator = torch.Generator(device=device).manual_seed(seed)
        
        print(f"🎲 Using seed: {seed}")
        
        # Run FLUX Depth generation
        result = flux_depth_pipeline(
            prompt=prompt,
            control_image=depth_map,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=input_image.height,
            width=input_image.width,
            generator=generator
        )
        
        result_image = result.images[0]
        
        print("✅ FLUX Depth generation completed!")
        
        # Clean up memory
        del flux_depth_pipeline
        
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
            output_filename = output_dir / f"flux_depth_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
            
            result_image.save(str(output_filename))
            
            # Prepare LoRA data for database storage
            lora_paths_for_db = []
            lora_scales_for_db = []
            if selected_loras:
                for lora_path, lora_info, lora_scale in selected_loras:
                    lora_paths_for_db.append(lora_path)
                    lora_scales_for_db.append(lora_scale)
            
            # Save to database
            save_flux_depth_generation(
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
            print("✅ FLUX DEPTH PROCESSING COMPLETED!")
            print("=" * 80)
        
        return result_image
        
    except Exception as e:
        print(f"❌ Error during FLUX Depth processing: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return None