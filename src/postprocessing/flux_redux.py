"""
FLUX Redux Post-Processing Module

Image variation and refinement using FLUX.1-Redux-dev model.
Creates slight variations of input images while maintaining overall structure and style.

Features:
- Image-to-image variation and refinement
- Maintains input image structure while adding variation
- Configurable guidance scale for variation intensity
- Memory-efficient processing with device detection
- No text prompts needed - pure image variation

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

def process_flux_redux(input_image, guidance_scale, steps, variation_strength, quantization, image_generator):
    """
    Process image variation using FLUX.1-Redux-dev model.
    
    Args:
        input_image (PIL.Image): Input image to create variations from
        guidance_scale (float): Controls how closely output follows input (1.0-5.0)
        steps (int): Number of inference steps
        variation_strength (float): Intensity of variation (0.1-1.0)
        quantization: "None", "8-bit", or "Auto" for memory optimization
        image_generator: Reference to ImageGenerator instance
        
    Returns:
        PIL.Image: Generated variation image
    """
    try:
        from core.database import save_flux_redux_generation
        
        print("=" * 80)
        print("🔄 FLUX REDUX PROCESSING - PARAMETERS")
        print("=" * 80)
        print(f"🎚️ Guidance Scale: {guidance_scale}")
        print(f"🔄 Steps: {steps}")
        print(f"💫 Variation Strength: {variation_strength}")
        
        if input_image is None:
            print("❌ Please provide an input image")
            return None
        
        # Ensure input image is RGB for consistent processing
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        
        print(f"📐 Input image: {input_image.width}x{input_image.height}")
        
        # Load FLUX Redux pipelines
        print("🔄 Loading FLUX.1-Redux-dev model...")
        try:
            from diffusers import FluxPriorReduxPipeline, FluxPipeline
        except ImportError:
            print("❌ FLUX Redux pipelines not available in this diffusers version")
            print("Please update diffusers: pip install -U diffusers")
            return None
        
        # Determine device and dtype
        device = image_generator.device
        dtype = image_generator.dtype
        
        print(f"🔧 Using device: {device}, dtype: {dtype}")
        
        # Load Redux Prior pipeline
        try:
            print("🔄 Loading FLUX Prior Redux pipeline...")
            pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Redux-dev",
                torch_dtype=dtype,
                use_safetensors=True
            )
            pipe_prior_redux = pipe_prior_redux.to(device)
            
            # Apply quantization to Prior Redux pipeline if requested
            if quantization and quantization != "None":
                from utils.quantization import quantize_pipeline_components
                
                if quantization in ["8-bit", "Auto"]:
                    print(f"🔧 Application quantification qint8 FLUX Prior Redux")
                    success, error = quantize_pipeline_components(pipe_prior_redux, device, prefer_4bit=False, verbose=True)
                    if not success:
                        print(f"⚠️  Quantification Prior Redux qint8 échouée: {error}")
                        print("🔄 Continuons sans quantification...")
            
            print("✅ FLUX Prior Redux pipeline loaded")
        except Exception as e:
            print(f"❌ Failed to load FLUX Prior Redux pipeline: {e}")
            return None
        
        # Load base FLUX pipeline (without text encoders for Redux)
        try:
            print("🔄 Loading FLUX base pipeline...")
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                text_encoder=None,
                text_encoder_2=None,
                torch_dtype=dtype,
                use_safetensors=True
            )
            pipe = pipe.to(device)
            
            # Apply quantization to base FLUX pipeline if requested
            if quantization and quantization != "None":
                from utils.quantization import quantize_pipeline_components
                
                if quantization in ["8-bit", "Auto"]:
                    print(f"🔧 Application quantification qint8 FLUX Base")
                    success, error = quantize_pipeline_components(pipe, device, prefer_4bit=False, verbose=True)
                    if not success:
                        print(f"⚠️  Quantification Base FLUX qint8 échouée: {error}")
                        print("🔄 Continuons sans quantification...")
            
            print("✅ FLUX base pipeline loaded")
        except Exception as e:
            print(f"❌ Failed to load FLUX base pipeline: {e}")
            return None
        
        # Enable memory efficient attention
        pipe_prior_redux.enable_attention_slicing()
        pipe.enable_attention_slicing()
        
        print("🎨 Generating image variation...")
        print(f"📐 Output dimensions: {input_image.width}x{input_image.height}")
        print(f"🎚️ Final guidance scale: {guidance_scale}")
        print(f"🔄 Final steps: {steps}")
        
        # Generate seed for reproducibility
        seed = random.randint(1, 2**32 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)  # Redux uses CPU generator
        
        print(f"🎲 Using seed: {seed}")
        
        # Step 1: Process image through Redux Prior
        print("🔄 Step 1: Processing through Redux Prior...")
        pipe_prior_output = pipe_prior_redux(input_image)
        print("✅ Redux Prior processing completed")
        
        # Step 2: Generate variation through FLUX pipeline
        print("🔄 Step 2: Generating variation with progress tracking...")
        
        # Adjust guidance scale based on variation strength
        adjusted_guidance = guidance_scale * variation_strength
        
        # Apply progress tracking for FLUX Redux
        global_progress_tracker.reset()
        global_progress_tracker.apply_tqdm_patches()
        
        try:
            result = pipe(
                guidance_scale=adjusted_guidance,
                num_inference_steps=steps,
                generator=generator,
                **pipe_prior_output,
            )
        finally:
            # Always restore patches after generation
            global_progress_tracker.remove_tqdm_patches()
            print("✅ FLUX Redux generation completed with progress tracking")
        
        result_image = result.images[0]
        
        print("✅ FLUX Redux generation completed!")
        
        # Clean up memory
        del pipe_prior_redux
        del pipe
        
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
            output_filename = output_dir / f"flux_redux_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
            
            result_image.save(str(output_filename))
            
            # Save to database
            save_flux_redux_generation(
                timestamp_str,
                seed,
                guidance_scale,
                steps,
                variation_strength,
                result_image.height,
                result_image.width,
                str(output_filename)
            )
            
            print(f"💾 Image saved: {output_filename}")
            print(f"📊 Saved to database: {timestamp_str}")
            print("✅ FLUX REDUX PROCESSING COMPLETED!")
            print("=" * 80)
        
        return result_image
        
    except Exception as e:
        print(f"❌ Error during FLUX Redux processing: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return None