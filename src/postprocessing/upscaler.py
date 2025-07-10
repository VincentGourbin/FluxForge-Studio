"""
Upscaler Post-Processing Module

Image resolution enhancement using ControlNet upscaler models.
Provides high-quality image upscaling with AI-powered detail enhancement.

Features:
- ControlNet-based upscaling for superior quality
- Configurable multiplier (1.5x to 4x)
- Memory-efficient processing
- Support for various image formats

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

def upscale_image(input_image, upscale_factor, quantization="None"):
    """
    High-quality image upscaling using FLUX.1-dev ControlNet Upscaler.
    
    Args:
        input_image (PIL.Image): Input image to upscale
        upscale_factor (float): Upscaling multiplier (1.0-4.0)
        quantization (str): Quantization method (None, 4-bit, 8-bit, Auto)
        
    Returns:
        PIL.Image: Upscaled image or None if processing fails
    """
    try:
        if input_image is None:
            print("❌ No input image provided for upscaling")
            return None
        
        original_width, original_height = input_image.size
        
        print("=" * 80)
        print("📈 CONTROLNET UPSCALER PROCESSING")
        print("=" * 80)
        print(f"🔍 Original size: {original_width}x{original_height}")
        print(f"📊 Upscale factor: {upscale_factor}x")
        
        # Calculate target dimensions
        target_width = int(original_width * upscale_factor)
        target_height = int(original_height * upscale_factor)
        
        print(f"🎯 Target size: {target_width}x{target_height}")
        
        # Load ControlNet upscaler
        print("🔄 Loading FLUX.1-dev ControlNet Upscaler...")
        try:
            from diffusers import FluxControlNetPipeline, FluxControlNetModel
        except ImportError:
            print("❌ FluxControlNetPipeline not available in this diffusers version")
            print("Please update diffusers: pip install -U diffusers")
            return None
        
        # Model configuration
        upscaler_model_id = "jasperai/Flux.1-dev-Controlnet-Upscaler"
        base_model_id = "black-forest-labs/FLUX.1-dev"
        
        # Determine device and dtype
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.bfloat16
        else:
            device = "cpu"
            dtype = torch.float32
        
        print(f"🔧 Using device: {device}, dtype: {dtype}")
        
        # Load ControlNet upscaler
        controlnet = FluxControlNetModel.from_pretrained(
            upscaler_model_id, 
            torch_dtype=dtype
        )
        
        # Load upscaler pipeline
        upscaler_pipeline = FluxControlNetPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnet,
            torch_dtype=dtype,
            use_safetensors=True
        )
        
        # Move to device
        upscaler_pipeline = upscaler_pipeline.to(device)
        
        # Enable memory efficient attention
        upscaler_pipeline.enable_attention_slicing()
        
        # Apply quantization if specified
        if quantization and quantization != "None":
            try:
                from utils.quantization import quantize_pipeline_components
                
                if quantization in ["8-bit", "Auto"]:
                    print(f"🔧 Application quantification qint8 pour upscaler...")
                    success, error = quantize_pipeline_components(upscaler_pipeline, device, prefer_4bit=False, verbose=True)
                    if not success:
                        print(f"⚠️  Quantification qint8 échouée: {error}")
                        print("🔄 Continuons sans quantification...")
                elif quantization == "4-bit":
                    print(f"⚠️  Quantification 4-bit non supportée pour upscaler")
                    print("💡 Conseil: Utilisez '8-bit' pour économie mémoire")
                    print("🔄 Continuons sans quantification...")
                else:
                    print(f"⚠️  Quantification {quantization} non supportée pour upscaler")
                    print("🔄 Continuons sans quantification...")
            except ImportError:
                print("⚠️  Module quantization non disponible")
                print("🔄 Continuons sans quantification...")
        
        # Ensure input image is RGB
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        
        # Generate seed for reproducibility
        seed = random.randint(1, 2**32 - 1)
        generator = torch.Generator(device=device).manual_seed(seed)
        
        print(f"🎲 Using seed: {seed}")
        print("🎨 Generating upscaled image...")
        
        # Generate upscaled image using ControlNet
        # Using recommended parameters for upscaler
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            
            upscaled_image = upscaler_pipeline(
                prompt="",  # Empty prompt as recommended for upscaler
                control_image=input_image,
                controlnet_conditioning_scale=0.6,  # Recommended strength
                num_inference_steps=28,  # Recommended steps
                guidance_scale=3.5,  # Recommended guidance
                height=target_height,
                width=target_width,
                generator=generator
            ).images[0]
        
        print("✅ Upscaling completed!")
        
        # Clean up memory
        del upscaler_pipeline
        del controlnet
        
        if device == 'cuda':
            torch.cuda.empty_cache()
        elif device == 'mps':
            torch.mps.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Save upscaled image
        timestamp = datetime.datetime.now()
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        output_dir = Path("outputimage")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = output_dir / f"upscaled_{timestamp.strftime('%Y%m%d_%H%M%S')}_x{upscale_factor}.png"
        
        upscaled_image.save(str(output_filename))
        
        # Save to database
        try:
            from core.database import save_upscaler_generation
            save_upscaler_generation(
                timestamp_str,
                seed,
                upscale_factor,
                target_height,
                target_width,
                str(output_filename)
            )
            print(f"📊 Saved to database: {timestamp_str}")
        except Exception as e:
            print(f"⚠️ Could not save to database: {e}")
        
        print(f"💾 Upscaled image saved: {output_filename}")
        print("✅ UPSCALER PROCESSING COMPLETED!")
        print("=" * 80)
        
        return upscaled_image
        
    except Exception as e:
        print(f"❌ Error during upscaling: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return None

def process_upscaler(input_image_path, multiplier, quantization="None"):
    """
    Process image with upscaler using ControlNet upscaler.
    
    Args:
        input_image_path (str): Path to input image file
        multiplier (float): Upscaling multiplier (1.5-4.0)
        quantization (str): Quantization method (None, 4-bit, 8-bit, Auto)
        
    Returns:
        PIL.Image: Upscaled image or None if processing fails
    """
    try:
        if not input_image_path:
            print("❌ No input image provided for upscaling")
            return None
        
        # Load input image
        input_image = Image.open(input_image_path)
        
        # Use the main upscaling function
        return upscale_image(input_image, multiplier, quantization)
        
    except Exception as e:
        print(f"❌ Error during upscaling: {e}")
        return None

def remove_background_wrapper(input_image, modelbgrm):
    """
    Wrapper function for background removal to maintain compatibility.
    
    Args:
        input_image (PIL.Image): Input image to process
        modelbgrm: Background removal model
        
    Returns:
        PIL.Image: Image with background removed
    """
    # Import the actual background removal function
    from .background_remover import remove_background
    
    if input_image is None:
        print("❌ No input image provided for background removal")
        return None
    
    try:
        result = remove_background(input_image, modelbgrm)
        print("✅ Background removal completed!")
        return result
    except Exception as e:
        print(f"❌ Error during background removal: {e}")
        return None