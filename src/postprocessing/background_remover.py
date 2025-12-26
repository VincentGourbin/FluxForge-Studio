# background_remover.py
"""
Background removal module using RMBG-2.0 model.
Provides functionality to remove backgrounds from images using AI-powered segmentation.

The model is loaded lazily on first use to avoid startup delays and authentication
errors when the model isn't needed.
"""

import os
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from PIL import Image
from core.config import device

# Lazy-loaded singleton for background removal model
_background_model = None
_model_loading = False


def get_hf_token():
    """Get HuggingFace token from environment variable."""
    return os.environ.get("HF_TOKEN")


def load_background_removal_model():
    """
    Load and initialize the RMBG-2.0 background removal model.
    Uses HF_TOKEN from environment for gated repository access.

    Returns:
        AutoModelForImageSegmentation: Loaded and configured model ready for inference
    """
    global _background_model, _model_loading

    # Return cached model if already loaded
    if _background_model is not None:
        return _background_model

    # Prevent concurrent loading
    if _model_loading:
        import time
        while _model_loading:
            time.sleep(0.1)
        return _background_model

    _model_loading = True

    try:
        print("🔄 Loading RMBG-2.0 background removal model...")
        hf_token = get_hf_token()

        _background_model = AutoModelForImageSegmentation.from_pretrained(
            'briaai/RMBG-2.0',
            trust_remote_code=True,
            token=hf_token
        )
        torch.set_float32_matmul_precision('high')
        _background_model.to(device)
        _background_model.eval()
        print("✅ RMBG-2.0 model loaded successfully")
        return _background_model
    finally:
        _model_loading = False


def get_background_model():
    """
    Get the background removal model, loading it lazily if needed.

    Returns:
        AutoModelForImageSegmentation: The loaded model
    """
    return load_background_removal_model()

# Image preprocessing configuration
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

def remove_background(input_image, modelbgrm=None):
    """
    Remove the background from an input image using the RMBG-2.0 model.

    Args:
        input_image (PIL.Image): Input image to process
        modelbgrm (AutoModelForImageSegmentation, optional): Loaded background removal model.
            If None, the model will be loaded lazily.

    Returns:
        PIL.Image: Image with background removed (transparent background)
    """
    import datetime
    import random
    from pathlib import Path

    # Lazy load model if not provided
    if modelbgrm is None:
        modelbgrm = get_background_model()

    # Convert to RGB format
    image = input_image.convert("RGB")

    # Apply preprocessing transformations
    input_images = transform_image(image).unsqueeze(0).to(device)
    
    # Generate segmentation mask
    with torch.no_grad():
        preds = modelbgrm(input_images)[-1].sigmoid().cpu()
    
    # Process the prediction mask
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    
    # Apply alpha channel (transparency) using the mask
    image.putalpha(mask)
    
    # Save result to file and database
    timestamp = datetime.datetime.now()
    timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
    
    # Generate seed for database consistency
    seed = random.randint(1, 2**32 - 1)
    
    # Create output directory and filename
    output_dir = Path("outputimage")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / f"bgrm_{timestamp.strftime('%Y%m%d_%H%M%S')}_{seed}.png"
    
    # Save the image with transparent background
    image.save(str(output_filename))
    
    # Save to database
    try:
        from core.database import save_background_removal_generation
        save_background_removal_generation(
            timestamp_str,
            seed,
            image.size[1],  # height
            image.size[0],  # width
            str(output_filename)
        )
        print(f"📊 Background removal saved to database: {timestamp_str}")
    except Exception as e:
        print(f"⚠️ Could not save background removal to database: {e}")
    
    print(f"💾 Background removal result saved: {output_filename}")
    
    return image