# background_remover.py
"""
Background removal module using RMBG-2.0 model.
Provides functionality to remove backgrounds from images using AI-powered segmentation.
"""

import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from PIL import Image
from config import device

def load_background_removal_model():
    """
    Load and initialize the RMBG-2.0 background removal model.
    
    Returns:
        AutoModelForImageSegmentation: Loaded and configured model ready for inference
    """
    modelbgrm = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
    torch.set_float32_matmul_precision('high')
    modelbgrm.to(device)
    modelbgrm.eval()
    return modelbgrm

# Image preprocessing configuration
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

def remove_background(input_image, modelbgrm):
    """
    Remove the background from an input image using the RMBG-2.0 model.
    
    Args:
        input_image (PIL.Image): Input image to process
        modelbgrm (AutoModelForImageSegmentation): Loaded background removal model
        
    Returns:
        PIL.Image: Image with background removed (transparent background)
    """
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
    
    return image