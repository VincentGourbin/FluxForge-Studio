"""
Mask Utilities Module

Tools for mask creation and manipulation for FLUX Fill operations.
Provides functions for generating masks for inpainting and outpainting.

Features:
- Automatic mask generation from ImageEditor data
- Outpainting mask creation with expansion percentages
- Mask preview generation
- Image difference detection

Author: MFLUX Team
"""

import numpy as np
from PIL import Image

def extract_inpainting_mask_from_editor(image_editor_data):
    """
    Extract inpainting mask from Gradio ImageEditor data.
    
    Args:
        image_editor_data (dict): Data from Gradio ImageEditor component
        
    Returns:
        PIL.Image or None: Extracted mask (L mode) where WHITE=fill, BLACK=keep
    """
    if image_editor_data is None or 'background' not in image_editor_data:
        return None
    
    base_image = image_editor_data['background']
    if base_image is None:
        return None
    
    mask = None
    
    # Method 1: Try to get mask from composite (preferred)
    composite = image_editor_data.get('composite')
    if composite and base_image:
        bg_array = np.array(base_image.convert('RGB'))
        comp_array = np.array(composite.convert('RGB'))
        
        # Find pixels that changed (where user drew)
        diff = np.sum(np.abs(bg_array.astype(int) - comp_array.astype(int)), axis=2)
        mask_array = (diff > 30).astype(np.uint8) * 255  # White where user drew
        
        if np.sum(mask_array) > 0:
            mask = Image.fromarray(mask_array, mode='L')
    
    # Method 2: Fallback to layers extraction
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

def validate_mask(mask, min_area_threshold=100):
    """
    Validate that a mask has sufficient area for processing.
    
    Args:
        mask (PIL.Image): Mask to validate (L mode)
        min_area_threshold (int): Minimum white pixels required
        
    Returns:
        bool: True if mask is valid for processing
    """
    if mask is None:
        return False
    
    mask_array = np.array(mask)
    white_pixels = np.sum(mask_array > 128)  # Count white/light pixels
    
    return white_pixels >= min_area_threshold

def invert_mask(mask):
    """
    Invert a mask (swap BLACK and WHITE areas).
    
    Args:
        mask (PIL.Image): Input mask (L mode)
        
    Returns:
        PIL.Image: Inverted mask
    """
    if mask is None:
        return None
    
    mask_array = np.array(mask)
    inverted_array = 255 - mask_array
    
    return Image.fromarray(inverted_array, mode='L')

def resize_mask_to_image(mask, target_image):
    """
    Resize mask to match target image dimensions.
    
    Args:
        mask (PIL.Image): Input mask
        target_image (PIL.Image): Target image for size reference
        
    Returns:
        PIL.Image: Resized mask
    """
    if mask is None or target_image is None:
        return mask
    
    return mask.resize(target_image.size, Image.Resampling.NEAREST)

def blend_mask_edges(mask, blur_radius=2):
    """
    Apply slight blur to mask edges for smoother transitions.
    
    Args:
        mask (PIL.Image): Input mask (L mode)
        blur_radius (int): Blur radius for edge softening
        
    Returns:
        PIL.Image: Mask with softened edges
    """
    if mask is None:
        return None
    
    from PIL import ImageFilter
    
    # Apply slight gaussian blur to soften edges
    blurred_mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return blurred_mask