"""
Image Processing Utilities Module

General-purpose image processing functions and utilities.
Provides common operations for image manipulation and format conversion.

Features:
- Image format conversions and validation
- Memory management helpers
- File I/O utilities
- Common image transformations

Author: MFLUX Team
"""

import os
import gc
import torch
from PIL import Image
from pathlib import Path

def ensure_rgb_format(image):
    """
    Ensure image is in RGB format for model compatibility.
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        PIL.Image: Image converted to RGB format
    """
    if image is None:
        return None
    
    if image.mode != 'RGB':
        return image.convert('RGB')
    
    return image

def validate_image_size(image, min_size=64, max_size=2048):
    """
    Validate image dimensions are within acceptable range.
    
    Args:
        image (PIL.Image): Image to validate
        min_size (int): Minimum dimension (width or height)
        max_size (int): Maximum dimension (width or height)
        
    Returns:
        bool: True if image size is valid
    """
    if image is None:
        return False
    
    width, height = image.size
    
    return (width >= min_size and height >= min_size and 
            width <= max_size and height <= max_size)

def resize_image_if_needed(image, max_dimension=1024):
    """
    Resize image if it exceeds maximum dimension while preserving aspect ratio.
    
    Args:
        image (PIL.Image): Input image
        max_dimension (int): Maximum allowed dimension
        
    Returns:
        PIL.Image: Resized image (if needed)
    """
    if image is None:
        return None
    
    width, height = image.size
    
    if width <= max_dimension and height <= max_dimension:
        return image
    
    # Calculate scaling factor
    scale_factor = max_dimension / max(width, height)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def cleanup_memory(device='auto'):
    """
    Clean up GPU/MPS memory to prevent accumulation.
    
    Args:
        device (str): Device type ('cuda', 'mps', 'cpu', or 'auto')
    """
    try:
        if device == 'auto':
            # Auto-detect device
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        # Clear PyTorch cache based on device
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif device == 'mps':
            torch.mps.empty_cache()
            torch.mps.synchronize()
        
        # Force garbage collection
        gc.collect()
        
    except Exception as e:
        print(f"Warning: Memory cleanup failed: {e}")

def save_image_with_metadata(image, output_path, metadata=None):
    """
    Save image with optional metadata preservation.
    
    Args:
        image (PIL.Image): Image to save
        output_path (str or Path): Output file path
        metadata (dict): Optional metadata to embed
        
    Returns:
        bool: True if save was successful
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if metadata:
            # Add metadata to image info
            pnginfo = None
            if output_path.suffix.lower() == '.png':
                from PIL.PngImagePlugin import PngInfo
                pnginfo = PngInfo()
                for key, value in metadata.items():
                    pnginfo.add_text(key, str(value))
            
            image.save(output_path, pnginfo=pnginfo)
        else:
            image.save(output_path)
        
        return True
        
    except Exception as e:
        print(f"Error saving image: {e}")
        return False

def load_image_safely(image_path):
    """
    Safely load an image with error handling.
    
    Args:
        image_path (str or Path): Path to image file
        
    Returns:
        PIL.Image or None: Loaded image or None if failed
    """
    try:
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return None
        
        image = Image.open(image_path)
        
        # Verify image can be loaded
        image.verify()
        
        # Reload image for actual use (verify() closes the file)
        image = Image.open(image_path)
        
        return image
        
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def get_image_info(image):
    """
    Get comprehensive information about an image.
    
    Args:
        image (PIL.Image): Image to analyze
        
    Returns:
        dict: Image information (size, mode, format, etc.)
    """
    if image is None:
        return {}
    
    info = {
        'size': image.size,
        'width': image.width,
        'height': image.height,
        'mode': image.mode,
        'format': image.format,
        'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info
    }
    
    return info

def create_thumbnail(image, size=(256, 256), maintain_aspect=True):
    """
    Create a thumbnail of the image.
    
    Args:
        image (PIL.Image): Input image
        size (tuple): Thumbnail size (width, height)
        maintain_aspect (bool): Whether to maintain aspect ratio
        
    Returns:
        PIL.Image: Thumbnail image
    """
    if image is None:
        return None
    
    try:
        thumbnail = image.copy()
        
        if maintain_aspect:
            thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
        else:
            thumbnail = thumbnail.resize(size, Image.Resampling.LANCZOS)
        
        return thumbnail
        
    except Exception as e:
        print(f"Error creating thumbnail: {e}")
        return None

def convert_image_format(image, target_format='RGB'):
    """
    Convert image to target format with proper handling.
    
    Args:
        image (PIL.Image): Input image
        target_format (str): Target format ('RGB', 'RGBA', 'L', etc.)
        
    Returns:
        PIL.Image: Converted image
    """
    if image is None:
        return None
    
    if image.mode == target_format:
        return image
    
    try:
        # Handle special conversions
        if target_format == 'RGB' and image.mode == 'RGBA':
            # Create white background for RGBA to RGB conversion
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
            return background
        else:
            return image.convert(target_format)
            
    except Exception as e:
        print(f"Error converting image format: {e}")
        return image