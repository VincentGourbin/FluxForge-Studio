"""
Canny Processing Module

Canny edge detection preprocessing for ControlNet operations.
Provides optimized edge detection with configurable thresholds.

Features:
- Configurable low and high thresholds
- RGB output compatible with ControlNet
- Memory-efficient processing
- Preview generation for real-time feedback

Author: MFLUX Team
"""

import cv2
import numpy as np
from PIL import Image

def preprocess_canny(img, low_threshold=100, high_threshold=200):
    """
    Preprocess image for Canny ControlNet by generating edge detection.
    
    Args:
        img (PIL.Image): Input image to process
        low_threshold (int): Lower threshold for edge detection (default: 100)
        high_threshold (int): Higher threshold for edge detection (default: 200)
        
    Returns:
        PIL.Image: Black and white edge detection image suitable for Canny ControlNet
    """
    try:
        # Convert PIL image to numpy array
        image_array = np.array(img)
        
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection with configurable thresholds
        canny_edges = cv2.Canny(image_array, low_threshold, high_threshold)
        
        # Convert to PIL Image first (single channel)
        canny_pil = Image.fromarray(canny_edges, mode='L')
        
        # Convert to RGB (this matches the expected format for control models)
        canny_rgb = canny_pil.convert('RGB')
        
        return canny_rgb
        
    except Exception as e:
        print(f"Error in Canny preprocessing: {e}")
        return None

def generate_canny_preview(input_image, low_threshold=100, high_threshold=200):
    """
    Generate Canny edge preview for real-time UI feedback.
    
    Args:
        input_image (str or PIL.Image): Input image (file path or PIL Image)
        low_threshold (int): Lower threshold for edge detection
        high_threshold (int): Higher threshold for edge detection
        
    Returns:
        PIL.Image or None: Canny edge preview image
    """
    try:
        if input_image is None:
            return None
        
        # Handle both file paths and PIL Images
        if isinstance(input_image, str):
            img = Image.open(input_image)
        else:
            img = input_image
        
        # Generate Canny edges
        canny_result = preprocess_canny(img, low_threshold, high_threshold)
        
        return canny_result
        
    except Exception as e:
        print(f"Error generating Canny preview: {e}")
        return None

def auto_canny_thresholds(image, sigma=0.33):
    """
    Automatically determine optimal Canny thresholds using Otsu's method.
    
    Args:
        image (PIL.Image): Input image
        sigma (float): Sigma value for threshold calculation (default: 0.33)
        
    Returns:
        tuple: (low_threshold, high_threshold) optimal values
    """
    try:
        # Convert to grayscale numpy array
        gray = np.array(image.convert('L'))
        
        # Compute the median of the single channel pixel intensities
        median_intensity = np.median(gray)
        
        # Apply automatic Canny edge detection using the computed median
        low_threshold = int(max(0, (1.0 - sigma) * median_intensity))
        high_threshold = int(min(255, (1.0 + sigma) * median_intensity))
        
        return low_threshold, high_threshold
        
    except Exception as e:
        print(f"Error in auto threshold calculation: {e}")
        return 100, 200  # Return default values

def validate_canny_thresholds(low_threshold, high_threshold):
    """
    Validate and adjust Canny threshold values.
    
    Args:
        low_threshold (int): Lower threshold value
        high_threshold (int): Higher threshold value
        
    Returns:
        tuple: (validated_low, validated_high) corrected threshold values
    """
    # Ensure thresholds are within valid range
    low_threshold = max(0, min(255, int(low_threshold)))
    high_threshold = max(0, min(255, int(high_threshold)))
    
    # Ensure high threshold is greater than low threshold
    if high_threshold <= low_threshold:
        high_threshold = low_threshold + 50
        if high_threshold > 255:
            high_threshold = 255
            low_threshold = max(0, high_threshold - 50)
    
    return low_threshold, high_threshold

def enhance_canny_edges(canny_image, dilation_kernel_size=3):
    """
    Enhance Canny edges using morphological operations.
    
    Args:
        canny_image (PIL.Image): Input Canny edge image
        dilation_kernel_size (int): Size of dilation kernel for edge enhancement
        
    Returns:
        PIL.Image: Enhanced edge image
    """
    try:
        # Convert to numpy array
        canny_array = np.array(canny_image.convert('L'))
        
        # Create dilation kernel
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        
        # Apply morphological dilation to enhance edges
        enhanced_edges = cv2.dilate(canny_array, kernel, iterations=1)
        
        # Convert back to RGB PIL Image
        enhanced_rgb = np.stack([enhanced_edges, enhanced_edges, enhanced_edges], axis=2)
        
        return Image.fromarray(enhanced_rgb)
        
    except Exception as e:
        print(f"Error enhancing Canny edges: {e}")
        return canny_image