"""Advanced image preprocessing for better OCR accuracy."""

import cv2
import numpy as np
from PIL import Image


def enhance_image_for_ocr(image, enhance_contrast=True, denoise=True, sharpen=False):
    """
    Enhance image for better OCR accuracy.
    
    Args:
        image: PIL Image or numpy array
        enhance_contrast: Apply CLAHE for better contrast
        denoise: Apply denoising filter
        sharpen: Apply sharpening filter
    
    Returns:
        Enhanced PIL Image
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
        if img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    else:
        img_array = image.copy()
    
    # Convert RGB to BGR for OpenCV
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
    
    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    
    # Denoise
    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Sharpen
    if sharpen:
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        gray = cv2.filter2D(gray, -1, kernel)
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    # Convert back to PIL Image
    return Image.fromarray(enhanced)


def optimize_image_size(image, max_dimension=2048, min_dimension=512, maintain_aspect=True):
    """
    Optimize image size for faster processing while maintaining quality.
    
    Args:
        image: PIL Image
        max_dimension: Maximum width or height
        min_dimension: Minimum width or height (won't upscale)
        maintain_aspect: Maintain aspect ratio
    
    Returns:
        Resized PIL Image
    """
    width, height = image.size
    
    # Don't upscale small images
    if width < min_dimension and height < min_dimension:
        return image
    
    # Calculate new size
    if maintain_aspect:
        if width > height:
            if width > max_dimension:
                ratio = max_dimension / width
                new_width = max_dimension
                new_height = int(height * ratio)
            else:
                return image
        else:
            if height > max_dimension:
                ratio = max_dimension / height
                new_height = max_dimension
                new_width = int(width * ratio)
            else:
                return image
    else:
        new_width = min(width, max_dimension)
        new_height = min(height, max_dimension)
    
    # Use high-quality resampling
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def preprocess_receipt_image(image_path, enhance=True, optimize=True):
    """
    Complete preprocessing pipeline for receipt images.
    
    Args:
        image_path: Path to image file
        enhance: Apply image enhancement
        optimize: Optimize image size
    
    Returns:
        Preprocessed PIL Image
    """
    image = Image.open(image_path)
    
    if optimize:
        image = optimize_image_size(image)
    
    if enhance:
        image = enhance_image_for_ocr(image, enhance_contrast=True, denoise=True, sharpen=False)
    
    return image


