"""
EXR Utilities for VFX Production
Handles reading/writing EXR image sequences with proper color space conversion
"""
import os
import glob
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Optional


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """
    Convert linear RGB to sRGB color space
    Args:
        linear: Array in linear color space [0, inf]
    Returns:
        Array in sRGB color space [0, 1]
    """
    linear = np.clip(linear, 0, None)
    mask = linear <= 0.0031308
    srgb = np.where(
        mask,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055
    )
    return np.clip(srgb, 0, 1)


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """
    Convert sRGB to linear RGB color space
    Args:
        srgb: Array in sRGB color space [0, 1]
    Returns:
        Array in linear color space [0, inf]
    """
    srgb = np.clip(srgb, 0, 1)
    mask = srgb <= 0.04045
    linear = np.where(
        mask,
        srgb / 12.92,
        np.power((srgb + 0.055) / 1.055, 2.4)
    )
    return np.clip(linear, 0, None)


def read_exr_file(exr_path: str) -> Tuple[np.ndarray, dict]:
    """
    Read a single EXR file using OpenCV.
    Args:
        exr_path: Path to EXR file
    Returns:
        Tuple of (image array [H, W, C], metadata dict)
    """
    if not os.path.exists(exr_path):
        raise FileNotFoundError(f"EXR file not found: {exr_path}")
    
    img = cv2.imread(exr_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise ValueError(f"cv2.imread failed to load EXR: {exr_path}")
    
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if img.dtype in (np.uint16, np.uint8):
        max_val = np.iinfo(img.dtype).max
        img = img.astype(np.float32) / max_val
    else:
        img = img.astype(np.float32)
    
    height, width = img.shape[:2]
    metadata = {
        'width': width,
        'height': height,
        'channels': ['R', 'G', 'B'] if img.shape[2] == 3 else ['Y'],
        'pixel_type': str(img.dtype),
        'source': 'opencv'
    }
    return img, metadata


def write_exr_file(output_path: str, img: np.ndarray, metadata: Optional[dict] = None):
    """
    Write an image array to EXR file
    Args:
        output_path: Output EXR file path
        img: Image array in linear color space [H, W, 3] or [H, W] as numpy array
        metadata: Optional metadata dict from original EXR
    """
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    
    img = img.astype(np.float32)
    
    if img.shape[2] == 3:
        img_to_write = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_to_write = img
    
    success = cv2.imwrite(output_path, img_to_write)
    if not success:
        raise RuntimeError(f"Failed to write EXR file: {output_path}")


def read_exr_sequence(sequence_path: str, pattern: str = "*.exr") -> Tuple[List[Image.Image], List[dict], float]:
    """
    Read EXR image sequence
    Args:
        sequence_path: Path to directory containing EXR files or single EXR file
        pattern: File pattern for sequence (e.g., "*.exr", "frame_*.exr")
    Returns:
        Tuple of (list of PIL Images in sRGB, list of metadata dicts, fps estimate)
    """
    frames = []
    metadata_list = []
    
    if os.path.isfile(sequence_path):
        # Single file
        exr_files = [sequence_path]
    elif os.path.isdir(sequence_path):
        # Directory with sequence
        exr_files = sorted(glob.glob(os.path.join(sequence_path, pattern)))
        if not exr_files:
            # Try common patterns
            for pat in ["*.exr", "*.EXR", "frame_*.exr", "*.%04d.exr"]:
                exr_files = sorted(glob.glob(os.path.join(sequence_path, pat)))
                if exr_files:
                    break
    else:
        raise ValueError(f"Invalid path: {sequence_path}")
    
    if not exr_files:
        raise ValueError(f"No EXR files found in: {sequence_path}")
    
    for exr_path in exr_files:
        # Read EXR (linear color space)
        img_linear, metadata = read_exr_file(exr_path)
        
        # Convert linear to sRGB for model processing
        img_srgb = linear_to_srgb(img_linear)
        
        # Convert to uint8 [0, 255] for PIL
        img_uint8 = (np.clip(img_srgb, 0, 1) * 255).astype(np.uint8)
        
        # Create PIL Image
        pil_img = Image.fromarray(img_uint8, 'RGB')
        
        frames.append(pil_img)
        metadata_list.append(metadata)
    
    # Estimate FPS (default 24 for image sequences)
    fps = 24.0
    
    return frames, metadata_list, fps


def write_exr_sequence(frames: List[Image.Image], output_dir: str, 
                      metadata_list: Optional[List[dict]] = None,
                      prefix: str = "frame", start_frame: int = 0):
    """
    Write PIL Images to EXR sequence
    Args:
        frames: List of PIL Images in sRGB [0, 255]
        output_dir: Output directory
        metadata_list: Optional list of metadata dicts from original EXR files
        prefix: Filename prefix
        start_frame: Starting frame number
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, pil_img in enumerate(frames):
        frame_num = start_frame + i
        
        # Convert PIL to numpy array [0, 255] -> [0, 1]
        img_srgb = np.array(pil_img).astype(np.float32) / 255.0
        
        # Ensure RGB
        if len(img_srgb.shape) == 2:
            # Grayscale to RGB
            img_srgb = np.stack([img_srgb, img_srgb, img_srgb], axis=2)
        elif img_srgb.shape[2] == 4:
            # RGBA to RGB
            img_srgb = img_srgb[:, :, :3]
        
        # Convert sRGB to linear
        img_linear = srgb_to_linear(img_srgb)
        
        # Get metadata if available
        metadata = None
        if metadata_list and i < len(metadata_list):
            metadata = metadata_list[i]
        
        # Write EXR file
        output_path = os.path.join(output_dir, f"{prefix}_{frame_num:04d}.exr")
        write_exr_file(output_path, img_linear, metadata)


def read_exr_mask_sequence(mask_path: str, pattern: str = "*.exr") -> List[Image.Image]:
    """
    Read EXR mask sequence (grayscale)
    Args:
        mask_path: Path to directory or single EXR file
        pattern: File pattern
    Returns:
        List of PIL Images (grayscale masks)
    """
    frames = []
    
    if os.path.isfile(mask_path):
        exr_files = [mask_path]
    elif os.path.isdir(mask_path):
        exr_files = sorted(glob.glob(os.path.join(mask_path, pattern)))
    else:
        raise ValueError(f"Invalid mask path: {mask_path}")
    
    for exr_path in exr_files:
        # Read EXR
        img_linear, _ = read_exr_file(exr_path)
        
        # If RGB, convert to grayscale
        if len(img_linear.shape) == 3 and img_linear.shape[2] == 3:
            # Use luminance: 0.2126*R + 0.7152*G + 0.0722*B
            img_gray = (0.2126 * img_linear[:, :, 0] + 
                       0.7152 * img_linear[:, :, 1] + 
                       0.0722 * img_linear[:, :, 2])
        else:
            img_gray = img_linear[:, :, 0] if len(img_linear.shape) == 3 else img_linear
        
        # Normalize to [0, 1] then to [0, 255]
        img_gray = np.clip(img_gray, 0, None)
        img_max = img_gray.max()
        if img_max > 0:
            img_gray = img_gray / img_max
        
        img_uint8 = (img_gray * 255).astype(np.uint8)
        
        # Create PIL Image (grayscale)
        pil_img = Image.fromarray(img_uint8, 'L')
        frames.append(pil_img)
    
    return frames

