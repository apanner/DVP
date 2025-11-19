"""
EXR Utilities for VFX Production
Handles reading/writing EXR image sequences with proper color space conversion
USES OPENCV ONLY - No OpenEXR dependency
"""
import os
import glob
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


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
    Read a single EXR file using OpenCV
    Args:
        exr_path: Path to EXR file
    Returns:
        Tuple of (image array [H, W, C] in linear color space, metadata dict)
    """
    if not os.path.exists(exr_path):
        raise FileNotFoundError(f"EXR file not found: {exr_path}")
    
    # Read EXR with OpenCV (IMREAD_ANYDEPTH preserves float values)
    img_cv = cv2.imread(exr_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    
    if img_cv is None:
        raise ValueError(f"OpenCV failed to read EXR file: {exr_path}")
    
    # Convert BGR to RGB if color image
    if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # Handle float images (EXR uses float)
    if img_cv.dtype != np.uint8:
        # Normalize float images to [0, 1] range
        if img_cv.max() > 1.0:
            # Values are in high range (e.g., 0-65535 or 0-10000)
            # Normalize to [0, 1] for linear color space
            img_float = img_cv.astype(np.float32) / img_cv.max()
        else:
            # Values are already in [0, 1] range
            img_float = img_cv.astype(np.float32)
    else:
        # uint8 - convert to float32 [0, 1]
        img_float = img_cv.astype(np.float32) / 255.0
    
    height, width = img_float.shape[:2]
    
    # Store metadata
    metadata = {
        'width': width,
        'height': height,
        'channels': ['R', 'G', 'B'] if len(img_float.shape) == 3 else ['Y'],
        'pixel_type': 'FLOAT',
        'header': {}
    }
    
    return img_float, metadata


def write_exr_file(output_path: str, img: np.ndarray, metadata: Optional[dict] = None):
    """
    Write an image array to EXR file using OpenCV
    Args:
        output_path: Output EXR file path
        img: Image array in linear color space [H, W, 3] or [H, W] as numpy array
        metadata: Optional metadata dict (not used with OpenCV, kept for compatibility)
    """
    # Ensure float32
    img = img.astype(np.float32)
    
    # Handle different image shapes
    if len(img.shape) == 2:
        # Grayscale - convert to RGB
        img = np.stack([img, img, img], axis=2)
    elif len(img.shape) == 3:
        if img.shape[2] == 1:
            # Single channel - convert to RGB
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] == 4:
            # RGBA - take RGB only
            img = img[:, :, :3]
    
    # Normalize to [0, 1] if values are outside range
    img_max = img.max()
    if img_max > 1.0:
        img = img / img_max
    
    # Convert to uint16 for EXR-like storage (OpenCV EXR support)
    # Scale to 0-65535 range
    img_uint16 = (np.clip(img, 0, 1) * 65535).astype(np.uint16)
    
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_uint16, cv2.COLOR_RGB2BGR)
    
    # Write EXR file (OpenCV supports EXR writing)
    # Note: OpenCV's EXR writing may not preserve all metadata, but works for basic use
    success = cv2.imwrite(output_path, img_bgr)
    
    if not success:
        # Fallback: Write as TIFF (lossless, supports float)
        tiff_path = output_path.replace('.exr', '.tiff')
        logger.warning(f"OpenCV EXR write failed for {output_path}, writing as TIFF: {tiff_path}")
        cv2.imwrite(tiff_path, img_bgr)
        raise ValueError(f"Failed to write EXR file: {output_path}")


def read_exr_sequence(sequence_path: str, pattern: str = "*.exr") -> Tuple[List[Image.Image], List[dict], float]:
    """
    Read EXR image sequence using OpenCV
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
        try:
            # Read EXR (linear color space) using OpenCV
            img_linear, metadata = read_exr_file(exr_path)
            
            # Convert linear to sRGB for model processing
            img_srgb = linear_to_srgb(img_linear)
            
            # Convert to uint8 [0, 255] for PIL
            img_uint8 = (np.clip(img_srgb, 0, 1) * 255).astype(np.uint8)
            
            # Create PIL Image
            pil_img = Image.fromarray(img_uint8, 'RGB')
            
            frames.append(pil_img)
            metadata_list.append(metadata)
        except Exception as e:
            logger.warning(f"Failed to read EXR file {exr_path}: {e}")
            # Skip this file and continue
            continue
    
    if not frames:
        raise ValueError(f"No EXR files could be read from: {sequence_path}")
    
    # Estimate FPS (default 24 for image sequences)
    fps = 24.0
    
    return frames, metadata_list, fps


def write_exr_sequence(frames: List[Image.Image], output_dir: str, 
                      metadata_list: Optional[List[dict]] = None,
                      prefix: str = "frame", start_frame: int = 0):
    """
    Write PIL Images to EXR sequence using OpenCV
    Args:
        frames: List of PIL Images in sRGB [0, 255]
        output_dir: Output directory
        metadata_list: Optional list of metadata dicts (not used with OpenCV, kept for compatibility)
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
        
        # Write EXR file
        output_path = os.path.join(output_dir, f"{prefix}_{frame_num:04d}.exr")
        write_exr_file(output_path, img_linear, metadata_list[i] if metadata_list and i < len(metadata_list) else None)


def read_exr_mask_sequence(mask_path: str, pattern: str = "*.exr") -> List[Image.Image]:
    """
    Read EXR mask sequence (grayscale) using OpenCV
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
        if not exr_files:
            # Try common patterns
            for pat in ["*.exr", "*.EXR", "mask.*.exr", "*.mask.exr"]:
                exr_files = sorted(glob.glob(os.path.join(mask_path, pat)))
                if exr_files:
                    break
    else:
        raise ValueError(f"Invalid mask path: {mask_path}")
    
    if not exr_files:
        raise ValueError(f"No EXR mask files found in: {mask_path}")
    
    for exr_path in exr_files:
        try:
            # Read EXR using OpenCV
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
        except Exception as e:
            logger.warning(f"Failed to read EXR mask file {exr_path}: {e}")
            # Skip this file and continue
            continue
    
    if not frames:
        raise ValueError(f"No EXR mask files could be read from: {mask_path}")
    
    return frames
