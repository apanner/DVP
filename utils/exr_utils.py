"""
EXR Utilities for VFX Production
Handles reading/writing EXR image sequences with proper color space conversion
USES OpenEXR Python bindings for EXR (OpenCV from pip doesn't have EXR support)
Falls back to OpenCV for other formats

NOTE: Supports both installation methods:
  - Separate packages: pip install Imath OpenEXR (works for all versions)
  - Newer versions: pip install OpenEXR>=3.4.0 (includes Imath)
"""
import os
import glob
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import OpenEXR - required for EXR file reading
# OpenCV from pip doesn't have EXR support compiled in
# Handles both:
#   - Separate Imath package (older versions: OpenEXR==1.3.2 + Imath==0.0.2)
#   - Imath included in OpenEXR (newer versions: OpenEXR>=3.4.0)
try:
    import OpenEXR
    
    # Try importing Imath - handle both separate package and included module
    try:
        import Imath  # Separate package (older versions)
    except ImportError:
        # Try Imath as submodule of OpenEXR (newer versions >=3.4.0)
        if hasattr(OpenEXR, 'Imath'):
            Imath = OpenEXR.Imath
        else:
            # Last resort: try importing from OpenEXR namespace
            try:
                from OpenEXR import Imath
            except ImportError:
                raise ImportError("Imath module not found. Install: pip install Imath OpenEXR")
    
    OPENEXR_AVAILABLE = True
except ImportError as e:
    OPENEXR_AVAILABLE = False
    logger.warning(f"OpenEXR not available - EXR files cannot be read. Install: pip install Imath OpenEXR (or OpenEXR>=3.4.0)")


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
    Read a single EXR file using OpenEXR Python bindings
    (OpenCV from pip doesn't have EXR support compiled in)
    Args:
        exr_path: Path to EXR file
    Returns:
        Tuple of (image array [H, W, C] in linear color space, metadata dict)
    """
    if not os.path.exists(exr_path):
        raise FileNotFoundError(f"EXR file not found: {exr_path}")
    
    if not OPENEXR_AVAILABLE:
        raise ImportError("OpenEXR Python bindings not available. Install: pip install Imath OpenEXR (or OpenEXR>=3.4.0)")
    
    # Read EXR using OpenEXR (proper method for EXR files)
    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()
    
    # Get image dimensions
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # Determine channels
    channels = header['channels']
    channel_names = list(channels.keys())
    
    # Read RGB channels (handle different channel names)
    r_channel = None
    g_channel = None
    b_channel = None
    
    for ch_name in ['R', 'Red', 'RED', 'r']:
        if ch_name in channel_names:
            r_channel = ch_name
            break
    
    for ch_name in ['G', 'Green', 'GREEN', 'g']:
        if ch_name in channel_names:
            g_channel = ch_name
            break
    
    for ch_name in ['B', 'Blue', 'BLUE', 'b']:
        if ch_name in channel_names:
            b_channel = ch_name
            break
    
    # If RGB not found, try to use first available channels or grayscale
    if not all([r_channel, g_channel, b_channel]):
        if len(channel_names) == 1:
            # Single channel (grayscale) - duplicate to RGB
            single_channel = channel_names[0]
            pixel_type = channels[single_channel].type
            ch_str = exr_file.channel(single_channel, Imath.PixelType(Imath.PixelType.FLOAT))
            ch_data = np.frombuffer(ch_str, dtype=np.float32).reshape(height, width)
            img_float = np.stack([ch_data, ch_data, ch_data], axis=2)
        else:
            # Try to use first 3 channels
            if len(channel_names) >= 3:
                r_channel = channel_names[0]
                g_channel = channel_names[1]
                b_channel = channel_names[2]
            else:
                raise ValueError(f"Could not find RGB channels in EXR file. Available: {channel_names}")
    
    # Read pixel data as float32
    if r_channel and g_channel and b_channel:
        r_str = exr_file.channel(r_channel, Imath.PixelType(Imath.PixelType.FLOAT))
        g_str = exr_file.channel(g_channel, Imath.PixelType(Imath.PixelType.FLOAT))
        b_str = exr_file.channel(b_channel, Imath.PixelType(Imath.PixelType.FLOAT))
        
        r = np.frombuffer(r_str, dtype=np.float32).reshape(height, width)
        g = np.frombuffer(g_str, dtype=np.float32).reshape(height, width)
        b = np.frombuffer(b_str, dtype=np.float32).reshape(height, width)
        
        # Stack into RGB image
        img_float = np.stack([r, g, b], axis=2)
    
    # Store metadata
    pixel_type = channels[channel_names[0]].type if channel_names else None
    metadata = {
        'width': width,
        'height': height,
        'channels': channel_names,
        'pixel_type': str(pixel_type) if pixel_type else 'FLOAT',
        'header': header
    }
    
    exr_file.close()
    return img_float, metadata


def write_exr_file(output_path: str, img: np.ndarray, metadata: Optional[dict] = None):
    """
    Write an image array to EXR file using OpenEXR Python bindings
    (OpenCV from pip doesn't have EXR support compiled in)
    Args:
        output_path: Output EXR file path
        img: Image array in linear color space [H, W, 3] or [H, W] as numpy array
        metadata: Optional metadata dict from original EXR
    """
    if not OPENEXR_AVAILABLE:
        raise ImportError("OpenEXR Python bindings not available. Install: pip install Imath OpenEXR (or OpenEXR>=3.4.0)")
    
    # Ensure float32
    img = img.astype(np.float32)
    
    # Handle different image shapes
    if len(img.shape) == 2:
        # Grayscale
        height, width = img.shape
        channels = {'Y': img}
    elif len(img.shape) == 3:
        # RGB
        height, width, channels_count = img.shape
        if channels_count == 3:
            channels = {
                'R': img[:, :, 0],
                'G': img[:, :, 1],
                'B': img[:, :, 2]
            }
        elif channels_count == 1:
            channels = {'Y': img[:, :, 0]}
        else:
            raise ValueError(f"Unsupported channel count: {channels_count}")
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    
    # Create header
    header = OpenEXR.Header(width, height)
    if metadata and 'header' in metadata:
        # Copy relevant attributes from original
        orig_header = metadata['header']
        if 'compression' in orig_header:
            header['compression'] = orig_header['compression']
    
    # Set pixel type to FLOAT
    header['channels'] = {ch: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) 
                          for ch in channels.keys()}
    
    # Create EXR file
    exr_file = OpenEXR.OutputFile(output_path, header)
    
    # Prepare channel data as float32 arrays
    channel_arrays = {}
    for ch_name, ch_data in channels.items():
        ch_data_float = ch_data.astype(np.float32)
        # Ensure data is contiguous
        if not ch_data_float.flags['C_CONTIGUOUS']:
            ch_data_float = np.ascontiguousarray(ch_data_float)
        channel_arrays[ch_name] = ch_data_float
    
    # Write scanline by scanline (OpenEXR Python API requirement)
    for y in range(height):
        scanline_dict = {}
        for ch_name, ch_array in channel_arrays.items():
            # Extract one scanline: y-th row
            scanline = ch_array[y, :].astype(np.float32)
            # Ensure contiguous
            if not scanline.flags['C_CONTIGUOUS']:
                scanline = np.ascontiguousarray(scanline)
            # Convert to bytes
            scanline_bytes = scanline.tobytes()
            scanline_dict[ch_name] = scanline_bytes
        exr_file.writePixels(scanline_dict)
    
    exr_file.close()


def read_exr_sequence(sequence_path: str, pattern: str = "*.exr") -> Tuple[List[Image.Image], List[dict], float]:
    """
    Read EXR image sequence using OpenEXR Python bindings
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
            # Read EXR (linear color space) using OpenEXR
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
    Write PIL Images to EXR sequence using OpenEXR Python bindings
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
            # Read EXR using OpenEXR
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
