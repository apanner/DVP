"""
Image Utilities for VFX Formats
Supports EXR, JPG, TIFF, PNG with proper format handling
"""
import os
import glob
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

# Supported VFX formats
SUPPORTED_FORMATS = ['exr', 'EXR', 'jpg', 'jpeg', 'JPG', 'JPEG', 'tiff', 'tif', 'TIFF', 'TIF', 'png', 'PNG']


def detect_image_format(file_path: str) -> str:
    """
    Smart format detection from file extension
    Handles all common image formats automatically
    """
    ext = os.path.splitext(file_path)[1].lower().lstrip('.')
    
    # EXR formats
    if ext in ['exr']:
        return 'exr'
    # JPEG formats
    elif ext in ['jpg', 'jpeg']:
        return 'jpg'
    # TIFF formats
    elif ext in ['tiff', 'tif']:
        return 'tiff'
    # PNG formats
    elif ext in ['png']:
        return 'png'
    # Try to detect from file content if extension unknown
    else:
        # Try to read file header to detect format
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)
                # EXR magic number: 76 2F 31 01
                if header[:4] == b'\x76\x2f\x31\x01':
                    return 'exr'
                # PNG magic number: 89 50 4E 47
                elif header[:4] == b'\x89PNG':
                    return 'png'
                # JPEG magic number: FF D8 FF
                elif header[:3] == b'\xff\xd8\xff':
                    return 'jpg'
                # TIFF magic number: 49 49 2A 00 or 4D 4D 00 2A
                elif header[:4] in [b'II*\x00', b'MM\x00*']:
                    return 'tiff'
        except:
            pass
        
        # Default fallback
        logger.warning(f"Unknown format for {file_path}, defaulting to PNG")
        return 'png'


def auto_detect_format_from_directory(directory: str) -> Optional[str]:
    """
    Automatically detect the most common format in a directory
    Tries all supported formats and returns the most frequent one
    """
    if not os.path.isdir(directory):
        return None
    
    format_counts = {'exr': 0, 'jpg': 0, 'tiff': 0, 'png': 0}
    
    # Scan all supported formats
    for fmt_ext in ['exr', 'EXR', 'jpg', 'jpeg', 'JPG', 'JPEG', 
                     'tiff', 'tif', 'TIFF', 'TIF', 'png', 'PNG']:
        pattern = os.path.join(directory, f'*.{fmt_ext}')
        files = glob.glob(pattern)
        if files:
            detected = detect_image_format(files[0])
            format_counts[detected] += len(files)
    
    # Return most common format
    if any(format_counts.values()):
        most_common = max(format_counts.items(), key=lambda x: x[1])
        if most_common[1] > 0:
            logger.info(f"Auto-detected format: {most_common[0].upper()} ({most_common[1]} files)")
            return most_common[0]
    
    return None


def read_image_sequence(sequence_path: str, pattern: str = None, is_mask: bool = False) -> Tuple[List[Image.Image], List[dict], float, str]:
    """
    Smart image sequence reader - automatically detects and handles ANY format
    
    Supports: EXR, JPG, JPEG, TIFF, TIF, PNG (case-insensitive)
    Automatically detects format from files in directory
    
    Args:
        sequence_path: Path to directory or single file
        pattern: Optional file pattern (if None, auto-detects all formats)
        is_mask: Whether this is a mask sequence (affects color conversion)
        
    Returns:
        Tuple of (list of PIL Images, list of metadata dicts, fps estimate, format)
    """
    frames = []
    metadata_list = []
    format_type = None
    image_files = []
    
    if os.path.isfile(sequence_path):
        # Single file - detect format from file
        image_files = [sequence_path]
        format_type = detect_image_format(sequence_path)
        logger.info(f"Reading single file: {sequence_path} (format: {format_type.upper()})")
        
    elif os.path.isdir(sequence_path):
        # Directory with sequence - smart auto-detection
        if pattern:
            # Use provided pattern
            image_files = sorted(glob.glob(os.path.join(sequence_path, pattern)))
            if image_files:
                format_type = detect_image_format(image_files[0])
        else:
            # Auto-detect: try ALL supported formats automatically
            all_formats = ['*.exr', '*.EXR', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG', 
                          '*.tiff', '*.tif', '*.TIFF', '*.TIF', '*.png', '*.PNG']
            
            # Collect all image files
            for fmt_pattern in all_formats:
                found = glob.glob(os.path.join(sequence_path, fmt_pattern))
                image_files.extend(found)
            
            image_files = sorted(image_files)
            
            if image_files:
                # Smart format detection: use most common format in directory
                format_type = auto_detect_format_from_directory(sequence_path)
                if not format_type:
                    # Fallback: detect from first file
                    format_type = detect_image_format(image_files[0])
                logger.info(f"Auto-detected format: {format_type.upper()} ({len(image_files)} files)")
            else:
                # Try recursive search if no files found
                for fmt_pattern in all_formats:
                    found = glob.glob(os.path.join(sequence_path, '**', fmt_pattern), recursive=True)
                    image_files.extend(found)
                image_files = sorted(image_files)
                
                if image_files:
                    format_type = auto_detect_format_from_directory(sequence_path) or detect_image_format(image_files[0])
                    logger.info(f"Found files recursively: {format_type.upper()} ({len(image_files)} files)")
    else:
        raise ValueError(f"Invalid path: {sequence_path}")
    
    if not image_files:
        raise ValueError(f"No image files found in: {sequence_path}. Supported formats: EXR, JPG, TIFF, PNG")
    
    if not format_type or format_type == 'unknown':
        # Last resort: try to detect from first file content
        format_type = detect_image_format(image_files[0])
        if format_type == 'unknown':
            logger.warning(f"Could not detect format, defaulting to PNG")
            format_type = 'png'
    
    # Import EXR utils if needed
    if format_type == 'exr':
        try:
            from utils.exr_utils import read_exr_sequence, read_exr_mask_sequence
            if is_mask:
                # Use mask-specific EXR reader for masks
                frames = read_exr_mask_sequence(sequence_path, pattern or "*.exr")
                metadata_list = [{'path': '', 'format': 'exr', 'size': f.size} for f in frames]
                fps = 24.0
                return frames, metadata_list, fps, 'exr'
            else:
                frames, metadata_list, fps = read_exr_sequence(sequence_path, pattern or "*.exr")
                return frames, metadata_list, fps, 'exr'
        except ImportError:
            logger.warning("OpenEXR not available, trying PIL fallback")
            format_type = 'png'  # Fallback
        except Exception as e:
            logger.warning(f"EXR read failed: {e}, trying PIL fallback")
            format_type = 'png'  # Fallback
    
    # Read using PIL for standard formats
    for img_path in image_files:
        try:
            if format_type == 'exr':
                # Try to read EXR with PIL (may not work, but try)
                img = Image.open(img_path)
            else:
                img = Image.open(img_path)
            
            # Convert based on whether it's a mask or image
            if is_mask:
                # For masks, convert to grayscale (L mode)
                if img.mode != 'L':
                    img = img.convert('L')
            else:
                # For images, convert to RGB
                if img.mode != 'RGB':
                    if img.mode == 'RGBA':
                        # Create white background for RGBA
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                        img = background
                    else:
                        img = img.convert('RGB')
            
            frames.append(img)
            metadata_list.append({
                'path': img_path,
                'format': format_type,
                'size': img.size
            })
        except Exception as e:
            logger.warning(f"Failed to read {img_path}: {e}")
            continue
    
    # Estimate FPS (default 24 for image sequences)
    fps = 24.0
    
    return frames, metadata_list, fps, format_type


def write_image_sequence(frames: List[Image.Image], output_dir: str, 
                        format_type: str = 'exr',
                        metadata_list: Optional[List[dict]] = None,
                        prefix: str = "frame", start_frame: int = 0):
    """
    Write PIL Images to image sequence
    
    Args:
        frames: List of PIL Images
        output_dir: Output directory
        format_type: Output format ('exr', 'jpg', 'tiff', 'png')
        metadata_list: Optional metadata from original files
        prefix: Filename prefix
        start_frame: Starting frame number
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, pil_img in enumerate(frames):
        frame_num = start_frame + i
        
        if format_type.lower() == 'exr':
            # Use EXR utils
            try:
                from utils.exr_utils import write_exr_sequence
                # Write single frame
                write_exr_sequence([pil_img], output_dir, metadata_list, prefix, frame_num)
                continue
            except ImportError:
                logger.warning("OpenEXR not available, saving as PNG instead")
                format_type = 'png'
        
        # Determine extension
        ext_map = {
            'jpg': 'jpg',
            'jpeg': 'jpg',
            'tiff': 'tiff',
            'tif': 'tiff',
            'png': 'png'
        }
        ext = ext_map.get(format_type.lower(), 'png')
        
        # Save frame
        output_path = os.path.join(output_dir, f"{prefix}_{frame_num:04d}.{ext}")
        
        if format_type.lower() in ['jpg', 'jpeg']:
            pil_img.save(output_path, 'JPEG', quality=95)
        elif format_type.lower() in ['tiff', 'tif']:
            pil_img.save(output_path, 'TIFF', compression='lzw')
        else:  # PNG
            pil_img.save(output_path, 'PNG')


def resize_image_sequence(input_path: str, output_path: str, 
                         target_size: Tuple[int, int],
                         format_type: Optional[str] = None) -> Tuple[str, str]:
    """
    Smart resize function - automatically detects format and preserves it
    
    Args:
        input_path: Input sequence path (directory or file)
        output_path: Output directory for resized images
        target_size: Target (width, height)
        format_type: Optional format override (if None, auto-detects from input)
        
    Returns:
        Tuple of (output_directory, detected_format)
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Smart format detection: read sequence and auto-detect format
    frames, metadata_list, fps, detected_format = read_image_sequence(input_path)
    
    # Use provided format or auto-detected format
    output_format = format_type or detected_format
    
    if not output_format or output_format == 'unknown':
        # Try to detect from input path
        if os.path.isdir(input_path):
            output_format = auto_detect_format_from_directory(input_path) or 'png'
        else:
            output_format = detect_image_format(input_path) or 'png'
    
    logger.info(f"Resizing sequence: {input_path} → {output_path} (format: {output_format.upper()})")
    
    if not frames:
        raise ValueError(f"No frames found in: {input_path}")
    
    original_size = frames[0].size
    logger.info(f"Resizing {len(frames)} frames from {original_size[0]}x{original_size[1]} to {target_size[0]}x{target_size[1]}")
    
    # Get file list for output naming
    if os.path.isdir(input_path):
        import glob
        pattern_map = {
            'exr': '*.exr',
            'jpg': '*.jpg',
            'jpeg': '*.jpg',
            'tiff': '*.tiff',
            'tif': '*.tiff',
            'png': '*.png'
        }
        pattern = pattern_map.get(output_format.lower(), '*.exr')
        input_files = sorted(glob.glob(os.path.join(input_path, pattern)))
    else:
        input_files = [input_path]
    
    # Resize and save frames using OpenCV
    for i, frame in enumerate(frames):
        # Convert PIL to numpy array (RGB)
        img_np = np.array(frame)
        
        # Convert RGB to BGR for OpenCV
        if len(img_np.shape) == 3:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_np
        
        # Resize using OpenCV (fast and simple)
        resized_bgr = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Convert back to RGB
        if len(resized_bgr.shape) == 3:
            resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        else:
            resized_rgb = resized_bgr
        
        # Convert back to PIL Image
        resized_pil = Image.fromarray(resized_rgb)
        
        # Determine output filename
        if input_files and i < len(input_files):
            base_name = os.path.splitext(os.path.basename(input_files[i]))[0]
            ext = output_format.lower()
            if ext == 'jpeg':
                ext = 'jpg'
            output_filename = f"{base_name}.{ext}"
        else:
            output_filename = f"frame_{i:04d}.{output_format.lower()}"
        
        output_file = os.path.join(output_path, output_filename)
        
        # Save based on format
        if output_format.lower() == 'exr':
            # Use EXR utils for EXR
            try:
                from utils.exr_utils import write_exr_file, srgb_to_linear
                # Convert PIL to numpy for EXR writing
                img_array = np.array(resized_pil).astype(np.float32) / 255.0
                # Convert sRGB to linear for EXR
                img_linear = srgb_to_linear(img_array)
                # Get metadata if available
                metadata = metadata_list[i] if i < len(metadata_list) else None
                write_exr_file(output_file, img_linear, metadata)
            except Exception as e:
                logger.error(f"Failed to write EXR: {e}, falling back to PNG")
                resized_pil.save(output_file.replace('.exr', '.png'), 'PNG')
        elif output_format.lower() in ['jpg', 'jpeg']:
            resized_pil.save(output_file, 'JPEG', quality=95)
        elif output_format.lower() in ['tiff', 'tif']:
            resized_pil.save(output_file, 'TIFF', compression='lzw')
        else:  # PNG
            resized_pil.save(output_file, 'PNG')
    
    logger.info(f"Resized sequence saved to: {output_path}")
    
    return output_path, output_format


def get_sequence_resolution(sequence_path: str) -> Optional[Tuple[int, int]]:
    """
    Get resolution of first frame in sequence
    
    Args:
        sequence_path: Path to sequence
        
    Returns:
        (width, height) or None if cannot read
    """
    try:
        frames, _, _, _ = read_image_sequence(sequence_path)
        if frames:
            return frames[0].size
    except Exception as e:
        logger.warning(f"Could not get resolution from {sequence_path}: {e}")
    
    return None

