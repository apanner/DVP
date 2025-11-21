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

# Enable OpenEXR support for OpenCV (like VDA/iMatte does)
# This allows cv2.imread/cv2.imwrite to handle EXR files directly
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

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
    
    # Use OpenImageIO for EXR files (works great in Colab)
    if format_type == 'exr':
        try:
            import OpenImageIO as oiio
            
            logger.info(f"Reading EXR sequence with OpenImageIO: {sequence_path}")
            frames = []
            metadata_list = []
            
            for img_path in image_files:
                try:
                    # Read EXR with OpenImageIO
                    img_buf = oiio.ImageBuf(str(img_path))
                    if img_buf.has_error:
                        logger.warning(f"Failed to read EXR: {img_buf.geterror()}")
                        continue
                    
                    # Get image specs
                    spec = img_buf.spec()
                    width = spec.width
                    height = spec.height
                    nchannels = spec.nchannels
                    
                    # Get pixel data as numpy array (float32)
                    img_array = img_buf.get_pixels(oiio.FLOAT)
                    img = np.array(img_array).reshape((height, width, nchannels))
                    
                    # Handle different channel formats
                    if nchannels == 1:
                        # Grayscale - convert to RGB
                        img = np.stack([img[:, :, 0], img[:, :, 0], img[:, :, 0]], axis=2)
                    elif nchannels == 3:
                        # RGB - keep as is
                        pass
                    elif nchannels == 4:
                        # RGBA to RGB (drop alpha)
                        img = img[:, :, :3]
                    elif nchannels > 4:
                        # Multi-channel - take first 3
                        img = img[:, :, :3]
                    
                    # Convert to PIL Image
                    # OpenImageIO reads as float32, may have HDR values > 1
                    # Normalize to 0-1 range for PIL (clip HDR values)
                    img_normalized = np.clip(img, 0, None)
                    img_max = img_normalized.max()
                    if img_max > 1.0:
                        # HDR - normalize to 0-1 for PIL
                        img_normalized = img_normalized / img_max
                    
                    # Convert to uint8 for PIL
                    img_uint8 = (img_normalized * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_uint8)
                    
                    # Convert based on whether it's a mask or image
                    if is_mask:
                        # For masks, convert to grayscale (L mode)
                        if pil_img.mode != 'L':
                            pil_img = pil_img.convert('L')
                    else:
                        # For images, ensure RGB
                        if pil_img.mode != 'RGB':
                            pil_img = pil_img.convert('RGB')
                    
                    frames.append(pil_img)
                    metadata_list.append({
                        'path': img_path,
                        'format': 'exr',
                        'size': (width, height)
                    })
                except Exception as e:
                    logger.warning(f"Failed to read EXR {img_path}: {e}")
                    continue
            
            if frames:
                fps = 24.0
                logger.info(f"✅ Read {len(frames)} EXR frames with OpenImageIO")
                return frames, metadata_list, fps, 'exr'
            else:
                logger.warning("No EXR frames could be read, trying fallback")
                format_type = 'png'  # Fallback
        except ImportError:
            logger.warning("OpenImageIO not available, trying OpenEXR Python bindings...")
            try:
                from utils.exr_utils import read_exr_sequence, read_exr_mask_sequence
                if is_mask:
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
        except Exception as e:
            logger.warning(f"OpenImageIO EXR read failed: {e}, trying fallback")
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
    Resize sequence ONE FILE AT A TIME (read -> resize -> write -> next)
    This avoids loading all frames into memory, which can hang on Windows
    
    Args:
        input_path: Input sequence path (directory or file)
        output_path: Output directory for resized images
        target_size: Target (width, height)
        format_type: Optional format override (if None, auto-detects from input)
        
    Returns:
        Tuple of (output_directory, detected_format)
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Detect format from first file (fast - no full read)
    if os.path.isdir(input_path):
        # Find first file
        for pattern in ['*.exr', '*.EXR', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG',
                       '*.tiff', '*.TIFF', '*.tif', '*.TIF', '*.png', '*.PNG']:
            files = sorted(glob.glob(os.path.join(input_path, pattern)))
            if files:
                first_file = files[0]
                break
        else:
            raise ValueError(f"No image files found in: {input_path}")
    
        # Get all files of same format
        detected_format = detect_image_format(first_file)
        pattern_map = {
            'exr': '*.exr',
            'jpg': '*.jpg',
            'jpeg': '*.jpg',
            'tiff': '*.tiff',
            'tif': '*.tiff',
            'png': '*.png'
        }
        pattern = pattern_map.get(detected_format.lower(), '*.exr')
        input_files = sorted(glob.glob(os.path.join(input_path, pattern)))
    else:
        input_files = [input_path]
        detected_format = detect_image_format(input_path)
    
    # Use provided format or detected
    output_format = format_type or detected_format
    if not output_format or output_format == 'unknown':
        output_format = 'png'
    
    logger.info(f"Resizing sequence: {input_path} → {output_path}")
    logger.info(f"Format: {output_format.upper()}, Files: {len(input_files)}, Target: {target_size[0]}x{target_size[1]}")
    
    if not input_files:
        raise ValueError(f"No files found in: {input_path}")
        
    # Process ONE FILE AT A TIME (read -> resize -> write -> next)
    def process_single_file(file_path, file_index):
        """Process a single file: read -> resize -> write"""
        try:
            filename = os.path.basename(file_path)
            logger.info(f"   Processing [{file_index+1}/{len(input_files)}]: {filename}...")
            
            # Determine output filename
            base_name = os.path.splitext(filename)[0]
            ext = output_format.lower()
            if ext == 'jpeg':
                ext = 'jpg'
            output_filename = f"{base_name}.{ext}"
            output_file = os.path.join(output_path, output_filename)
            
            # Read file based on format
            if output_format.lower() == 'exr':
                # Use OpenImageIO for EXR (works great in Colab, simpler than OpenEXR)
                logger.info(f"   [READ] Reading EXR with OpenImageIO: {filename}...")
                
                try:
                    import OpenImageIO as oiio
                    
                    # Read EXR using OpenImageIO (handles all formats automatically)
                    img_buf = oiio.ImageBuf(str(file_path))
                    if img_buf.has_error:
                        raise ValueError(f"Failed to read EXR: {img_buf.geterror()}")
                    
                    # Get image specs
                    spec = img_buf.spec()
                    width = spec.width
                    height = spec.height
                    nchannels = spec.nchannels
                    
                    logger.info(f"   [READ] EXR read complete! Size: {width}x{height}, Channels: {nchannels}")
                    
                    # Resize using OpenImageIO's built-in resize (like sample_exr.py)
                    # This handles EXR properly with Lanczos3 filter
                    logger.info(f"   [RESIZE] Resizing with OpenImageIO (Lanczos3)...")
                    resized_buf = oiio.ImageBuf()
                    roi = oiio.ROI(0, target_size[0], 0, target_size[1])
                    oiio.ImageBufAlgo.resize(resized_buf, img_buf, filtername="lanczos3", roi=roi)
                    
                    if resized_buf.has_error:
                        raise ValueError(f"Failed to resize EXR: {resized_buf.geterror()}")
                    
                    # Write EXR using OpenImageIO (simple and reliable!)
                    logger.info(f"   [WRITE] Writing EXR with OpenImageIO: {output_filename}...")
                    resized_buf.write(str(output_file))
                    
                    if resized_buf.has_error:
                        raise ValueError(f"Failed to write EXR: {resized_buf.geterror()}")
                    
                    if not os.path.exists(output_file):
                        raise ValueError(f"EXR file was not created: {output_file}")
                    
                    logger.info(f"   ✅ [{file_index+1}/{len(input_files)}] {filename} → {output_filename} (OpenImageIO)")
                    
                except ImportError:
                    logger.info(f"   [READ] OpenImageIO not available, trying imageio...")
                    # Fallback to imageio
                    try:
                        import imageio
                        img = imageio.imread(file_path)
                        if img is None:
                            raise ValueError(f"Failed to read EXR file: {file_path}")
                        
                        # Handle channels
                        if len(img.shape) == 2:
                            img = np.stack([img, img, img], axis=2)
                        elif len(img.shape) == 3 and img.shape[2] == 4:
                            img = img[:, :, :3]
                        
                        # Normalize and resize
                        img_normalized = np.clip(img, 0, None)
                        img_max = img_normalized.max()
                        img_for_resize = img_normalized / img_max if img_max > 1.0 else img_normalized
                        
                        img_uint8 = (np.clip(img_for_resize, 0, 1) * 255).astype(np.uint8)
                        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
                        resized_bgr = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_LANCZOS4)
                        resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
                        resized_float = resized_rgb.astype(np.float32) / 255.0
                        
                        if img_max > 1.0:
                            resized_float = resized_float * img_max
                        
                        imageio.imwrite(output_file, resized_float, format='EXR-FI')
                        logger.info(f"   ✅ [{file_index+1}/{len(input_files)}] {filename} → {output_filename} (imageio)")
                    except ImportError:
                        logger.info(f"   [READ] imageio not available, using OpenEXR Python bindings...")
                        # Last fallback: OpenEXR Python bindings
                        from utils.exr_utils import read_exr_file, linear_to_srgb, srgb_to_linear, write_exr_file
                        
                        img_linear, metadata = read_exr_file(file_path)
                        img_srgb = linear_to_srgb(img_linear)
                        img_normalized = np.clip(img_srgb, 0, 1)
                        
                        img_uint8 = (img_normalized * 255).astype(np.uint8)
                        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
                        resized_bgr = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_LANCZOS4)
                        resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
                        resized_float = resized_rgb.astype(np.float32) / 255.0
                        resized_linear = srgb_to_linear(resized_float)
                        
                        metadata = {'width': target_size[0], 'height': target_size[1]}
                        write_exr_file(output_file, resized_linear, metadata)
                        logger.info(f"   ✅ [{file_index+1}/{len(input_files)}] {filename} → {output_filename} (OpenEXR)")
                    
                except Exception as e:
                    logger.error(f"   ❌ Error processing EXR {filename}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    raise
                
            else:
                # Read with PIL for other formats
                img = Image.open(file_path)
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    if img.mode == 'RGBA':
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                        img = background
                    else:
                        img = img.convert('RGB')
                
                # Convert to numpy for OpenCV
                img_np = np.array(img)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                # Resize using OpenCV (Lanczos4 - same as DVP/Cell3)
                resized_bgr = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_LANCZOS4)
                resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
                resized_pil = Image.fromarray(resized_rgb)
                
                # Save based on format
                if output_format.lower() in ['jpg', 'jpeg']:
                    resized_pil.save(output_file, 'JPEG', quality=95)
                elif output_format.lower() in ['tiff', 'tif']:
                    resized_pil.save(output_file, 'TIFF', compression='lzw')
                else:  # PNG
                    resized_pil.save(output_file, 'PNG')
                
                logger.info(f"   ✅ [{file_index+1}/{len(input_files)}] {filename} → {output_filename}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Error processing [{file_index+1}/{len(input_files)}] {os.path.basename(file_path)}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    # Process files - SEQUENTIAL for EXR (OpenEXR not thread-safe on Windows)
    # For other formats, can use parallel processing
    if output_format.lower() == 'exr':
        # EXR: Process sequentially (OpenEXR not thread-safe on Windows)
        logger.info(f"Processing {len(input_files)} EXR files SEQUENTIALLY (OpenEXR not thread-safe)...")
        completed = 0
        failed = 0
        
        for i, file_path in enumerate(input_files):
            result = process_single_file(file_path, i)
            if result:
                completed += 1
            else:
                failed += 1
            
            if (i + 1) % 5 == 0 or (i + 1) == len(input_files):
                logger.info(f"   Progress: {completed} succeeded, {failed} failed / {len(input_files)} total")
    else:
        # Other formats: Can use parallel processing
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import multiprocessing
        
        num_workers = min(multiprocessing.cpu_count(), 8, len(input_files))
        logger.info(f"Processing {len(input_files)} files in parallel using {num_workers} workers...")
        
        completed = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_file, file_path, i): (i, file_path) 
                      for i, file_path in enumerate(input_files)}
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    completed += 1
                else:
                    failed += 1
                
                if (completed + failed) % 5 == 0 or (completed + failed) == len(input_files):
                    logger.info(f"   Progress: {completed} succeeded, {failed} failed / {len(input_files)} total")
    
    logger.info(f"   ✅ All {len(input_files)} files processed")
    logger.info(f"Resized sequence saved to: {output_path}")
    
    return output_path, output_format


def get_sequence_resolution(sequence_path: str) -> Optional[Tuple[int, int]]:
    """
    Get resolution of first frame in sequence (FAST - reads only one file header)
    
    Args:
        sequence_path: Path to sequence
        
    Returns:
        (width, height) or None if cannot read
    """
    try:
        # FAST: Find first file and read only its header/metadata, not all frames
        if os.path.isfile(sequence_path):
            first_file = sequence_path
        elif os.path.isdir(sequence_path):
            # Find first image file
            for pattern in ['*.exr', '*.EXR', '*.tiff', '*.TIFF', '*.tif', '*.TIF', 
                           '*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
                files = sorted(glob.glob(os.path.join(sequence_path, pattern)))
                if files:
                    first_file = files[0]
                    break
            else:
                return None
        else:
            return None
        
        # Detect format and read only header/metadata
        format_type = detect_image_format(first_file)
        
        if format_type == 'exr':
            # EXR: Read only header (very fast) - don't read full file!
            try:
                # Try OpenImageIO first (works great in Colab)
                import OpenImageIO as oiio
                img_buf = oiio.ImageBuf(str(first_file))
                if not img_buf.has_error:
                    spec = img_buf.spec()
                    width = spec.width
                    height = spec.height
                    logger.debug(f"Read EXR header with OpenImageIO: {width}x{height} from {first_file}")
                    return (width, height)
                else:
                    raise ValueError(f"OpenImageIO error: {img_buf.geterror()}")
            except ImportError:
                # Fallback to OpenEXR Python bindings
                try:
                    from utils.exr_utils import OPENEXR_AVAILABLE
                    if not OPENEXR_AVAILABLE:
                        raise ImportError("OpenEXR not available")
                    
                    import OpenEXR
                    # Only read header, not full file - this is FAST!
                    exr_file = OpenEXR.InputFile(first_file)
                    header = exr_file.header()
                    dw = header['dataWindow']
                    width = dw.max.x - dw.min.x + 1
                    height = dw.max.y - dw.min.y + 1
                    exr_file.close()
                    logger.debug(f"Read EXR header: {width}x{height} from {first_file}")
                    return (width, height)
                except Exception as e:
                    logger.warning(f"Could not read EXR header from {first_file}: {e}")
            except Exception as e:
                logger.warning(f"Could not read EXR header from {first_file}: {e}")
                # Fallback: try PIL (may not work for EXR, but worth trying)
                try:
                    img = Image.open(first_file)
                    return img.size
                except Exception:
                    return None
        else:
            # Other formats: Use PIL (reads only header, not full image)
            img = Image.open(first_file)
            return img.size
    except Exception as e:
        logger.warning(f"Could not get resolution from {sequence_path}: {e}")
    
    return None

