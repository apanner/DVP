"""
Resize Image Sequence - One Frame at a Time
Processes frames individually: read -> resize -> write -> next
Avoids loading all frames into memory (fixes Windows OpenEXR hanging issue)
"""
import os
import glob
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def resize_image_sequence_onebyone(input_path: str, output_path: str, 
                                   target_size: Tuple[int, int],
                                   format_type: Optional[str] = None) -> Tuple[str, str]:
    """
    Resize sequence ONE FRAME AT A TIME (read -> resize -> write -> next)
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
    
    # Detect format from first file (fast)
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
        from utils.image_utils import detect_image_format
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
        from utils.image_utils import detect_image_format
        detected_format = detect_image_format(input_path)
    
    # Use provided format or detected
    output_format = format_type or detected_format
    if not output_format or output_format == 'unknown':
        output_format = 'png'
    
    logger.info(f"Resizing sequence: {input_path} → {output_path}")
    logger.info(f"Format: {output_format.upper()}, Files: {len(input_files)}, Target: {target_size[0]}x{target_size[1]}")
    
    if not input_files:
        raise ValueError(f"No files found in: {input_path}")
    
    # Process ONE FRAME AT A TIME
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import multiprocessing
    
    num_workers = min(multiprocessing.cpu_count(), 8, len(input_files))
    logger.info(f"Processing {len(input_files)} files one-by-one using {num_workers} workers...")
    
    def process_single_file(file_path, file_index):
        """Process a single file: read -> resize -> write"""
        try:
            # Determine output filename
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            ext = output_format.lower()
            if ext == 'jpeg':
                ext = 'jpg'
            output_filename = f"{base_name}.{ext}"
            output_file = os.path.join(output_path, output_filename)
            
            # Read file based on format
            if output_format.lower() == 'exr':
                # Read EXR using OpenEXR
                from utils.exr_utils import read_exr_file, linear_to_srgb, srgb_to_linear, write_exr_file
                
                # Read EXR (linear color space)
                img_linear, metadata = read_exr_file(file_path)
                
                # Convert linear to sRGB for processing
                img_srgb = linear_to_srgb(img_linear)
                
                # Convert to uint8 [0, 255] for OpenCV
                img_uint8 = (np.clip(img_srgb, 0, 1) * 255).astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
                
                # Resize using OpenCV (Lanczos4 - same as DVP)
                resized_bgr = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_LANCZOS4)
                
                # Convert back to RGB
                resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
                
                # Convert back to float32 [0, 1] and then to linear for EXR
                img_resized_float = resized_rgb.astype(np.float32) / 255.0
                img_resized_linear = srgb_to_linear(img_resized_float)
                
                # Write EXR file
                write_exr_file(output_file, img_resized_linear, metadata)
                
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
                
                # Resize using OpenCV
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
            
            return True
        except Exception as e:
            logger.error(f"Error processing {os.path.basename(file_path)}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    # Process files in parallel (but each worker processes one file at a time)
    completed = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_file, file_path, i): (i, file_path) 
                  for i, file_path in enumerate(input_files)}
        
        for future in as_completed(futures):
            if future.result():
                completed += 1
                if completed % 10 == 0 or completed == len(input_files):
                    logger.info(f"   Progress: {completed}/{len(input_files)} files processed")
    
    logger.info(f"   ✅ All {len(input_files)} files processed")
    logger.info(f"Resized sequence saved to: {output_path}")
    
    return output_path, output_format

