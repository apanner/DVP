"""
Resize Processor
Handles input/mask resizing based on GPU optimization
Creates workspace structure with resized versions
"""
import os
from pathlib import Path
from typing import Tuple, Optional
import logging

from utils.image_utils import (
    read_image_sequence, 
    resize_image_sequence,
    get_sequence_resolution,
    write_image_sequence
)
from PIL import Image
from utils.gpu_detector import detect_gpu_info, optimize_max_size_for_gpu

logger = logging.getLogger(__name__)


class ResizeProcessor:
    """Processes and resizes input/mask sequences for GPU optimization"""
    
    def __init__(self, output_base_dir: str):
        """
        Initialize resize processor
        
        Args:
            output_base_dir: Base output directory (user selected)
        """
        self.output_base_dir = output_base_dir
        self.workspace_dir = os.path.join(output_base_dir, "iMagic_workspace")
        self.input_resize_dir = os.path.join(self.workspace_dir, "input_resize")
        self.mask_resize_dir = os.path.join(self.workspace_dir, "mask_resize")
        self.output_dir = os.path.join(self.workspace_dir, "output")
        
        # Detect GPU
        self.gpu_info = detect_gpu_info()
        
    def create_workspace(self):
        """Create workspace directory structure"""
        os.makedirs(self.input_resize_dir, exist_ok=True)
        os.makedirs(self.mask_resize_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Workspace created: {self.workspace_dir}")
    
    def process_sequence(self, input_path: str, mask_path: str, 
                        sequence_name: str,
                        max_img_size: Optional[int] = None) -> Tuple[str, str, Tuple[int, int]]:
        """
        Process and resize input/mask sequences
        
        Args:
            input_path: Path to input sequence
            mask_path: Path to mask sequence
            sequence_name: Name for this sequence
            max_img_size: Optional max size override
            
        Returns:
            Tuple of (resized_input_path, resized_mask_path, final_size)
        """
        # Create workspace
        self.create_workspace()
        
        # Get input resolution
        input_resolution = get_sequence_resolution(input_path)
        if not input_resolution:
            raise ValueError(f"Could not determine resolution from: {input_path}")
        
        logger.info(f"Input resolution: {input_resolution[0]}x{input_resolution[1]}")
        
        # Optimize max size based on GPU
        if max_img_size is None:
            max_img_size = optimize_max_size_for_gpu(self.gpu_info, input_resolution)
        
        # Calculate target size (maintain aspect ratio)
        input_width, input_height = input_resolution
        max_dimension = max(input_width, input_height)
        
        if max_dimension <= max_img_size:
            # No resize needed, but still copy to workspace
            target_size = (input_width, input_height)
            logger.info(f"No resize needed (input {max_dimension} <= max {max_img_size})")
        else:
            # Calculate resize maintaining aspect ratio
            scale = max_img_size / max_dimension
            target_width = int(input_width * scale)
            target_height = int(input_height * scale)
            
            # Ensure dimensions are multiples of 8 (required by models)
            target_width = (target_width // 8) * 8
            target_height = (target_height // 8) * 8
            
            target_size = (target_width, target_height)
            logger.info(f"Resizing to: {target_size[0]}x{target_size[1]} (scale: {scale:.3f})")
        
        # Create sequence-specific directories
        seq_input_dir = os.path.join(self.input_resize_dir, sequence_name)
        seq_mask_dir = os.path.join(self.mask_resize_dir, sequence_name)
        
        # Smart auto-detection: detect formats automatically (no manual specification needed!)
        try:
            _, _, _, input_format = read_image_sequence(input_path, is_mask=False)
            logger.info(f"✅ Auto-detected input format: {input_format.upper()}")
        except Exception as e:
            logger.warning(f"Could not auto-detect input format: {e}, trying fallback...")
            # Fallback: try directory scan
            from utils.image_utils import auto_detect_format_from_directory
            input_format = auto_detect_format_from_directory(input_path) or 'exr'
            logger.info(f"Fallback input format: {input_format.upper()}")
        
        # Detect mask format separately (can be different from input!)
        try:
            _, _, _, mask_format = read_image_sequence(mask_path, is_mask=True)
            logger.info(f"✅ Auto-detected mask format: {mask_format.upper()}")
        except Exception as e:
            logger.warning(f"Could not auto-detect mask format: {e}, trying fallback...")
            # Fallback: try directory scan
            from utils.image_utils import auto_detect_format_from_directory
            mask_format = auto_detect_format_from_directory(mask_path) or input_format
            logger.info(f"Fallback mask format: {mask_format.upper()}")
        
        logger.info(f"📋 Formats: Input={input_format.upper()}, Mask={mask_format.upper()}")
        
        # Resize input sequence (preserves input format)
        logger.info(f"Resizing input sequence: {input_path} (format: {input_format.upper()})")
        resize_image_sequence(
            input_path,
            seq_input_dir,
            target_size,
            format_type=input_format
        )
        
        # Resize mask sequence (preserves mask format - can be different!)
        logger.info(f"Resizing mask sequence: {mask_path} (format: {mask_format.upper()})")
        resize_image_sequence(
            mask_path,
            seq_mask_dir,
            target_size,
            format_type=mask_format  # Use mask's own format, not input format!
        )
        
        logger.info(f"✅ Resized sequences saved:")
        logger.info(f"   Input: {seq_input_dir}")
        logger.info(f"   Mask: {seq_mask_dir}")
        
        return seq_input_dir, seq_mask_dir, target_size
    
    def get_output_path(self, sequence_name: str) -> str:
        """Get output path for processed sequence"""
        return os.path.join(self.output_dir, sequence_name)

