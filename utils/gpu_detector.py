"""
GPU Detection and Optimization Utilities
Detects available GPU memory and optimizes processing parameters
"""
import torch
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def detect_gpu_info() -> Dict:
    """
    Detect GPU information and available memory
    
    Returns:
        Dictionary with GPU info:
        {
            'available': bool,
            'device_name': str,
            'total_memory_gb': float,
            'free_memory_gb': float,
            'device': torch.device
        }
    """
    info = {
        'available': False,
        'device_name': 'CPU',
        'total_memory_gb': 0.0,
        'free_memory_gb': 0.0,
        'device': torch.device('cpu')
    }
    
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, will use CPU (very slow)")
        return info
    
    try:
        device = torch.device('cuda')
        info['device'] = device
        info['available'] = True
        
        # Get GPU name
        gpu_name = torch.cuda.get_device_name(0)
        info['device_name'] = gpu_name
        
        # Get memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory
        total_memory_gb = total_memory / (1024 ** 3)
        info['total_memory_gb'] = total_memory_gb
        
        # Get free memory (approximate)
        torch.cuda.empty_cache()
        free_memory = torch.cuda.memory_reserved(0)
        free_memory_gb = (total_memory - free_memory) / (1024 ** 3)
        info['free_memory_gb'] = free_memory_gb
        
        logger.info(f"GPU detected: {gpu_name}")
        logger.info(f"Total memory: {total_memory_gb:.2f} GB")
        logger.info(f"Free memory: {free_memory_gb:.2f} GB")
        
    except Exception as e:
        logger.error(f"Error detecting GPU: {e}")
        info['available'] = False
    
    return info


def optimize_max_size_for_gpu(gpu_info: Dict, input_resolution: Optional[Tuple[int, int]] = None) -> int:
    """
    Optimize max_img_size based on available GPU memory
    
    Args:
        gpu_info: GPU information dictionary from detect_gpu_info()
        input_resolution: Optional (width, height) of input for better estimation
        
    Returns:
        Recommended max_img_size (longest edge)
    """
    if not gpu_info['available']:
        logger.warning("No GPU available, using minimum size")
        return 640
    
    total_memory_gb = gpu_info['total_memory_gb']
    
    # Memory requirements based on resolution (from DiffuEraser docs):
    # 640x360: 12GB
    # 960x540: 20GB
    # 1280x720: 33GB
    # 1920x1080: ~50GB+
    # 2560x1440: ~80GB+ (for A100 80GB)
    
    # Conservative estimates (leave some headroom)
    if total_memory_gb >= 70:  # A100 80GB
        max_size = 2560  # Maximum for A100 80GB
    elif total_memory_gb >= 50:  # A100 40GB or similar
        max_size = 1920  # High-end GPUs
    elif total_memory_gb >= 33:  # V100 32GB
        max_size = 1280
    elif total_memory_gb >= 24:  # V100 24GB
        max_size = 960
    elif total_memory_gb >= 16:  # T4 16GB, V100 16GB
        max_size = 960
    elif total_memory_gb >= 12:  # T4 12GB
        max_size = 640
    else:
        max_size = 512  # Very limited
    
    # If input resolution is provided, don't upscale unnecessarily
    if input_resolution:
        input_max = max(input_resolution[0], input_resolution[1])
        if input_max < max_size:
            # Use input size if it's smaller, but ensure it's multiple of 8
            max_size = ((input_max // 8) * 8)
            if max_size < 256:
                max_size = 256
    
    logger.info(f"Optimized max_img_size: {max_size} (for {total_memory_gb:.1f}GB GPU)")
    
    return max_size


def get_optimal_settings(gpu_info: Dict, input_resolution: Optional[Tuple[int, int]] = None) -> Dict:
    """
    Get optimal processing settings based on GPU
    
    Args:
        gpu_info: GPU information dictionary
        input_resolution: Optional input resolution
        
    Returns:
        Dictionary with optimal settings:
        {
            'max_img_size': int,
            'mask_dilation_iter': int,
            'use_fp16': bool,
            'batch_size': int (for future use)
        }
    """
    max_img_size = optimize_max_size_for_gpu(gpu_info, input_resolution)
    
    # Use FP16 for memory savings if GPU supports it
    use_fp16 = gpu_info['available'] and gpu_info['total_memory_gb'] < 24
    
    # Mask dilation - keep default
    mask_dilation_iter = 8
    
    settings = {
        'max_img_size': max_img_size,
        'mask_dilation_iter': mask_dilation_iter,
        'use_fp16': use_fp16,
        'batch_size': 1  # Currently always 1
    }
    
    logger.info(f"Optimal settings: max_size={max_img_size}, fp16={use_fp16}")
    
    return settings

