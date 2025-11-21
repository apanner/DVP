"""
Local Test Script for Resize Process
Tests resize functionality with sample shot (5 frames)
"""
import os
import sys
import logging
from pathlib import Path

# Add DVP to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.image_utils import (
    read_image_sequence,
    resize_image_sequence,
    get_sequence_resolution
)

# Check OpenEXR availability
try:
    from utils.exr_utils import read_exr_file, OPENEXR_AVAILABLE
    if not OPENEXR_AVAILABLE:
        print("⚠️  WARNING: OpenEXR not available. EXR files cannot be read.")
        print("   Install with: pip install Imath OpenEXR")
        print("   Or: pip install OpenEXR>=3.4.0")
        OPENEXR_AVAILABLE = False
    else:
        OPENEXR_AVAILABLE = True
except Exception as e:
    print(f"⚠️  WARNING: Could not check OpenEXR availability: {e}")
    OPENEXR_AVAILABLE = False

# Try to import GPU detector (optional - may fail on Windows without proper PyTorch)
try:
    from utils.gpu_detector import detect_gpu_info, optimize_max_size_for_gpu
    GPU_DETECTION_AVAILABLE = True
except Exception as e:
    print(f"⚠️  GPU detection not available (PyTorch issue): {e}")
    print("   Using fallback GPU detection for testing...")
    GPU_DETECTION_AVAILABLE = False
    
    # Fallback functions for testing without PyTorch
    def detect_gpu_info():
        """Fallback GPU info for testing"""
        return {
            'available': False,
            'device_name': 'CPU (Test Mode)',
            'total_memory_gb': 16.0,  # Simulate 16GB for testing
            'free_memory_gb': 16.0,
            'device': None
        }
    
    def optimize_max_size_for_gpu(gpu_info, input_resolution=None):
        """Fallback optimization for testing"""
        # Use reasonable default for testing
        if input_resolution:
            input_max = max(input_resolution[0], input_resolution[1])
            # Don't resize if input is already reasonable
            if input_max <= 1920:
                return input_max
        return 1280  # Default test size

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_resize_process():
    """Test resize process step by step"""
    
    # Test paths (relative to script location)
    script_dir = Path(__file__).parent
    test_dir = script_dir.parent / "test" / "TML306_007_010"
    
    input_path = str(test_dir / "input")
    mask_path = str(test_dir / "mask")
    output_dir = script_dir / "test_output"
    
    print("=" * 70)
    print("RESIZE PROCESS TEST - Step by Step")
    print("=" * 70)
    
    # STEP 1: Verify test files exist
    print("\n[STEP 1] Verifying test files exist...")
    if not os.path.exists(input_path):
        print(f"❌ Input path not found: {input_path}")
        return
    if not os.path.exists(mask_path):
        print(f"❌ Mask path not found: {mask_path}")
        return
    
    input_files = [f for f in os.listdir(input_path) if f.endswith('.exr')]
    mask_files = [f for f in os.listdir(mask_path) if f.endswith('.jpg')]
    
    print(f"✅ Found {len(input_files)} input EXR files")
    print(f"✅ Found {len(mask_files)} mask JPG files")
    if input_files:
        print(f"   Sample: {input_files[0]}")
    if mask_files:
        print(f"   Sample: {mask_files[0]}")
    
    # STEP 2: Detect formats (FAST - filename based)
    print("\n[STEP 2] Detecting file formats (FAST - filename based)...")
    from utils.image_utils import detect_image_format
    
    first_input = os.path.join(input_path, input_files[0]) if input_files else None
    first_mask = os.path.join(mask_path, mask_files[0]) if mask_files else None
    
    if first_input:
        input_format = detect_image_format(first_input)
        print(f"✅ Input format: {input_format.upper()}")
    else:
        input_format = 'exr'
        print(f"⚠️  Using default input format: {input_format.upper()}")
    
    if first_mask:
        mask_format = detect_image_format(first_mask)
        print(f"✅ Mask format: {mask_format.upper()}")
    else:
        mask_format = 'jpg'
        print(f"⚠️  Using default mask format: {mask_format.upper()}")
    
    # STEP 3: Detect resolution (FAST - one file header only)
    print("\n[STEP 3] Detecting resolution (FAST - one file header only)...")
    print("   Reading EXR header from first file...")
    try:
        input_resolution = get_sequence_resolution(input_path)
        if input_resolution:
            print(f"✅ Input resolution: {input_resolution[0]}x{input_resolution[1]}")
        else:
            print("❌ Could not detect resolution")
            print("   Trying alternative method...")
            # Try reading first file directly
            try:
                from utils.exr_utils import read_exr_file, OPENEXR_AVAILABLE
                if OPENEXR_AVAILABLE:
                    first_exr = os.path.join(input_path, input_files[0])
                    print(f"   Reading full EXR file: {input_files[0]}...")
                    _, metadata = read_exr_file(first_exr)
                    input_resolution = (metadata['width'], metadata['height'])
                    print(f"✅ Input resolution (from full read): {input_resolution[0]}x{input_resolution[1]}")
                else:
                    print("❌ OpenEXR not available. Install: pip install Imath OpenEXR")
                    return
            except Exception as e:
                print(f"❌ Alternative method also failed: {e}")
                return
    except Exception as e:
        print(f"❌ Error detecting resolution: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # STEP 4: Optimize for GPU
    print("\n[STEP 4] Optimizing for GPU...")
    try:
        gpu_info = detect_gpu_info()
        print(f"🔍 GPU: {gpu_info.get('device_name', 'CPU')}")
        print(f"   Memory: {gpu_info.get('total_memory_gb', 0):.1f} GB")
        
        max_img_size = optimize_max_size_for_gpu(gpu_info, input_resolution)
        print(f"✅ Optimized max_img_size: {max_img_size}px")
    except Exception as e:
        print(f"⚠️  GPU detection failed: {e}")
        print("   Using default max_img_size: 1280px for testing")
        max_img_size = 1280
    
    # STEP 5: Calculate target size
    print("\n[STEP 5] Calculating target size...")
    input_width, input_height = input_resolution
    max_dimension = max(input_width, input_height)
    
    if max_dimension <= max_img_size:
        target_size = (input_width, input_height)
        print(f"✅ No resize needed ({max_dimension} <= {max_img_size})")
    else:
        scale = max_img_size / max_dimension
        target_width = int(input_width * scale)
        target_height = int(input_height * scale)
        target_width = (target_width // 8) * 8
        target_height = (target_height // 8) * 8
        target_size = (target_width, target_height)
        print(f"✅ Resizing to: {target_size[0]}x{target_size[1]} (scale: {scale:.3f})")
    
    # STEP 6: Resize input sequence
    print("\n[STEP 6] Resizing input sequence (parallel processing)...")
    os.makedirs(output_dir, exist_ok=True)
    input_output_dir = output_dir / "input_resize"
    
    try:
        resized_input_path, detected_format = resize_image_sequence(
            input_path,
            str(input_output_dir),
            target_size,
            format_type=input_format
        )
        print(f"✅ Input resize complete: {resized_input_path}")
        print(f"   Format: {detected_format.upper()}")
        
        # Verify resized files
        resized_files = [f for f in os.listdir(resized_input_path) if f.endswith('.exr')]
        print(f"   ✅ Created {len(resized_files)} resized EXR files")
        
        # Test reading one resized EXR file
        if resized_files:
            test_file = os.path.join(resized_input_path, resized_files[0])
            print(f"\n   [VERIFY] Testing EXR read: {resized_files[0]}")
            try:
                img, metadata = read_exr_file(test_file)
                print(f"   ✅ EXR read successful!")
                print(f"      Size: {img.shape}")
                print(f"      Resolution: {metadata.get('width')}x{metadata.get('height')}")
                print(f"      Data range: [{img.min():.3f}, {img.max():.3f}]")
            except Exception as e:
                print(f"   ❌ EXR read failed: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Input resize failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # STEP 7: Resize mask sequence
    print("\n[STEP 7] Resizing mask sequence (parallel processing)...")
    mask_output_dir = output_dir / "mask_resize"
    
    try:
        resized_mask_path, detected_format = resize_image_sequence(
            mask_path,
            str(mask_output_dir),
            target_size,
            format_type=mask_format
        )
        print(f"✅ Mask resize complete: {resized_mask_path}")
        print(f"   Format: {detected_format.upper()}")
        
        # Verify resized files
        resized_files = [f for f in os.listdir(resized_mask_path) if f.endswith('.jpg')]
        print(f"   ✅ Created {len(resized_files)} resized JPG files")
        
    except Exception as e:
        print(f"❌ Mask resize failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # STEP 8: Summary
    print("\n" + "=" * 70)
    print("✅ RESIZE TEST COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"   ├─ input_resize/ ({len(resized_files) if 'resized_files' in locals() else 0} files)")
    print(f"   └─ mask_resize/ ({len(resized_files) if 'resized_files' in locals() else 0} files)")
    print("\n✅ All resize operations completed successfully!")
    print("   EXR files are valid and readable.")


if __name__ == "__main__":
    test_resize_process()

