# Repository Configuration

## Customized DVP Repository

This project uses the **customized DVP repository** for all Colab processing:

**Repository URL**: `https://github.com/apanner/DVP.git`

**Repository Name**: `DVP`

**Clone Location in Colab**: `/content/DVP`

---

## What's Different from Original?

The customized DVP repository (`apanner/DVP`) includes:

1. **Image Sequence Support**:
   - Direct EXR/TIFF/PNG/JPG input/output (no MP4 conversion)
   - `utils/image_utils.py` - Smart format detection
   - `utils/exr_utils.py` - EXR color space handling

2. **Modified Core Components**:
   - `diffueraser/diffueraser.py` - Accepts image sequences directly
   - `propainter/inference.py` - Outputs image sequences directly
   - Preserves input format throughout pipeline

3. **VFX Production Workflow**:
   - Full-res input/mask support
   - GPU-optimized resizing
   - Format preservation (EXR → EXR, etc.)

---

## Configuration Files

### `google_desk_app/colab_templates/repo_config.py`

This file is included in the Colab cellcode and specifies the repository:

```python
REPO_URL = "https://github.com/apanner/DVP.git"
REPO_NAME = "DVP"
```

### Fallback Values

If `repo_config.py` is not found, the code uses these fallback values:
- `REPO_URL = "https://github.com/apanner/DVP.git"`
- `REPO_NAME = "DVP"`

---

## Colab Workflow

1. **Cell 1**: Clones `https://github.com/apanner/DVP.git` to `/content/DVP`
2. **Cell 2**: Imports from `/content/DVP/diffueraser/` and `/content/DVP/propainter/`
3. **Cell 3**: Uses `/content/DVP/utils/` for image sequence processing

---

## Verification

After cloning in Colab, verify these paths exist:

- ✅ `/content/DVP/diffueraser/diffueraser.py`
- ✅ `/content/DVP/propainter/inference.py`
- ✅ `/content/DVP/utils/image_utils.py`
- ✅ `/content/DVP/utils/exr_utils.py`

---

## Important Notes

- **Do NOT use the original repository** (`lixiaowen-xw/DiffuEraser`) for processing
- The original repository only supports MP4 video input/output
- This customized repository supports image sequences (EXR, TIFF, PNG, JPG)
- All Colab templates are configured to use `apanner/DVP`

---

## Model Weights

Model weights are still downloaded from HuggingFace:
- `lixiaowen/diffuEraser` - DiffuEraser weights
- `stable-diffusion-v1-5` - Base model
- `PCM_Weights` - Prior model weights
- `propainter` - ProPainter weights

These are **model weights only**, not code. The code comes from `apanner/DVP`.

