# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**UnifiedCV_HRM** - A Hierarchical Reasoning Model (HRM) adapted for computer vision tasks (image/video classification, detection, segmentation). The core architecture is a Hierarchical Vision Transformer with hierarchical reasoning layers (H-layer and L-layer) that perform iterative reasoning steps with dynamic halting.

### Key Characteristics
- **Architecture**: Hierarchical Vision Transformer with HRM layers (H-layer for high-level reasoning, L-layer for low-level feature processing)
- **Tasks**: Image/video classification, object detection, semantic segmentation (multi-task)
- **Datasets**: COCO (multi-task), WLASL (video sign language recognition), UCF-Crime, HMDB51
- **Framework**: PyTorch + Hugging Face Transformers integration (PreTrainedModel)
- **Training**: Hugging Face Hub integration for checkpoint management, COCO multi-task and WLASL video training

## Key Files

| File | Description |
|------|-------------|
| `model.py` | Core architecture: HRM layers, HierarchicalVisionTransformer, UnifiedCVHRM (HF-compatible) |
| `dataset.py` | COCO multi-task dataset (classification/detection/segmentation) |
| `utils.py` | Checkpoint utilities (HF Hub save/load, safetensors, config) |
| `train.py` | Main training script (COCO multi-task + WLASL video classification) |
| `utils.py` (WLASL) | WLASL video dataset loader, training loop |
| `WLASL_train.py` | WLASL video training script (multiple class sizes, HF Hub) |

## Common Commands

### Training (WLASL Video Classification)
```bash
# Train WLASL video classification (pushes to HF Hub)
python WLASL_train.py

# Or via train.py (COCO multi-task + WLASL)
python train.py
```

### Model Loading (via Hugging Face)
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("codewithdark/UnifiedCVHRM-WLASL-cls-1000-v")
```

### Testing / Inference
```python
from model import UnifiedCVHRM, UnifiedCVHRMConfig
import torch

config = UnifiedCVHRMConfig(
    img_size=64, patch_size=8, hidden_size=256, 
    max_steps=3, input_type='video', num_classes=1000
)
model = UnifiedCVHRM(config)
# Forward pass
features, steps = model.backbone(video_tensor)  # video: (B, C, T, H, W)
logits = model.classification_head(features)
```

### Checkpoint Management
- Checkpoints pushed to HF Hub: `codewithdark/UnifiedCVHRM-WLASL-cls-{N}-v`
- Local checkpoints: `./checkpoints/`
- Uses `safetensors` + `config.json` + `checkpoint_meta_epoch_*.pt`
- Resume training via `load_from_hf()` in utils.py

## Architecture Overview

### Core Components (model.py)
1. **Tokenizers**: 
   - `ImagePatchEmbed` - 2D Conv patch embedding (images)
   - `VideoTokenizer` - 3D CNN tokenizer (video)

2. **HierarchicalVisionTransformer** (backbone):
   - Tokenizer → positional encoding → cls token
   - **HRMLayer** (×2): H-layer + L-layer with cross-attention
   - Hierarchical reasoning loop with dynamic halting (`max_steps`, halt predictor)
   - Output: features + reasoning steps count

3. **Task Heads** (attached to UnifiedCVHRM):
   - `ClassificationHead` - cls token → classes
   - `SegmentationHead` - spatial upsampling (Conv2d/Conv3d)
   - `DetectionHead` - cls token → cls + bbox

4. **UnifiedCVHRM** - HF `PreTrainedModel` wrapper with `UnifiedCVHRMConfig`

### Key Architectural Details
- **HRMLayer**: 2 TransformerBlocks + CrossAttention for text conditioning (optional)
- **TransformerBlock**: Post-norm, RMSNorm, SwiGLU FFN, MultiHeadAttention (non-causal for vision)
- **Hierarchical Reasoning**: 
  - `z_L = L_layer(z_L, z_H + seq)` (low-level conditioned on high + input)
  - `z_H = H_layer(z_H, z_L)` (high-level updated from low)
  - Dynamic halting via sigmoid predictor on `z_H` mean
- **Tokenizer output shapes**: Image `(1, H/p, W/p)`, Video `(T/2, H/4, W/4)`

## Training Configuration (train.py / WLASL_train.py)

### WLASL Video Training
- **Datasets**: 300, 500, 1000, 2000 classes (top frequent)
- **Video**: 8 frames, 64×64, 3 channels
- **Model**: hidden=256, heads=8, max_steps=3
- **Training**: 100 epochs, batch=8, lr=1e-4, AdamW(wd=0.01)
- **HF Hub**: Separate repos per class size
- **Dataset path**: `E:\Dataset\WSLASL` (configure in train.py)

### COCO Multi-Task (train.py)
- **Tasks**: Classification (multi-label), Detection (single bbox), Segmentation (semantic)
- **Image size**: 64×64, patch=8
- **Losses**: BCEWithLogitsLoss (cls), CE (seg), CE+SmoothL1 (det)
- **Note**: COCO path needs configuration (`./coco`)

## Dataset Notes

### WLASL (WLASL_train.py / dataset.py)
- Expects `WLASL_v0.3.json` + `videos/` folder at `data_path`
- Samples: `[gloss, video_id, bbox]` filtered by split
- Sampling: uniform frame sampling (configurable `max_frames`)
- Augmentation: random brightness/contrast

### COCO (dataset.py)
- `COCOMultiTaskDataset`: multi-task from instances
- Classification: multi-hot from categories
- Detection: first annotation bbox (normalized cx,cy,w,h)
- Segmentation: primary category mask (resized to img_size)

## Development Notes

### Important Configurations (model.py)
```python
UnifiedCVHRMConfig(
    img_size=64,           # input resolution
    patch_size=8,          # patch size (img_size must be divisible)
    hidden_size=256,       # must be divisible by num_heads
    num_heads=8,
    expansion=2.0,         # SwiGLU expansion
    max_steps=3,           # max reasoning steps
    input_type='image'|'video',
    num_classes=80         # task-specific head size
)
```

### Model Loading (utils.py)
- `load_from_hf(repo_id, num_classes, task_name)` - loads via `AutoModel.from_pretrained`
- `load_model_checkpoint(model, path_or_repo, from_hf=True)` - loads safetensors/.pt + config
- `save_and_push_checkpoint(...)` - saves config + safetensors + meta, pushes to HF

### Training Loop (train.py / WLASL_train.py)
- `get_dataloader()` - returns DataLoader + loss criterions
- `train_for_task()` - single/multi-task training loop with HF Hub push
- Per-epoch eval on val set (accuracy, precision, recall, F1)
- Gradient clipping (1.0), gradient accumulation not used

### GPU Setup
```python
device = set_device()  # Auto-detects CUDA, enables TF32
```

## Common Development Tasks

### Add New Task Head
1. Add new head class in `model.py` (e.g., `KeypointHead`)
2. Add to `UnifiedCVHRM.__init__` and `forward()`
3. Extend `UnifiedCVHRMConfig` if needed
4. Add loss in `train.py` + dataloader support in `dataset.py`

### Modify HRM Architecture
- Core logic in `HRMLayer.forward()` (hierarchical loop)
- Modify `max_steps`, halt threshold (0.5), or add layers
- Change `TransformerBlock` for pre-norm, rotary, etc.

### Change Video Tokenizer
- Modify `VideoTokenizer` conv3d stack
- Adjust `output_shape` property for downstream heads

### Add New Dataset
- Implement `Dataset` class in `dataset.py`
- Add to `get_dataloader()` in `utils.py` / `train.py`
- Handle collate_fn for variable-length targets (detection/seg)

## HF Hub Repositories
- Pattern: `codewithdark/UnifiedCVHRM-{dataset}-{task}-{classes}-{v/i}`
- Example: `codewithdark/UnifiedCVHRM-WLASL-cls-1000-v`
- Files: `config.json`, `model_*.safetensors`, `checkpoint_meta_epoch_*.pt`

## Important Paths (Update for Your Environment)
- WLASL data: `E:\Dataset\WSLASL` (in train.py:396, utils.py:137, 233)
- COCO data: `./coco` (in dataset.py:20, train.py:387)
- Local checkpoints: `./checkpoints/`

## Known Limitations / TODOs
- Multi-task training for WLASL not implemented (detection/seg need bbox/mask)
- COCO training uses synthetic fallback if data missing
- Video tokenizer fixed to 8 frames, 64×64
- Detection head: single bbox from cls token (simplified)
- Segmentation: bilinear/trilinear upsampling only
- Windows: `num_workers=0` in DataLoaders (pickling issues)