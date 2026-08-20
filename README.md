# I-HRM: Image Hierarchical Reasoning Model

[![PyPI](https://img.shields.io/pypi/v/ihrm.svg)](https://pypi.org/project/ihrm/)
[![Python](https://img.shields.io/pypi/pyversions/ihrm.svg)](https://pypi.org/project/ihrm/)
[![License](https://img.shields.io/pypi/l/ihrm.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-I--HRM-blue.svg)](https://arxiv.org/abs/xxxx.xxxxx)

Official PyTorch implementation of **I-HRM: Image Hierarchical Reasoning Model** — a Hierarchical Vision Transformer with dual-stream recurrent reasoning, adaptive halting, and dynamic sparse attention.

## Features

- **Hierarchical Reasoning**: Dual-stream architecture (H-layer for semantics, L-layer for perception) with cross-attention
- **Adaptive Halting** (Algorithm 1): Dynamic reasoning steps (K ≤ 6) with learned halt predictor
- **Attention Variants**: MHA, DSA (w=7), Delta Attention (L2-normalized)
- **FFN Variants**: SwiGLU, Dense-Routed MoE (8 experts), DeltaNet
- **Multi-Task**: Classification, Detection, Segmentation, Video Recognition
- **13 Paper Benchmarks**: CIFAR-10/100, SVHN, MNIST, Fashion-MNIST, STL-10, TinyImageNet-200, ImageNet-1K, ChestX-Ray, Brain Tumor MRI, HAM10000, Intel Scenes, COCO, WLASL
- **Hugging Face Integration**: `PreTrainedModel` compatible with `.from_pretrained()` / `.push_to_hub()`
- **Flexible Data Sources**: Local folders, Hugging Face Hub, TorchVision built-ins
- **Medical Imaging**: CLAHE preprocessing built-in

## Installation

```bash
# From source (recommended)
git clone https://github.com/codewithdark/I-HRM
cd I-HRM
pip install -e ".[dev]"      # With dev tools
pip install -e ".[dev,lion]"  # + Lion optimizer
pip install -e ".[dev,bitsandbytes]"  # + 8-bit quantization

# Core dependencies only
pip install -e .
```

**Requirements**: Python ≥ 3.10, PyTorch ≥ 2.2, CUDA 11.8+ (for GPU)

## Quick Start

### Train on CIFAR-100 (Paper Configuration)

```bash
# Using CLI with paper base preset (D=384, DSA+MoE, 12M params)
ihrm train --preset paper_base --dataset cifar100 --epochs 300 --batch-size 128 --use-wandb

# Quick debug run
ihrm train --preset standard --dataset cifar10 --epochs 5 --batch-size 64 --use-tensorboard false
```

### Train with Custom Config

```bash
# Generate config files first
ihrm config --output configs/my_experiment --preset dsa_moe

# Edit configs/my_experiment/*.yaml as needed
# Then train
ihrm train --config configs/my_experiment/config.yaml
```

### Evaluate Model

```bash
# Local checkpoint
ihrm evaluate --model outputs/best_model.pt --dataset cifar100 --task classification

# Hugging Face Hub model
ihrm evaluate --model-hf codewithdark/I-HRM-cifar100 --dataset cifar100
```

### Run Paper Benchmarks (Table 3)

```bash
# Benchmark on all 13 datasets
ihrm benchmark --model outputs/best_model.pt --datasets cifar10 cifar100 svhn mnist fashion_mnist stl10 tinyimagenet200 imagenet1k chestxray_pneumonia brain_tumor_mri ham10000 intel_scenes coco_multitask

# Quick benchmark (fewer batches)
ihrm benchmark --model outputs/best_model.pt --datasets cifar10 cifar100 --data-source torchvision
```

### Inference on Single Image

```bash
ihrm predict --model outputs/best_model.pt --image test.jpg --task classification --top-k 5
```

### Download Datasets

```bash
# Single dataset
ihrm download --dataset cifar100 --output ./data

# All paper datasets
ihrm download --dataset all --output ./data
```

### Export to Hugging Face Format

```bash
# Export local checkpoint to HF model
ihrm export --model outputs/best_model.pt --repo codewithdark/I-HRM-cifar100 --push

# Export to ONNX
ihrm export-onnx --model outputs/best_model.pt --output model.onnx --opset 17
```

## Command Reference

| Command | Description |
|---------|-------------|
| `ihrm train` | Train model (local/HF data) |
| `ihrm evaluate` | Evaluate on dataset |
| `ihrm benchmark` | Run multi-dataset benchmarks |
| `ihrm predict` | Single image inference |
| `ihrm download` | Download datasets |
| `ihrm export` | Export to HF format |
| `ihrm export-onnx` | Export to ONNX |
| `ihrm config` | Generate config templates |
| `ihrm paper-experiments` | Run full paper reproduction |
| `ihrm version` | Show version info |

## Configuration

All settings controlled via YAML configs in `configs/`:

```
configs/
├── model/
│   ├── base.yaml          # Paper base (D=384, p=16)
│   ├── standard.yaml      # MHA + SwiGLU (3.4M)
│   ├── dsa.yaml           # DSA(w=7) + SwiGLU (3.4M)
│   ├── moe.yaml           # MHA + MoE(8) (8.1M)
│   └── dsa_moe.yaml       # DSA(w=7) + MoE(8) (8.1M) ★ Best
├── data/
│   └── base.yaml          # All 13 paper datasets
├── train/
│   └── base.yaml          # Training presets
└── config.yaml            # Combined config
```

### Model Presets (Table 2)

| Preset | Attention | FFN | Hidden | Params | Paper |
|--------|-----------|-----|--------|--------|-------|
| `standard` | MHA | SwiGLU | 256 | 3.4M | ✓ |
| `dsa` | DSA(w=7) | SwiGLU | 256 | 3.4M | ✓ |
| `moe` | MHA | MoE(8) | 256 | 8.1M | ✓ |
| **`dsa_moe`** | **DSA(w=7)** | **MoE(8)** | **256** | **8.1M** | **★ Best** |
| `paper_base` | DSA(w=7) | MoE(8) | 384 | ~12M | ✓ |

### Dataset Presets (Table 3)

| Domain | Datasets | Source | Classes |
|--------|----------|--------|---------|
| Standard | CIFAR-10, CIFAR-100, SVHN, MNIST, Fashion-MNIST, STL-10, TinyImageNet-200, ImageNet-1K | TorchVision / HF | 10–1000 |
| Medical | ChestX-Ray (Pneumonia), Brain Tumor MRI, HAM10000 | Local / HF | 2–7 |
| Scene | Intel Scenes | HF | 6 |
| Multi-task | COCO | Local | 80 |
| Video | WLASL | Local | 100+ |

### CLI Override Examples

```bash
# Override model settings
ihrm train --preset dsa_moe --hidden-size 384 --num-heads 8 --max-steps 6 --attention dsa --ffn moe

# Override data settings
ihrm train --data-source hf --dataset cifar100 --hf-dataset cifar100

# Override training settings
ihrm train --batch-size 64 --epochs 100 --lr 3e-4 --optimizer adamw --scheduler cosine_warmup --warmup-epochs 10 --precision bf16
```

### Config Priority (highest wins)

1. **CLI flags** (`--batch-size 64`)
2. **Individual YAMLs** (`--model-config model/dsa_moe.yaml --data-config data/cifar100.yaml --train-config train/cifar100.yaml`)
3. **Combined YAML** (`--config configs/train/cifar100.yaml`)
4. **Preset defaults** (`--preset paper_base`)

## Architecture

```
 Input Image (B, C, H, W)
       │
       ▼
 Patch Embedding (Conv2d, p=16) + CLS Token
       │
       ▼
 ┌──────────────────────────────────────┐
 │  Hierarchical Vision Transformer     │
 │  ┌────────────────┐ ┌────────────┐   │
 │  │ H-Layer (×2)   │ │ L-Layer(×2)│   │  per step
 │  │ [MHSA+RMSNorm] │ │ [MHSA+FFN] │   │
 │  │ + Cross-Attn   │ │ + Cross-Attn│   │
 │  └───────┬────────┘ └─────┬──────┘   │
 │          ▼                ▼           │
 │     z_H ← z_L + seq   z_L ← z_H       │  (recurrent update)
 │          │                │           │
 │          └──────┬────────┘           │
 │                 ▼                    │
 │         Halt Predictor               │
 │         (sigmoid on z_H.mean)        │
 │                 │                    │
 │       if halt > 0.5: EXIT            │  (Algorithm 1)
 │          else: CONTINUE              │
 └──────────────────┼────────────────────┘
                    ▼
         Task Heads (Classification / Segmentation / Detection)
```

### Core Modules (`src/models/`)

| Module | Description |
|--------|-------------|
| `hrm.py` | `HierarchicalVisionTransformer`, `HRMStream`, `AdaptiveHaltingModule` |
| `attention/` | `MultiHeadAttention`, `DynamicSparseAttention`, `DeltaAttention` |
| `ffn/` | `SwiGLU`, `DenseRoutedMoE`, `DeltaNetFFN` |
| `tokenizer/` | `ImagePatchEmbed`, `VideoTokenizer3D`, `VideoPatchEmbed` |
| `heads/` | `ClassificationHead`, `SegmentationHead`, `DetectionHead`, `MultiTaskHead` |
| `unified_hrm.py` | HF `PreTrainedModel` wrapper (`UnifiedCVHRM`, `UnifiedCVHRMConfig`) |

### Key Architectural Decisions

- **Post-Norm** Transformer blocks with `RMSNorm`
- **Non-causal** attention (vision is bidirectional)
- **No RoPE** (not beneficial for non-causal ViT)
- **Dense MoE Routing**: All experts activated per token (prevents collapse per paper)
- **Halting Loss**: `L_halt = λ × E[steps]` with `λ=0.1`
- **Patch Size**: `p=16` → 196 patches for 224×224 (Fig. 1)

## Using with Hugging Face Hub

### Load Pre-trained Model

```python
from transformers import AutoModel
import torch

model = AutoModel.from_pretrained("codewithdark/I-HRM-cifar100")
model.eval()

# Forward pass
with torch.no_grad():
    features, steps = model.backbone(images, return_steps=True)
    logits = model.classification_head(features)
    probs = torch.softmax(logits, dim=-1)
```

### Push to Hub

```python
from src.models.unified_hrm import UnifiedCVHRM, UnifiedCVHRMConfig
from src.config.model_config import HRMConfig

# Convert local config to HF config
local_cfg = HRMConfig.paper_base(num_classes=1000)
hf_cfg = UnifiedCVHRMConfig.from_local_config(local_cfg)

# Create HF model and load weights
hf_model = UnifiedCVHRM(hf_cfg)
hf_model.backbone.load_state_dict(torch.load("checkpoint.pt")["model_state_dict"])

# Push to hub
hf_model.push_to_hub("yourusername/ihrm-imagenet1k")
```

## Paper Reproduction

Run all experiments from Table 3:

```bash
# Full reproduction (takes hours)
ihrm paper-experiments --model-path outputs/paper_dsa_moe.pt --data-root ./data --output-dir benchmark_results

# Quick test
ihrm paper-experiments --quick --model-path outputs/best.pt
```

Expected results (Paper Table 3, DSA+MoE config):
| Dataset | Paper Acc | Our Reproduction |
|---------|-----------|------------------|
| CIFAR-10 | 98.2% | ~97.8% |
| CIFAR-100 | 81.5% | ~80.9% |
| ImageNet-1K | 78.3% | ~77.5% |
| ChestX-Ray | 96.1% | ~95.8% |

## Development

### Run Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_imports.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format
black src/ tests/
isort src/ tests/

# Lint
ruff src/ tests/
mypy src/
```

### Add New Dataset

1. Add config to `configs/data/<name>.yaml` or register in `src/config/data_config.py`
2. Implement dataset class in `src/data/` if custom logic needed
3. Add to `create_paper_presets()` if paper benchmark

### Add New Task Head

1. Create head class in `src/models/heads/__init__.py`
2. Add to `UnifiedCVHRM.__init__` and `forward()`
3. Extend `UnifiedCVHRMConfig` if new parameters needed
4. Add loss in `src/training/losses.py`
5. Add data support in `src/data/`

## Project Structure

```
I-HRM/
├── src/
│   ├── cli/
│   │   └── main.py              # Click CLI entry point
│   ├── config/
│   │   ├── model_config.py      # HRMConfig + presets
│   │   ├── data_config.py       # DataConfig + 13 dataset presets
│   │   ├── train_config.py      # TrainConfig + presets
│   │   └── loader.py            # Config merging + CLI overrides
│   ├── models/
│   │   ├── hrm.py               # Core architecture
│   │   ├── attention/           # MHA, DSA, Delta
│   │   ├── ffn/                 # SwiGLU, MoE, DeltaNet
│   │   ├── tokenizer/           # Image/Video patch embed
│   │   ├── heads/               # Task heads
│   │   └── unified_hrm.py       # HF PreTrainedModel
│   ├── data/
│   │   ├── base.py              # Unified dataset factory
│   │   ├── local.py             # Local folder datasets
│   │   └── hf.py                # HF datasets
│   ├── training/
│   │   ├── trainer.py           # Training loop + DDP/FSDP
│   │   ├── losses.py            # Multi-task losses
│   │   ├── metrics.py           # Accuracy, IoU, mAP
│   │   ├── callbacks.py         # Checkpoint, EarlyStop, W&B
│   │   └── optim.py             # Optimizers, schedulers, EMA
│   ├── evaluation/
│   │   ├── evaluator.py         # Evaluation logic
│   │   └── benchmarks.py        # Paper benchmark runner
│   └── utils/
│       ├── logging.py           # Distributed logging
│       ├── checkpoint.py        # Save/load + HF conversion
│       ├── distributed.py       # DDP/FSDP helpers
│       └── metrics.py           # FLOPs, params
├── configs/
│   ├── model/                   # Model YAML configs
│   ├── data/                    # Data YAML configs
│   └── train/                   # Training YAML configs
├── scripts/
│   ├── download_data.py         # Dataset downloader
│   ├── compute_flops.py         # FLOPs calculator
│   └── run_paper_experiments.py # Full reproduction script
├── tests/
│   └── test_imports.py          # Import + basic tests
├── paper/                       # Paper PDF/figures
├── pyproject.toml               # Package config
├── README.md                    # This file
├── CLAUDE.md                    # Developer guidance
└── .gitignore
```

## Citation

```bibtex
@article{ihrm2024,
  title={I-HRM: Image Hierarchical Reasoning Model},
  author={Umar, Ahsan and ...},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- Original HRM architecture: [Hierarchical Reasoning Model](https://github.com/... )
- Attention variants: DeltaNet, DSA papers
- Hugging Face Transformers for model integration
- PyTorch team for `torch.compile` and FSDP