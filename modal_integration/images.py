"""
Modal Docker Image Definitions for HRM Training

Defines the container images with all necessary dependencies.
"""

import modal as modal_client

# ─────────────────────────────────────────────────────────────────
# Base Image with All Dependencies
# ─────────────────────────────────────────────────────────────────

hrm_base_image = (
    modal_client.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "wget",
        "unzip",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        "ffmpeg",  # For video processing
    )
    .pip_install(
        # Core ML
        "torch>=2.3.0",
        "torchvision>=0.18.0",
        "torchaudio>=2.3.0",
        # Hugging Face
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "huggingface-hub>=0.22.0",
        "safetensors>=0.4.0",
        "datasets>=2.18.0",
        # Vision & Model Utils
        "timm>=1.0.0",
        "fvcore>=0.1.5",
        "thop>=0.1.1",
        "einops>=0.7.0",
        # Config & CLI
        "pyyaml>=6.0.1",
        "click>=8.1.0",
        # Data Processing
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "opencv-python>=4.9.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        # Training & Logging
        "tqdm>=4.66.0",
        "wandb>=0.16.0",
        "tensorboard>=2.15.0",
        "rich>=13.0.0",
        # COCO & Evaluation
        "pycocotools>=2.0.7",
        # Analysis
        "matplotlib>=3.7.0",
        "pandas>=2.1.0",
        # ZIM Archive Support
        "zimply>=0.1.0",
    )
    .run_commands("mkdir -p /root/HRM/src")
)

# ─────────────────────────────────────────────────────────────────
# Specialized Images
# ─────────────────────────────────────────────────────────────────

# Lightweight image for dataset download only
download_image = (
    modal_client.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "unzip", "git")
    .pip_install(
        "datasets>=2.18.0",
        "huggingface-hub>=0.22.0",
        "torchvision>=0.18.0",
        "tqdm>=4.66.0",
        "zimply>=0.1.0",
    )
)

# Image for evaluation only (no training deps)
eval_image = (
    modal_client.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.3.0",
        "torchvision>=0.18.0",
        "transformers>=4.40.0",
        "huggingface-hub>=0.22.0",
        "safetensors>=0.4.0",
        "datasets>=2.18.0",
        "timm>=1.0.0",
        "einops>=0.7.0",
        "pyyaml>=6.0.1",
        "click>=8.1.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "opencv-python>=4.9.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "tqdm>=4.66.0",
        "rich>=13.0.0",
        "pycocotools>=2.0.7",
    )
    .run_commands("mkdir -p /root/HRM/src")
)

# Image with DeepSpeed for large-scale training
deepspeed_image = hrm_base_image.pip_install(
    "deepspeed>=0.14.0",
    "mpi4py>=3.1.5",
)

# Image with FSDP support (already in PyTorch 2.3+)
fsdp_image = hrm_base_image

# ─────────────────────────────────────────────────────────────────
# Image Selection Helper
# ─────────────────────────────────────────────────────────────────

def get_image_for_task(task: str) -> modal_client.Image:
    """Get appropriate image for a task."""
    images = {
        "train": hrm_base_image,
        "train_deepspeed": deepspeed_image,
        "train_fsdp": fsdp_image,
        "download": download_image,
        "eval": eval_image,
        "benchmark": eval_image,
    }
    return images.get(task, hrm_base_image)


# ─────────────────────────────────────────────────────────────────
# GPU Configuration Mapping
# ─────────────────────────────────────────────────────────────────

# Cost-effective GPU mapping per dataset
GPU_CONFIGS = {
    # Small image datasets - single/dual A10G
    "cifar10": "A10G:2",
    "cifar100": "A10G:2",
    "svhn": "A10G:1",
    "mnist": "A10G:1",
    "fashion_mnist": "A10G:1",
    "stl10": "A10G:2",
    
    # Medium image datasets - 4x A10G
    "tinyimagenet200": "A10G:4",
    
    # Large image datasets - 8x A100-80GB
    "imagenet1k": "A100-80GB:8",
    
    # Video datasets - 4x A10G for small, 8x A100 for large
    "wlasl100": "A10G:4",
    "wlasl300": "A10G:4",
    "wlasl1000": "A100-40GB:8",
    "pose_action": "A10G:4",
    "ucf101": "A10G:4",
    "ucf_crime": "A10G:4",
}

def get_gpu_config(dataset: str) -> str:
    """Get GPU configuration for a dataset."""
    return GPU_CONFIGS.get(dataset, "A10G:2")


# ─────────────────────────────────────────────────────────────────
# Estimated Training Times & Costs
# ─────────────────────────────────────────────────────────────────

ESTIMATED_TRAINING = {
    "cifar10": {"hours": 2, "gpu_hrs": 4, "est_cost": 3.00},
    "cifar100": {"hours": 3, "gpu_hrs": 6, "est_cost": 4.50},
    "svhn": {"hours": 1, "gpu_hrs": 1, "est_cost": 0.75},
    "mnist": {"hours": 0.5, "gpu_hrs": 0.5, "est_cost": 0.38},
    "fashion_mnist": {"hours": 0.5, "gpu_hrs": 0.5, "est_cost": 0.38},
    "stl10": {"hours": 2, "gpu_hrs": 4, "est_cost": 3.00},
    "tinyimagenet200": {"hours": 6, "gpu_hrs": 24, "est_cost": 18.00},
    "imagenet1k": {"hours": 36, "gpu_hrs": 288, "est_cost": 1008.00},
    "wlasl100": {"hours": 8, "gpu_hrs": 32, "est_cost": 24.00},
    "wlasl300": {"hours": 12, "gpu_hrs": 48, "est_cost": 36.00},
    "wlasl1000": {"hours": 24, "gpu_hrs": 192, "est_cost": 480.00},
    "pose_action": {"hours": 8, "gpu_hrs": 32, "est_cost": 24.00},
    "ucf101": {"hours": 10, "gpu_hrs": 40, "est_cost": 30.00},
}