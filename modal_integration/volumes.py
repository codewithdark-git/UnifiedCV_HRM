"""
Modal Volume Definitions for HRM Training

Defines persistent volumes for datasets and checkpoints.
"""

import modal as modal_client

# Dataset volume - stores all downloaded datasets
datasets_volume = modal_client.Volume.from_name("hrm-datasets", create_if_missing=True)

# Checkpoint volume - stores model checkpoints, logs, W&B runs
checkpoints_volume = modal_client.Volume.from_name("hrm-checkpoints", create_if_missing=True)

# Optional: HF cache volume for faster downloads
hf_cache_volume = modal_client.Volume.from_name("hrm-hf-cache", create_if_missing=True)

# Volume mount paths
VOLUME_MOUNTS = {
    "/data": datasets_volume,
    "/checkpoints": checkpoints_volume,
    "/root/.cache/huggingface": hf_cache_volume,
}

# Dataset directory structure on volume
DATASET_PATHS = {
    "cifar10": "/data/cifar10",
    "cifar100": "/data/cifar100",
    "svhn": "/data/svhn",
    "mnist": "/data/mnist",
    "fashion_mnist": "/data/fashion_mnist",
    "stl10": "/data/stl10",
    "tinyimagenet200": "/data/tinyimagenet200",
    "imagenet1k": "/data/imagenet1k",
    "wlasl100": "/data/wlasl100",
    "wlasl300": "/data/wlasl300",
    "wlasl1000": "/data/wlasl1000",
    "pose_action": "/data/pose_action",
    "ucf101": "/data/ucf101",
    "ucf_crime": "/data/ucf_crime",
}

# Checkpoint directory structure
CHECKPOINT_PATHS = {
    "base": "/checkpoints",
    "wandb": "/checkpoints/wandb",
    "tensorboard": "/checkpoints/tensorboard",
    "models": "/checkpoints/models",
}