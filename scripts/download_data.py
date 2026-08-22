#!/usr/bin/env python
"""
Download Datasets Script

Downloads all paper benchmark datasets from HuggingFace or TorchVision.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config.loader import ConfigManager
from src.config.data_config import DatasetRegistry
from src.data import get_dataset


def download_dataset(dataset_name: str, data_root: str, source: str = "auto"):
    """Download a single dataset."""
    cfg = DatasetRegistry.get(dataset_name)

    if cfg is None:
        print(f"Dataset {dataset_name} not in registry, trying to infer...")
        from src.config.data_config import DataConfig, DataSource
        # Try to create config
        cfg = DataConfig(
            name=dataset_name,
            source=DataSource(source) if source != "auto" else DataSource.LOCAL,
        )

    # Override root dir
    cfg.root_dir = os.path.join(data_root, dataset_name)

    print(f"Downloading {cfg.name} ({cfg.display_name})...")
    print(f"  Source: {cfg.source.value}")
    print(f"  Root: {cfg.root_dir}")

    try:
        # This will trigger download for TorchVision/HF datasets
        train_ds = get_dataset(cfg, split="train")
        val_ds = get_dataset(cfg, split="val")
        test_ds = get_dataset(cfg, split="test")

        print(f"  Train: {len(train_ds)} samples")
        print(f"  Val: {len(val_ds)} samples")
        print(f"  Test: {len(test_ds)} samples")
        print(f"  [OK] {dataset_name} downloaded successfully!")

    except Exception as e:
        print(f"  [FAIL] Failed to download {dataset_name}: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Download paper benchmark datasets")
    parser.add_argument("--dataset", nargs="+", default="all",
                        help="Dataset name(s) or 'all' for all 13 paper datasets")
    parser.add_argument("--data-root", default="./data",
                        help="Root directory for datasets")
    parser.add_argument("--source", choices=["local", "hf", "torchvision", "auto"],
                        default="auto", help="Data source override")
    parser.add_argument("--list", action="store_true",
                        help="List available datasets")

    args = parser.parse_args()

    if args.list:
        print("Available paper benchmark datasets:")
        for name in sorted(DatasetRegistry.list()):
            cfg = DatasetRegistry.get(name)
            print(f"  {name}: {cfg.display_name} ({cfg.source.value})")
        return

    # Paper benchmark datasets
    all_datasets = [
        "cifar10", "cifar100", "svhn", "mnist", "fashion_mnist",
        "stl10", "tinyimagenet200", "imagenet1k",
        "chestxray_pneumonia", "brain_tumor_mri", "ham10000",
        "intel_scenes", "coco_multitask",
    ]

    datasets = args.dataset if args.dataset != "all" else all_datasets

    print(f"Downloading {len(datasets)} datasets to {args.data_root}...")
    print("=" * 60)

    success = 0
    for ds in datasets:
        if download_dataset(ds, args.data_root, args.source):
            success += 1

    print("=" * 60)
    print(f"Completed: {success}/{len(datasets)} datasets downloaded successfully")

    if success < len(datasets):
        sys.exit(1)


if __name__ == "__main__":
    main()