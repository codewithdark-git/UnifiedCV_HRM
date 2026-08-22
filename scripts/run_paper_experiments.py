#!/usr/bin/env python
"""
Run Paper Experiments

Execute all paper benchmark experiments (Table 3 in paper).
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import yaml

from src.models.unified_hrm import UnifiedCVHRM, UnifiedCVHRMConfig
from src.evaluation.benchmarks import BenchmarkSuite, run_paper_experiments
from src.config.loader import ConfigManager, load_config
from src.config.model_config import HRMConfig
from src.config.data_config import DataConfig, DatasetRegistry


def load_model_from_checkpoint(path: str, strict: bool = False):
    """Load model from local checkpoint or HF repo."""
    path = Path(path)

    if path.exists():
        # Local checkpoint
        ckpt = torch.load(path, map_location="cpu")
        if "model_config" in ckpt:
            model_cfg = HRMConfig.from_dict(ckpt["model_config"])
        elif "config" in ckpt:
            model_cfg = HRMConfig.from_dict(ckpt["config"])
        else:
            raise ValueError("No model_config in checkpoint")

        from src.models.hrm import HierarchicalVisionTransformer
        model = HierarchicalVisionTransformer(model_cfg)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=strict)
        return model, model_cfg

    else:
        # HF repo
        hf_config = UnifiedCVHRMConfig.from_pretrained(str(path))
        hf_model = UnifiedCVHRM.from_pretrained(str(path), config=hf_config)
        return hf_model.backbone, hf_config.to_local_config()


def main():
    parser = argparse.ArgumentParser(description="Run paper benchmark experiments")
    parser.add_argument("--model", required=True,
                        help="Model checkpoint path or HF repo")
    parser.add_argument("--data-root", default="./data",
                        help="Root data directory")
    parser.add_argument("--output", default="./benchmark_results",
                        help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode (few batches)")
    parser.add_argument("--datasets", nargs="+",
                        help="Specific datasets to run (default: all 13)")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cuda, cpu")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use mixed precision")

    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Loading model from: {args.model}")

    # Load model
    model, model_cfg = load_model_from_checkpoint(args.model)
    model = model.to(device)
    model.eval()

    # Determine datasets
    if args.datasets:
        dataset_names = args.datasets
    else:
        dataset_names = [
            "cifar10", "cifar100", "svhn", "mnist", "fashion_mnist",
            "stl10", "tinyimagenet200", "imagenet1k",
            "chestxray_pneumonia", "brain_tumor_mri", "ham10000",
            "intel_scenes", "coco_multitask",
        ]

    print(f"Running benchmarks on {len(dataset_names)} datasets...")
    print(f"Data root: {args.data_root}")
    print(f"Output: {args.output}")

    # Run benchmarks
    benchmark = BenchmarkSuite(model, task="classification")
    results = benchmark.run(
        datasets=dataset_names,
        data_source="local",
        data_root=args.data_root,
        batch_size=args.batch_size,
    )

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "summary.txt", "w") as f:
        f.write("PAPER BENCHMARK RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Config: {model_cfg}\n\n")

        for ds, metrics in results.get("results", {}).items():
            acc = metrics.get("accuracy", metrics.get("test_accuracy", 0))
            f.write(f"{ds}: accuracy={acc:.4f}\n")

    print(f"\nResults saved to {output_dir}")
    print("\nSummary:")
    for ds, metrics in results.get("results", {}).items():
        acc = metrics.get("accuracy", metrics.get("test_accuracy", 0))
        print(f"  {ds}: {acc:.4f}")


if __name__ == "__main__":
    main()