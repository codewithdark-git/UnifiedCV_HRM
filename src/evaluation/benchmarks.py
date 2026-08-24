"""
Benchmark Suite

Run paper benchmarks across all 13 datasets (Table 3 in paper).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import time
from collections import defaultdict

from src.utils.logging import get_logger
from src.data import get_dataset, get_dataloader
from src.evaluation.evaluator import Evaluator, create_evaluator
from src.config.data_config import DatasetRegistry

logger = get_logger(__name__)


class BenchmarkSuite:
    """
    Paper benchmark runner.

    Runs evaluation across all 13 paper datasets:
    - CIFAR-10, CIFAR-100, SVHN, MNIST, Fashion-MNIST
    - STL-10, TinyImageNet-200, ImageNet-1K
    - ChestX-Ray, Brain Tumor MRI, HAM10000
    - Intel Scenes, COCO Multi-task, WLASL
    """

    PAPER_DATASETS = [
        "cifar10",
        "cifar100",
        "svhn",
        "mnist",
        "fashion_mnist",
        "stl10",
        "tinyimagenet200",
        "imagenet1k",
        "chestxray_pneumonia",
        "brain_tumor_mri",
        "ham10000",
        "intel_scenes",
        "coco_multitask",
    ]

    MEDICAL_DATASETS = [
        "chestxray_pneumonia",
        "brain_tumor_mri",
        "ham10000",
    ]

    VIDEO_DATASETS = [
        "wlasl",
    ]

    def __init__(
        self,
        model: nn.Module,
        task: str = "classification",
        device: Optional[torch.device] = None,
        use_amp: bool = True,
    ):
        """
        Args:
            model: Trained I-HRM model
            task: Task type
            device: Torch device
            use_amp: Use mixed precision
        """
        self.model = model
        self.task = task
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and self.device.type == "cuda"
        self.evaluator = Evaluator(model, task=task, device=self.device, use_amp=use_amp)

    def run(
        self,
        datasets: List[str],
        data_source: str = "local",
        data_root: str = "./data",
        batch_size: int = 128,
        num_workers: int = 4,
        max_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run benchmarks on multiple datasets.

        Args:
            datasets: List of dataset names
            data_source: "local", "hf", or "torchvision"
            data_root: Root data directory
            batch_size: Batch size
            num_workers: DataLoader workers
            max_batches: Limit batches per dataset (for quick testing)

        Returns:
            Dict with results per dataset
        """
        results = {}
        summary = {
            "model": self.model.__class__.__name__,
            "task": self.task,
            "datasets": {},
            "mean_accuracy": 0.0,
        }

        accuracies = []

        for dataset_name in datasets:
            logger.info(f"Running benchmark on {dataset_name}...")

            try:
                # Get dataset config from registry
                data_cfg = DatasetRegistry.get(dataset_name)
                if data_cfg is None:
                    logger.warning(f"Dataset {dataset_name} not in registry, skipping")
                    continue

                # Override source if specified
                if data_source != "local":
                    data_cfg.source = data_source

                # Build dataloader
                test_dataset = get_dataset(data_cfg, split="test")
                test_loader = get_dataloader(
                    test_dataset,
                    type("Config", (), {
                        "batch_size": batch_size,
                        "num_workers": num_workers,
                        "pin_memory": True,
                        "persistent_workers": num_workers > 0,
                        "prefetch_factor": 2,
                    })(),
                    shuffle=False,
                )

                # Evaluate
                start_time = time.time()
                metrics = self.evaluator.evaluate(test_loader, prefix="test", max_batches=max_batches)
                elapsed = time.time() - start_time

                metrics["eval_time_seconds"] = elapsed
                metrics["samples_per_second"] = len(test_dataset) / elapsed

                results[dataset_name] = metrics
                summary["datasets"][dataset_name] = metrics

                # Track accuracy
                acc_key = "test_accuracy" if "test_accuracy" in metrics else "accuracy"
                if acc_key in metrics:
                    accuracies.append(metrics[acc_key])

            except Exception as e:
                logger.error(f"Benchmark failed for {dataset_name}: {e}")
                results[dataset_name] = {"error": str(e)}

        if accuracies:
            summary["mean_accuracy"] = sum(accuracies) / len(accuracies)

        return {
            "results": results,
            "summary": summary,
        }

    def run_paper_benchmarks(
        self,
        data_root: str = "./data",
        batch_size: int = 128,
        quick: bool = False,
    ) -> Dict:
        """
        Run full paper benchmark suite (Table 3).

        Args:
            data_root: Data directory
            batch_size: Batch size
            quick: Run quick version (fewer batches)
        """
        max_batches = 10 if quick else None

        # Classify datasets by domain
        standard_datasets = [
            "cifar10", "cifar100", "svhn", "mnist", "fashion_mnist",
            "stl10", "tinyimagenet200", "imagenet1k",
        ]
        medical_datasets = self.MEDICAL_DATASETS
        video_datasets = self.VIDEO_DATASETS

        all_results = {}

        # Standard vision
        if standard_datasets:
            logger.info("Running standard vision benchmarks...")
            all_results["standard"] = self.run(
                standard_datasets,
                data_source="torchvision" if "cifar" in standard_datasets[0] else "hf",
                data_root=data_root,
                batch_size=batch_size,
                max_batches=max_batches,
            )

        # Medical
        if medical_datasets:
            logger.info("Running medical imaging benchmarks...")
            all_results["medical"] = self.run(
                medical_datasets,
                data_source="local",
                data_root=data_root,
                batch_size=min(batch_size, 32),
                max_batches=max_batches,
            )

        # Video
        if video_datasets:
            logger.info("Running video benchmarks...")
            all_results["video"] = self.run(
                video_datasets,
                data_source="local",
                data_root=data_root,
                batch_size=min(batch_size, 16),
                max_batches=max_batches,
            )

        return all_results

    def save_results(self, results: Dict, output_path: Union[str, Path]):
        """Save benchmark results to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Benchmark results saved to {path}")


def run_paper_experiments(
    model_path: str,
    data_root: str = "./data",
    output_dir: str = "./benchmark_results",
    quick: bool = False,
):
    """
    Run complete paper experiments from model checkpoint.

    Args:
        model_path: Path to model checkpoint or HF repo
        data_root: Data directory
        output_dir: Output directory for results
        quick: Quick test mode
    """
    # Load model
    if "hf.co" in model_path or "/" in model_path and not Path(model_path).exists():
        # HF repo
        from src.models.unified_hrm import UnifiedCVHRM, UnifiedCVHRMConfig
        hf_config = UnifiedCVHRMConfig.from_pretrained(model_path)
        model = UnifiedCVHRM.from_pretrained(model_path, config=hf_config).backbone
    else:
        # Local checkpoint
        from src.models.hrm import HierarchicalVisionTransformer
        from src.config.model_config import HRMConfig

        checkpoint = torch.load(model_path, map_location="cpu")
        model_cfg = HRMConfig(**checkpoint.get("model_config", {}))
        model = HierarchicalVisionTransformer(model_cfg)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    # Run benchmarks
    suite = BenchmarkSuite(model, task="classification")
    results = suite.run_paper_benchmarks(
        data_root=data_root,
        quick=quick,
    )

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    suite.save_results(results, Path(output_dir) / "paper_benchmarks.json")

    # Print summary
    print("\n" + "="*60)
    print("PAPER BENCHMARK RESULTS")
    print("="*60)

    for category, cat_results in results.items():
        print(f"\n{category.upper()}:")
        if "results" in cat_results:
            for ds, metrics in cat_results["results"].items():
                acc = metrics.get("test_accuracy", metrics.get("accuracy", 0))
                print(f"  {ds}: {acc:.4f}")

    return results


# Alias for backward compatibility
run_paper_benchmarks = run_paper_experiments