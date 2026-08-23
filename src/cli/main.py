#!/usr/bin/env python
"""
I-HRM: Image Hierarchical Reasoning Model - CLI Entry Point

A unified CLI for training, evaluating, and benchmarking I-HRM models
on local datasets and Hugging Face datasets.

Usage:
    ihrm train --config configs/train/classification.yaml --data.cifar10
    ihrm evaluate --model checkpoint.pt --data cifar10 --task classification
    ihrm benchmark --model hf-repo/ihrm-base --datasets cifar10 cifar100 imagenet1k
    ihrm predict --model checkpoint.pt --image image.jpg --task classification
    ihrm download --dataset cifar10 --output data/
    ihrm config --init configs/
"""

import click
import os
import sys
import yaml
import torch
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.loader import ConfigManager, merge_configs
from src.config.model_config import HRMConfig, AttentionType, FFNType, TaskType, get_preset_config
from src.config.data_config import DataConfig, DataSource, DatasetRegistry
from src.config.train_config import TrainConfig, OptimizerType, SchedulerType
from src.models.unified_hrm import UnifiedCVHRM, UnifiedCVHRMConfig
from src.models.hrm import HierarchicalVisionTransformer
from src.data import get_dataset, get_dataloader
from src.training.trainer import Trainer
from src.training.losses import MultiTaskLoss
from src.evaluation.evaluator import Evaluator
from src.evaluation.benchmarks import BenchmarkSuite
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# Config Loading Utilities
# ─────────────────────────────────────────────────────────────────

def load_merged_config(
    config_path: Optional[str] = None,
    model_config: Optional[Dict] = None,
    data_config: Optional[Dict] = None,
    train_config: Optional[Dict] = None,
    **overrides,
) -> tuple:
    """Load and merge configs from YAML + CLI overrides."""
    cm = ConfigManager()

    # Load base configs
    model_cfg = HRMConfig()
    data_cfg = DataConfig(name="default")
    train_cfg = TrainConfig()

    if config_path:
        loaded = cm.load_config(config_path)
        if "model" in loaded:
            model_cfg = HRMConfig(**loaded["model"])
        if "data" in loaded:
            data_cfg = DataConfig(**loaded["data"])
        if "train" in loaded:
            train_cfg = TrainConfig(**loaded["train"])

    # Apply CLI overrides - map data config keys to DataConfig fields
    if model_config:
        model_cfg = HRMConfig(**{**model_cfg.__dict__, **model_config})
    if data_config:
        # Map CLI data keys to DataConfig fields
        mapped_data = {}
        for k, v in data_config.items():
            if k == "dataset":
                mapped_data["name"] = v
                mapped_data["tv_dataset_name"] = v
            elif k == "data_path":
                mapped_data["root_dir"] = v
                mapped_data["tv_root"] = v  # Also set for torchvision
            elif k == "hf_dataset":
                mapped_data["hf_repo_id"] = v
            else:
                mapped_data[k] = v
        data_cfg = DataConfig(**{**data_cfg.__dict__, **mapped_data})
    if train_config:
        train_cfg = TrainConfig(**{**train_cfg.__dict__, **train_config})

    # Apply remaining overrides as flat dicts
    if overrides:
        for section, values in overrides.items():
            if section == "model":
                model_cfg = HRMConfig(**{**model_cfg.__dict__, **values})
            elif section == "data":
                data_cfg = DataConfig(**{**data_cfg.__dict__, **values})
            elif section == "train":
                train_cfg = TrainConfig(**{**train_cfg.__dict__, **values})

    return model_cfg, data_cfg, train_cfg


def build_model(config: HRMConfig) -> HierarchicalVisionTransformer:
    """Build model from config."""
    return HierarchicalVisionTransformer(config)


def build_hf_model(config: UnifiedCVHRMConfig) -> UnifiedCVHRM:
    """Build HF-compatible model from config."""
    return UnifiedCVHRM(config)


# ─────────────────────────────────────────────────────────────────
# CLI Commands
# ─────────────────────────────────────────────────────────────────

@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to config YAML")
@click.pass_context
def cli(ctx, verbose: bool, config: str):
    """I-HRM: Image Hierarchical Reasoning Model"""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["config_path"] = config
    setup_logging(verbose=verbose)


@cli.command()
@click.option("--model", type=click.Path(exists=True), help="Path to model checkpoint")
@click.option("--model-hf", help="Hugging Face model repo (e.g., user/ihrm-base)")
@click.option("--preset", type=click.Choice(["standard", "dsa", "moe", "dsa_moe", "paper_base"]), help="Model preset")
@click.option("--hidden-size", type=int, default=384, help="Hidden dimension")
@click.option("--num-heads", type=int, default=8, help="Number of attention heads")
@click.option("--num-layers", type=int, default=2, help="Layers per stream")
@click.option("--max-steps", type=int, default=6, help="Max reasoning steps")
@click.option("--attention", type=click.Choice(["mha", "dsa", "delta"]), default="dsa")
@click.option("--ffn", type=click.Choice(["swiglu", "moe", "deltanet"]), default="moe")
@click.option("--moe-experts", type=int, default=8, help="Number of MoE experts")
@click.option("--image-size", type=int, default=224, help="Input image size")
@click.option("--patch-size", type=int, default=16, help="Patch size")
@click.option("--num-classes", type=int, default=1000, help="Number of classes")
@click.option("--task", type=click.Choice(["classification", "segmentation", "detection", "all"]), default="classification")
@click.option("--data-source", type=click.Choice(["local", "hf", "torchvision"]), default="local")
@click.option("--dataset", help="Dataset name (cifar10, cifar100, imagenet1k, etc.)")
@click.option("--data-path", type=click.Path(), help="Local dataset path")
@click.option("--hf-dataset", help="HF dataset name (e.g., cifar10, imagenet-1k)")
@click.option("--batch-size", type=int, default=128, help="Batch size")
@click.option("--epochs", type=int, default=100, help="Number of epochs")
@click.option("--lr", type=float, default=3e-4, help="Learning rate")
@click.option("--optimizer", type=click.Choice(["adamw", "sgd", "lion"]), default="adamw")
@click.option("--scheduler", type=click.Choice(["cosine", "cosine_warmup", "step", "constant"]), default="cosine_warmup")
@click.option("--warmup-epochs", type=int, default=10, help="Warmup epochs")
@click.option("--weight-decay", type=float, default=0.05, help="Weight decay")
@click.option("--precision", type=click.Choice(["fp32", "fp16", "bf16"]), default="bf16")
@click.option("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
@click.option("--max-grad-norm", type=float, default=1.0, help="Max gradient norm")
@click.option("--output-dir", type=click.Path(), default="outputs", help="Output directory")
@click.option("--log-dir", type=click.Path(), default="logs", help="Log directory")
@click.option("--wandb", is_flag=True, help="Use Weights & Biases logging")
@click.option("--wandb-project", default="ihrm", help="W&B project name")
@click.option("--wandb-run", help="W&B run name")
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--resume", type=click.Path(exists=True), help="Resume from checkpoint")
@click.option("--compile", is_flag=True, help="Use torch.compile")
@click.option("--distributed", type=click.Choice(["ddp", "fsdp", "deepspeed"]), help="Distributed backend")
@click.option("--num-workers", type=int, default=8, help="Data loader workers")
@click.option("--pin-memory", is_flag=True, default=True, help="Pin memory")
@click.pass_context
def train(
    ctx,
    model: str,
    model_hf: str,
    preset: str,
    hidden_size: int,
    num_heads: int,
    num_layers: int,
    max_steps: int,
    attention: str,
    ffn: str,
    moe_experts: int,
    image_size: int,
    patch_size: int,
    num_classes: int,
    task: str,
    data_source: str,
    dataset: str,
    data_path: str,
    hf_dataset: str,
    batch_size: int,
    epochs: int,
    lr: float,
    optimizer: str,
    scheduler: str,
    warmup_epochs: int,
    weight_decay: float,
    precision: str,
    grad_accum: int,
    max_grad_norm: float,
    output_dir: str,
    log_dir: str,
    wandb: bool,
    wandb_project: str,
    wandb_run: str,
    seed: int,
    resume: str,
    compile: bool,
    distributed: str,
    num_workers: int,
    pin_memory: bool,
):
    """Train I-HRM model on local or HF datasets."""
    verbose = ctx.obj.get("verbose", False)
    config_path = ctx.obj.get("config_path")

    # Build config overrides from CLI
    model_overrides = {}
    if preset:
        model_overrides = get_preset_config(preset).__dict__
    if hidden_size != 384: model_overrides["hidden_size"] = hidden_size
    if num_heads != 8: model_overrides["num_heads"] = num_heads
    if num_layers != 2: model_overrides["num_layers_per_stream"] = num_layers
    if max_steps != 6: model_overrides["max_steps"] = max_steps
    if attention != "dsa": model_overrides["attention_type"] = attention
    if ffn != "moe": model_overrides["ffn_type"] = ffn
    if moe_experts != 8: model_overrides["moe_num_experts"] = moe_experts
    if image_size != 224: model_overrides["image_size"] = image_size
    if patch_size != 16: model_overrides["patch_size"] = patch_size
    if num_classes != 1000: model_overrides["num_classes"] = num_classes
    if task != "classification": model_overrides["task"] = task

    data_overrides = {}
    if data_source != "local": data_overrides["source"] = data_source
    if dataset: data_overrides["dataset"] = dataset
    if data_path: data_overrides["data_path"] = data_path
    if hf_dataset: data_overrides["hf_dataset"] = hf_dataset

    train_overrides = {}
    if batch_size != 128: train_overrides["batch_size"] = batch_size
    if epochs != 100: train_overrides["epochs"] = epochs
    if lr != 3e-4: train_overrides["lr"] = lr
    if optimizer != "adamw": train_overrides["optimizer"] = optimizer
    if scheduler != "cosine_warmup": train_overrides["scheduler"] = scheduler
    if warmup_epochs != 10: train_overrides["warmup_epochs"] = warmup_epochs
    if weight_decay != 0.05: train_overrides["weight_decay"] = weight_decay
    if precision != "bf16": train_overrides["precision"] = precision
    if grad_accum != 1: train_overrides["grad_accum"] = grad_accum
    if max_grad_norm != 1.0: train_overrides["max_grad_norm"] = max_grad_norm
    if output_dir != "outputs": train_overrides["output_dir"] = output_dir
    if log_dir != "logs": train_overrides["log_dir"] = log_dir
    if wandb: train_overrides["wandb"] = True
    if wandb_project != "ihrm": train_overrides["wandb_project"] = wandb_project
    if wandb_run: train_overrides["wandb_run"] = wandb_run
    if seed != 42: train_overrides["seed"] = seed
    if resume: train_overrides["resume"] = resume
    if compile: train_overrides["compile"] = True
    if distributed: train_overrides["distributed"] = distributed
    if num_workers != 8: train_overrides["num_workers"] = num_workers
    if not pin_memory: train_overrides["pin_memory"] = False

    # Load merged config
    model_cfg, data_cfg, train_cfg = load_merged_config(
        config_path,
        model_overrides if model_overrides else None,
        data_overrides if data_overrides else None,
        train_overrides if train_overrides else None,
    )

    # Set seed
    torch.manual_seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)

    # Build datasets
    logger.info(f"Loading dataset: {data_cfg.name} from {data_cfg.source.value}")
    train_dataset = get_dataset(data_cfg, split="train")
    val_dataset = get_dataset(data_cfg, split="val")
    test_dataset = get_dataset(data_cfg, split="test")

    train_loader = get_dataloader(train_dataset, train_cfg, shuffle=True)
    val_loader = get_dataloader(val_dataset, train_cfg, shuffle=False)
    test_loader = get_dataloader(test_dataset, train_cfg, shuffle=False)

    # Build model
    logger.info(f"Building model: {model_cfg.task.value}")
    model = build_model(model_cfg)

    # Load checkpoint if provided
    if model and os.path.exists(model):
        logger.info(f"Loading checkpoint from {model}")
        checkpoint = torch.load(model, map_location="cpu")
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)

    # Load HF model if provided
    if model_hf:
        logger.info(f"Loading HF model from {model_hf}")
        hf_config = UnifiedCVHRMConfig.from_pretrained(model_hf)
        hf_model = UnifiedCVHRM.from_pretrained(model_hf, config=hf_config)
        model = hf_model.get_backbone()

    # Setup trainer
    logger.info("Setting up trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=train_cfg,
        model_config=model_cfg,
        data_config=data_cfg,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    save_path = Path(train_cfg.output_dir) / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model_cfg.__dict__,
        "train_config": train_cfg.__dict__,
    }, save_path)
    logger.info(f"Model saved to {save_path}")

    # Push to HF Hub if requested
    if train_cfg.push_to_hub and train_cfg.hub_repo:
        logger.info(f"Pushing to HF Hub: {train_cfg.hub_repo}")
        hf_config = UnifiedCVHRMConfig.from_local_config(model_cfg)
        hf_model = UnifiedCVHRM(hf_config)
        hf_model.backbone.load_state_dict(model.state_dict())
        hf_model.push_to_hub(train_cfg.hub_repo)


@cli.command()
@click.option("--model", required=True, type=click.Path(exists=True), help="Path to model checkpoint")
@click.option("--model-hf", help="Hugging Face model repo")
@click.option("--dataset", required=True, help="Dataset name")
@click.option("--data-source", type=click.Choice(["local", "hf", "torchvision"]), default="local")
@click.option("--data-path", type=click.Path(), help="Local dataset path")
@click.option("--hf-dataset", help="HF dataset name")
@click.option("--task", type=click.Choice(["classification", "segmentation", "detection", "all"]), default="classification")
@click.option("--batch-size", type=int, default=128)
@click.option("--output", type=click.Path(), help="Output file for predictions")
@click.option("--metrics", multiple=True, default=["accuracy", "top5"], help="Metrics to compute")
@click.pass_context
def evaluate(ctx, model, model_hf, dataset, data_source, data_path, hf_dataset, task, batch_size, output, metrics):
    """Evaluate model on dataset."""
    verbose = ctx.obj.get("verbose", False)

    # Build data config - map CLI args to DataConfig fields
    data_cfg = DataConfig(
        source=DataSource(data_source),
        name=dataset,
        tv_dataset_name=dataset,
        root_dir=data_path,
        hf_repo_id=hf_dataset,
    )

    # Load model
    if model_hf:
        hf_config = UnifiedCVHRMConfig.from_pretrained(model_hf)
        hf_model = UnifiedCVHRM.from_pretrained(model_hf, config=hf_config)
        model = hf_model.get_backbone()
    else:
        checkpoint = torch.load(model, map_location="cpu")
        model_cfg = HRMConfig(**checkpoint.get("model_config", {}))
        model = build_model(model_cfg)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)

    model.eval()

    # Get dataset
    test_dataset = get_dataset(data_cfg, split="test")
    test_loader = get_dataloader(test_dataset, train_cfg=type('Config', (), {"batch_size": batch_size, "num_workers": 4, "pin_memory": True, "persistent_workers": True, "prefetch_factor": 2})())

    # Evaluate
    evaluator = Evaluator(model, task=task)
    results = evaluator.evaluate(test_loader, metrics=list(metrics))

    logger.info(f"Evaluation results: {results}")

    if output:
        import json
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output}")


@cli.command()
@click.option("--model", required=True, help="Model path or HF repo")
@click.option("--datasets", multiple=True, required=True, help="Datasets to benchmark")
@click.option("--data-source", type=click.Choice(["local", "hf", "torchvision"]), default="local")
@click.option("--data-root", type=click.Path(), default="data", help="Root data directory")
@click.option("--batch-size", type=int, default=128)
@click.option("--task", type=click.Choice(["classification", "segmentation", "detection", "all"]), default="classification")
@click.option("--output", type=click.Path(), help="Output file for results")
@click.option("--wandb", is_flag=True, help="Log to W&B")
@click.pass_context
def benchmark(ctx, model, datasets, data_source, data_root, batch_size, task, output, wandb):
    """Run paper benchmarks on multiple datasets."""
    verbose = ctx.obj.get("verbose", False)

    # Load model
    if "hf.co" in model or "/" in model and not os.path.exists(model):
        hf_config = UnifiedCVHRMConfig.from_pretrained(model)
        hf_model = UnifiedCVHRM.from_pretrained(model, config=hf_config)
        backbone = hf_model.get_backbone()
    else:
        checkpoint = torch.load(model, map_location="cpu")
        model_cfg = HRMConfig(**checkpoint.get("model_config", {}))
        backbone = build_model(model_cfg)
        backbone.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)

    # Run benchmarks
    benchmark_suite = BenchmarkSuite(backbone, task=task)
    results = benchmark_suite.run(
        datasets=list(datasets),
        data_source=data_source,
        data_root=data_root,
        batch_size=batch_size,
    )

    logger.info(f"Benchmark results: {results}")

    if output:
        import json
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output}")


@cli.command()
@click.option("--model", required=True, type=click.Path(exists=True), help="Path to model checkpoint")
@click.option("--model-hf", help="Hugging Face model repo")
@click.option("--image", required=True, type=click.Path(exists=True), help="Input image path")
@click.option("--task", type=click.Choice(["classification", "segmentation", "detection"]), default="classification")
@click.option("--top-k", type=int, default=5, help="Top K predictions")
@click.option("--output", type=click.Path(), help="Output file for predictions")
@click.pass_context
def predict(ctx, model, model_hf, image, task, top_k, output):
    """Run inference on a single image."""
    verbose = ctx.obj.get("verbose", False)

    # Load model
    if model_hf:
        hf_config = UnifiedCVHRMConfig.from_pretrained(model_hf)
        hf_model = UnifiedCVHRM.from_pretrained(model_hf, config=hf_config)
    else:
        checkpoint = torch.load(model, map_location="cpu")
        model_cfg = HRMConfig(**checkpoint.get("model_config", {}))
        hf_config = UnifiedCVHRMConfig.from_local_config(model_cfg)
        hf_model = UnifiedCVHRM(hf_config)
        hf_model.backbone.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)

    # Load and preprocess image
    from PIL import Image
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((hf_config.image_size, hf_config.image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image).convert("RGB")
    pixel_values = transform(img).unsqueeze(0)

    # Predict
    result = hf_model.generate(pixel_values, task=task, top_k=top_k)

    logger.info(f"Prediction: {result}")

    if output:
        import json
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Prediction saved to {output}")


@cli.command()
@click.option("--dataset", required=True, help="Dataset name (cifar10, cifar100, imagenet1k, etc.)")
@click.option("--output", type=click.Path(), default="data", help="Output directory")
@click.option("--split", multiple=True, default=["train", "val", "test"], help="Splits to download")
@click.pass_context
def download(ctx, dataset, output, split):
    """Download dataset from Hugging Face or torchvision."""
    verbose = ctx.obj.get("verbose", False)

    from datasets import load_dataset
    import torchvision.datasets as tv_datasets

    logger.info(f"Downloading {dataset}...")

    # Check if it's a torchvision dataset
    tv_datasets = {
        "cifar10": tv_datasets.CIFAR10,
        "cifar100": tv_datasets.CIFAR100,
        "mnist": tv_datasets.MNIST,
        "fashion_mnist": tv_datasets.FashionMNIST,
        "stl10": tv_datasets.STL10,
        "imagenet": tv_datasets.ImageNet,
    }

    if dataset.lower() in tv_datasets:
        for s in split:
            train = s != "test"
            ds = tv_datasets[dataset.lower()](
                root=output,
                train=train,
                download=True,
            )
            logger.info(f"Downloaded {dataset} {s} to {output}")
    else:
        # Try HF datasets
        try:
            for s in split:
                ds = load_dataset(dataset, split=s)
                ds.save_to_disk(os.path.join(output, dataset, s))
                logger.info(f"Downloaded {dataset} {s} to {output}")
        except Exception as e:
            logger.error(f"Failed to download {dataset}: {e}")
            raise


@cli.command()
@click.option("--output", type=click.Path(), default="configs", help="Output config directory")
@click.option("--preset", type=click.Choice(["standard", "dsa", "moe", "dsa_moe", "paper_base"]), default="paper_base")
@click.pass_context
def config(ctx, output, preset):
    """Generate default configuration files."""
    verbose = ctx.obj.get("verbose", False)

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate model config
    model_cfg = get_preset_config(preset)
    model_dict = {k: v for k, v in model_cfg.__dict__.items() if not k.startswith("_")}

    # Convert enums to strings
    for key in ["attention_type", "ffn_type", "task"]:
        if key in model_dict and hasattr(model_dict[key], "value"):
            model_dict[key] = model_dict[key].value

    with open(output_path / "model.yaml", "w") as f:
        yaml.dump({"model": model_dict}, f, default_flow_style=False)

    # Generate data config
    data_cfg = DataConfig(name="default", source=DataSource.LOCAL)
    data_dict = {k: v for k, v in data_cfg.__dict__.items() if not k.startswith("_")}
    if "source" in data_dict and hasattr(data_dict["source"], "value"):
        data_dict["source"] = data_dict["source"].value

    with open(output_path / "data.yaml", "w") as f:
        yaml.dump({"data": data_dict}, f, default_flow_style=False)

    # Generate train config
    train_cfg = TrainConfig()
    train_dict = {k: v for k, v in train_cfg.__dict__.items() if not k.startswith("_")}
    for key in ["optimizer", "scheduler", "precision", "distributed"]:
        if key in train_dict and hasattr(train_dict[key], "value"):
            train_dict[key] = train_dict[key].value

    with open(output_path / "train.yaml", "w") as f:
        yaml.dump({"train": train_dict}, f, default_flow_style=False)

    # Generate combined config
    with open(output_path / "config.yaml", "w") as f:
        yaml.dump({
            "model": model_dict,
            "data": data_dict,
            "train": train_dict,
        }, f, default_flow_style=False)

    logger.info(f"Config files generated in {output_path}")


@cli.command()
@click.option("--model", required=True, type=click.Path(exists=True), help="Path to model checkpoint")
@click.option("--output", type=click.Path(), help="Output path for HF model")
@click.option("--repo", help="HF Hub repo to push to")
@click.option("--private", is_flag=True, help="Make repo private")
@click.pass_context
def export(ctx, model, output, repo, private):
    """Export local checkpoint to HF format."""
    verbose = ctx.obj.get("verbose", False)

    checkpoint = torch.load(model, map_location="cpu")
    model_cfg = HRMConfig(**checkpoint.get("model_config", {}))
    hf_config = UnifiedCVHRMConfig.from_local_config(model_cfg)
    hf_model = UnifiedCVHRM(hf_config)
    hf_model.backbone.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)

    if output:
        hf_model.save_pretrained(output)
        logger.info(f"Model saved to {output}")

    if repo:
        hf_model.push_to_hub(repo, private=private)
        logger.info(f"Model pushed to {repo}")

    if not output and not repo:
        logger.warning("Specify --output or --repo to save/push model")


@cli.command()
@click.option("--model", required=True, help="Local model path or HF repo")
@click.option("--output", required=True, type=click.Path(), help="ONNX output path")
@click.option("--opset", type=int, default=17, help="ONNX opset version")
@click.option("--dynamic-batch", is_flag=True, help="Dynamic batch size")
@click.pass_context
def export_onnx(ctx, model, output, opset, dynamic_batch):
    """Export model to ONNX format."""
    verbose = ctx.obj.get("verbose", False)

    # Load model
    if os.path.exists(model):
        checkpoint = torch.load(model, map_location="cpu")
        model_cfg = HRMConfig(**checkpoint.get("model_config", {}))
        hf_config = UnifiedCVHRMConfig.from_local_config(model_cfg)
        hf_model = UnifiedCVHRM(hf_config)
        hf_model.backbone.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
    else:
        hf_config = UnifiedCVHRMConfig.from_pretrained(model)
        hf_model = UnifiedCVHRM.from_pretrained(model, config=hf_config)

    hf_model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, hf_config.in_channels, hf_config.image_size, hf_config.image_size)

    # Export
    dynamic_axes = {}
    if dynamic_batch:
        dynamic_axes = {"pixel_values": {0: "batch"}, "logits": {0: "batch"}}

    torch.onnx.export(
        hf_model,
        dummy_input,
        output,
        opset_version=opset,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    logger.info(f"Model exported to ONNX: {output}")


@cli.command()
@click.pass_context
def paper_experiments(ctx):
    """Run all paper benchmark experiments."""
    verbose = ctx.obj.get("verbose", False)

    # Paper benchmark datasets
    paper_datasets = [
        "cifar10", "cifar100", "svhn", "mnist", "fashion_mnist",
        "stl10", "tinyimagenet200", "imagenet1k",
        "chestxray", "brain_tumor", "ham10000",
        "intel_scenes", "coco", "wlasl",
    ]

    logger.info("Running paper benchmark experiments...")
    logger.info(f"Datasets: {paper_datasets}")

    # This would run the full benchmark suite
    # For now, just log the command that would be run
    cmd = "ihrm benchmark --model paper_base --datasets " + " ".join(paper_datasets)
    logger.info(f"Run: {cmd}")


@cli.command()
@click.option("--dataset", multiple=True, default=["cifar10", "cifar100", "imagenet1k"], help="Datasets to train")
@click.option("--preset", default="standard", help="Model preset")
@click.pass_context
def modal_train(ctx, dataset, preset):
    """Launch training jobs on Modal.com."""
    logger.info("Modal training requested")
    logger.info(f"Datasets: {dataset}, preset: {preset}")
    logger.info("Run: modal run modal/train.py")
    # Placeholder: would trigger Modal deployment
    click.echo("Modal training would start with datasets: " + ", ".join(dataset))


@cli.command()
@click.pass_context
def modal_setup(ctx):
    """Initialize Modal environment and secrets."""
    logger.info("Setting up Modal environment")
    click.echo("Modal setup: ensure modal token is configured and run 'modal deploy modal/setup.py'")


@cli.command()
@click.option("--job-id", help="Modal job ID to stream logs")
@click.pass_context
def modal_logs(ctx, job_id):
    """Stream logs from Modal training jobs."""
    logger.info(f"Streaming logs for job {job_id or 'all'}")
    click.echo(f"Modal logs: modal logs {job_id or ''}")


@cli.command()
@click.pass_context
def version(ctx):
    """Show version info."""
    click.echo("I-HRM: Image Hierarchical Reasoning Model")
    click.echo("Version: 1.0.0")
    click.echo("Paper: I-HRM: Image Hierarchical Reasoning Model")


if __name__ == "__main__":
    cli()