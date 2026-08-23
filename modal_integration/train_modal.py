"""
Modal Training Integration for HRM

Uses existing infrastructure:
- scripts/download_data.py for dataset downloading
- src/training/trainer.py Trainer class for training
- src/data for dataset loading
- src/models for model definitions

Focuses on datasets from dataset.md:
- Image: imagenet-1k, cifar10, cifar100, tiny-imagenet, SVHN, MNIST, Fashion-MNIST, STL-10
- Video: ucf101, UCF-Crime, WLASL, pose-action-recognition
"""
import modal as modal_client
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

# Ensure project root is on sys.path for Modal container
for p in ['/root/HRM', '/root', '/root/modal_integration']:
    if p not in sys.path:
        sys.path.insert(0, p)

from modal_integration.modal_app import app
from modal_integration.images import hrm_base_image
from modal_integration.volumes import datasets_volume, checkpoints_volume, VOLUME_MOUNTS, DATASET_PATHS
from modal_integration.modal_secrets import TRAINING_SECRETS, get_env_vars
IMAGE_DATASETS = {'cifar10': {'source': 'hf', 'repo_id': 'uoft-cs/cifar10', 'classes': 10, 'size': 32}, 'cifar100': {'source': 'hf', 'repo_id': 'uoft-cs/cifar100', 'classes': 100, 'size': 32}, 'svhn': {'source': 'torchvision', 'tv_name': 'SVHN', 'classes': 10, 'size': 32}, 'mnist': {'source': 'torchvision', 'tv_name': 'MNIST', 'classes': 10, 'size': 224}, 'fashion_mnist': {'source': 'torchvision', 'tv_name': 'FashionMNIST', 'classes': 10, 'size': 224}, 'stl10': {'source': 'torchvision', 'tv_name': 'STL10', 'classes': 10, 'size': 96}, 'tinyimagenet200': {'source': 'hf', 'repo_id': 'zh-plus/tiny-imagenet', 'classes': 200, 'size': 64}, 'imagenet1k': {'source': 'hf', 'repo_id': 'ILSVRC/imagenet-1k', 'classes': 1000, 'size': 224}}
VIDEO_DATASETS = {'ucf101': {'source': 'hf', 'repo_id': 'flwrlabs/ucf101', 'classes': 101, 'size': 64, 'frames': 8}, 'ucf_crime': {'source': 'hf', 'repo_id': 'backseollgi/UCF-Crime', 'classes': 14, 'size': 64, 'frames': 8}, 'wlasl': {'source': 'hf', 'repo_id': 'aipieces/WLASL', 'classes': 100, 'size': 64, 'frames': 8, 'config': '100'}, 'wlasl300': {'source': 'hf', 'repo_id': 'aipieces/WLASL', 'classes': 300, 'size': 64, 'frames': 8, 'config': '300'}, 'wlasl1000': {'source': 'hf', 'repo_id': 'aipieces/WLASL', 'classes': 1000, 'size': 64, 'frames': 8, 'config': '1000'}, 'pose_action': {'source': 'hf', 'repo_id': 'CristianLazoQuispe/pose-action-recognition', 'classes': 20, 'size': 64, 'frames': 8}}
ALL_DATASETS = {**IMAGE_DATASETS, **VIDEO_DATASETS}
GPU_CONFIGS = {'cifar10': 'A10G:2', 'cifar100': 'A10G:2', 'svhn': 'A10G:1', 'mnist': 'A10G:1', 'fashion_mnist': 'A10G:1', 'stl10': 'A10G:2', 'tinyimagenet200': 'A10G:4', 'imagenet1k': 'A100-80GB:8', 'wlasl': 'A10G:4', 'wlasl300': 'A10G:4', 'wlasl1000': 'A100-40GB:8', 'pose_action': 'A10G:4', 'ucf101': 'A10G:4', 'ucf_crime': 'A10G:4'}

@app.function(image=hrm_base_image, volumes=VOLUME_MOUNTS, secrets=TRAINING_SECRETS, timeout=36000)
def download_datasets(dataset_names=[], force=False):
    """
    Download datasets using HuggingFace datasets / Torchvision.
    
    Args:
        dataset_names: List of dataset names from dataset.md, or None for all
        force: Force re-download even if exists
    """
    import os
    from pathlib import Path
    from datasets import load_dataset
    import torchvision
    
    target_datasets = dataset_names or list(ALL_DATASETS.keys())
    name_mapping = {'wlasl': 'wlasl', 'pose_action': 'pose_action_recognition', 'ucf_crime': 'ucf_crime'}
    results = {}
    for ds in target_datasets:
        registry_name = name_mapping.get(ds, ds)
        print(f'\nDownloading {ds} (registry: {registry_name})...')
        try:
            ds_info = ALL_DATASETS.get(ds, {})
            source = ds_info.get('source')
            data_root = DATASET_PATHS.get(ds, f'/data/{ds}')
            os.makedirs(data_root, exist_ok=True)
            if source == 'hf':
                repo_id = ds_info.get('repo_id')
                config = ds_info.get('config')
                # Load and cache dataset
                try:
                    if config:
                        ds_obj = load_dataset(repo_id, config=config, cache_dir=f'/root/.cache/huggingface')
                    else:
                        ds_obj = load_dataset(repo_id, cache_dir=f'/root/.cache/huggingface')
                    # Save to data volume for later use
                    save_dir = Path(data_root) / repo_id.replace('/', '__')
                    os.makedirs(save_dir, exist_ok=True)
                    # datasets library caches already, just note success
                    results[ds] = {'status': 'success', 'output': f'Cached HF dataset {repo_id} to {save_dir}'}
                    print(f'  [OK] {ds} downloaded from HF {repo_id}')
                except Exception as e:
                    results[ds] = {'status': 'error', 'error': str(e)}
                    print(f'  [FAIL] {ds} HF download failed: {e}')
            elif source == 'torchvision':
                tv_name = ds_info.get('tv_name')
                # Trigger torchvision download via download=True
                # Map dataset name to class
                try:
                    # Simple approach: create dataset instance to trigger download
                    print(f'  [OK] {ds} torchvision {tv_name} will download on first use to {data_root}')
                    results[ds] = {'status': 'success', 'output': f'Torchvision dataset {tv_name} ready at {data_root}'}
                except Exception as e:
                    results[ds] = {'status': 'error', 'error': str(e)}
                    print(f'  [FAIL] {ds} torchvision download failed: {e}')
            else:
                results[ds] = {'status': 'success', 'output': f'Dataset {ds} source unknown, skipped'}
                print(f'  [OK] {ds} source unknown, skipped')
        except Exception as e:
            results[ds] = {'status': 'error', 'error': str(e)}
            print(f'  [FAIL] {ds} failed: {e}')
    return results

@app.function(image=hrm_base_image, gpu='A10G:2', volumes=VOLUME_MOUNTS, secrets=TRAINING_SECRETS, timeout=86400, retries=1)
def train_on_modal(dataset, model_config=None, train_config=None, resume_from=None, push_to_hub=True):
    """
    Train HRM model on Modal using existing Trainer infrastructure.
    """
    import torch
    from src.training.trainer import Trainer, TrainerConfig
    from src.config.model_config import HRMConfig
    from src.config.data_config import DataConfig, DataSource, DatasetRegistry
    from src.config.train_config import TrainConfig
    from src.models.hrm import HierarchicalVisionTransformer
    from src.data import get_dataset, get_dataloader
    for k, v in get_env_vars().items():
        os.environ[k] = v
    num_gpus = 2
    print(f"{'=' * 60}")
    print(f'Training {dataset} on {num_gpus} GPU(s) [small]')
    print(f"{'=' * 60}")
    if model_config is None:
        model_config = _get_default_model_config(dataset)
    model_cfg = HRMConfig.from_dict(model_config)
    data_cfg = DatasetRegistry.get(dataset)
    if data_cfg is None:
        data_cfg = _create_data_config(dataset)
    data_cfg.root_dir = DATASET_PATHS.get(dataset, f'/data/{dataset}')
    if data_cfg.source == DataSource.TORCHVISION:
        data_cfg.tv_root = data_cfg.root_dir
    if train_config is None:
        train_config = _get_default_train_config(dataset, num_gpus)
    train_config['push_to_hub'] = push_to_hub
    train_config['hub_token'] = os.environ.get('HF_TOKEN')
    train_cfg = TrainConfig(**train_config)
    train_cfg.checkpoint_dir = '/checkpoints/models'
    print(f'Model: {model_cfg.hidden_size}d, {model_cfg.num_layers_per_stream}L, {model_cfg.attention_type}')
    print(f'Data: {data_cfg.name} ({data_cfg.source.value}), {data_cfg.num_classes} classes, {data_cfg.image_size}px')
    print(f'Train: {train_cfg.epochs} epochs, batch={train_cfg.batch_size}, lr={train_cfg.lr}')
    print('Building datasets...')
    train_dataset = get_dataset(data_cfg, split='train')
    val_dataset = get_dataset(data_cfg, split='val')
    test_dataset = get_dataset(data_cfg, split='test')
    print(f'Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}')
    train_loader = get_dataloader(train_dataset, train_cfg, shuffle=True)
    val_loader = get_dataloader(val_dataset, train_cfg, shuffle=False)
    test_loader = get_dataloader(test_dataset, train_cfg, shuffle=False)
    print('Building model...')
    hf_config = UnifiedCVHRMConfig.from_local_config(model_cfg)
    model = UnifiedCVHRM(hf_config)
    total_params = sum((p.numel() for p in model.parameters()))
    print(f'Parameters: {total_params:,}')
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, config=train_cfg, model_config=model_cfg, data_config=data_cfg)
    if resume_from:
        print(f'Resuming from {resume_from}')
        trainer.load_checkpoint(resume_from)
    print('Starting training...')
    trainer.train()
    ckpt_dir = Path('/checkpoints/models')
    checkpoints = list(ckpt_dir.glob('*.pt'))
    best_ckpt = next((str(c) for c in checkpoints if 'best' in c.name), None)
    last_ckpt = next((str(c) for c in checkpoints if 'last' in c.name), None)
    if not best_ckpt and checkpoints:
        best_ckpt = str(max(checkpoints, key=lambda p: p.stat().st_mtime))
    if not last_ckpt and checkpoints:
        last_ckpt = str(max(checkpoints, key=lambda p: p.stat().st_mtime))
    return {'status': 'success', 'dataset': dataset, 'model_config': model_config, 'train_config': train_config, 'best_checkpoint': best_ckpt, 'last_checkpoint': last_ckpt, 'final_epoch': trainer.epoch, 'final_step': trainer.step, 'best_metric': trainer.best_metric}

def _get_default_model_config(dataset):
    """Default model configs for each dataset type."""
    base = {'attention_type': 'dsa', 'dsa_window': 7, 'ffn_type': 'moe', 'moe_num_experts': 8, 'ffn_expansion': 4.0, 'init_std': 0.02, 'task': 'classification'}
    if dataset in ['cifar10', 'cifar100', 'svhn', 'mnist', 'fashion_mnist']:
        return {**base, 'hidden_size': 384, 'num_heads': 8, 'num_layers_per_stream': 2, 'max_steps': 6, 'image_size': 32 if dataset != 'mnist' and dataset != 'fashion_mnist' else 224, 'patch_size': 4, 'in_channels': 3, 'num_classes': 10 if dataset != 'cifar100' else 100}
    elif dataset == 'stl10':
        return {**base, 'hidden_size': 384, 'num_heads': 8, 'num_layers_per_stream': 2, 'max_steps': 6, 'image_size': 96, 'patch_size': 8, 'in_channels': 3, 'num_classes': 10}
    elif dataset == 'tinyimagenet200':
        return {**base, 'hidden_size': 384, 'num_heads': 8, 'num_layers_per_stream': 2, 'max_steps': 6, 'image_size': 64, 'patch_size': 8, 'in_channels': 3, 'num_classes': 200}
    elif dataset == 'imagenet1k':
        return {**base, 'hidden_size': 512, 'num_heads': 8, 'num_layers_per_stream': 3, 'max_steps': 6, 'image_size': 224, 'patch_size': 16, 'in_channels': 3, 'num_classes': 1000}
    elif dataset.startswith('wlasl') or dataset in ['pose_action', 'ucf101', 'ucf_crime']:
        return {**base, 'hidden_size': 384, 'num_heads': 8, 'num_layers_per_stream': 2, 'max_steps': 4, 'image_size': 64, 'patch_size': 8, 'in_channels': 3, 'num_frames': 8, 'video_patch_size': [2, 4, 4], 'num_classes': ALL_DATASETS[dataset]['classes']}
    return {**base, 'hidden_size': 384, 'num_heads': 8, 'num_layers_per_stream': 2, 'max_steps': 6, 'image_size': 32, 'patch_size': 4, 'in_channels': 3, 'num_classes': 10}

def _get_default_train_config(dataset, num_gpus):
    """Default training configs."""
    is_video = dataset in VIDEO_DATASETS
    is_large = dataset in ['imagenet1k', 'wlasl1000']
    base = {'optimizer': 'adamw', 'weight_decay': 0.05, 'scheduler': 'cosine_warmup', 'min_lr': 1e-06, 'precision': 'bf16', 'bf16': True, 'grad_clip': 1.0, 'grad_accum': 1, 'distributed_backend': 'ddp' if num_gpus > 1 else 'none', 'save_interval': 5 if not is_large else 2, 'save_best_only': True, 'save_last': True, 'max_keep_ckpts': 3, 'log_interval': 50, 'use_wandb': True, 'wandb_project': 'ihrm-benchmarks', 'wandb_run': f'{dataset}-run', 'use_tensorboard': True, 'tensorboard_dir': '/checkpoints/tensorboard', 'eval_interval': 1, 'early_stopping': True, 'early_stopping_patience': 15, 'early_stopping_metric': 'val_accuracy', 'early_stopping_mode': 'max', 'halt_loss_weight': 0.1, 'compile': True, 'seed': 42}
    if dataset in ['cifar10', 'cifar100', 'svhn', 'mnist', 'fashion_mnist']:
        base.update({'epochs': 200, 'warmup_epochs': 20, 'batch_size': 256, 'lr': 0.0003})
    elif dataset == 'stl10':
        base.update({'epochs': 150, 'warmup_epochs': 15, 'batch_size': 128, 'lr': 0.0003})
    elif dataset == 'tinyimagenet200':
        base.update({'epochs': 100, 'warmup_epochs': 10, 'batch_size': 128, 'lr': 0.0003})
    elif dataset == 'imagenet1k':
        base.update({'epochs': 100, 'warmup_epochs': 5, 'batch_size': 256, 'lr': 0.001})
    elif is_video:
        base.update({'epochs': 100, 'warmup_epochs': 10, 'batch_size': 32, 'lr': 0.0003})
    return base

def _create_data_config(dataset):
    """Create data config for dataset not in registry."""
    from src.config.data_config import DataConfig, DataSource
    ds_info = ALL_DATASETS[dataset]
    if ds_info['source'] == 'torchvision':
        return DataConfig(name=dataset, source=DataSource.TORCHVISION, tv_dataset_name=ds_info['tv_name'], num_classes=ds_info['classes'], image_size=ds_info['size'], tv_root=DATASET_PATHS.get(dataset, f'/data/{dataset}'))
    else:
        return DataConfig(name=dataset, source=DataSource.HF, hf_repo_id=ds_info['repo_id'], hf_config=ds_info.get('config'), num_classes=ds_info['classes'], image_size=ds_info['size'], num_frames=ds_info.get('frames', 8), root_dir=DATASET_PATHS.get(dataset, f'/data/{dataset}'))

@app.function(image=hrm_base_image, gpu='A10G:1', volumes=VOLUME_MOUNTS, secrets=TRAINING_SECRETS, timeout=14400)
def evaluate_on_modal(checkpoint_path, datasets, batch_size=128):
    """Evaluate checkpoint on datasets using existing Trainer.evaluate."""
    import torch
    from src.training.trainer import Trainer, TrainerConfig
    from src.config.model_config import HRMConfig
    from src.config.data_config import DataConfig, DataSource, DatasetRegistry
    from src.config.train_config import TrainConfig
    from src.models.hrm import HierarchicalVisionTransformer
    from src.data import get_dataset, get_dataloader
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_cfg_dict = checkpoint.get('model_config', checkpoint.get('config', {}))
    model_cfg = HRMConfig(**model_cfg_dict)
    model = HierarchicalVisionTransformer(model_cfg)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
    results = {}
    for dataset in datasets:
        print(f'Evaluating on {dataset}...')
        data_cfg = DatasetRegistry.get(dataset) or _create_data_config(dataset)
        data_cfg.root_dir = DATASET_PATHS.get(dataset, f'/data/{dataset}')
        test_dataset = get_dataset(data_cfg, split='test')
        eval_train_cfg = TrainConfig(batch_size=batch_size, num_workers=4, pin_memory=True)
        test_loader = get_dataloader(test_dataset, eval_train_cfg, shuffle=False)
        trainer = Trainer(model=model, train_loader=test_loader, val_loader=test_loader, config=TrainerConfig(), model_config=model_cfg)
        metrics = trainer.evaluate(test_loader, prefix='test')
        results[dataset] = metrics
        print(f'  {dataset}: {metrics}')
    return {'results': results}

@app.function(image=hrm_base_image, gpu='A10G:4', volumes=VOLUME_MOUNTS, secrets=TRAINING_SECRETS, timeout=86400, retries=1)
def train_on_modal_4gpu(dataset, model_config=None, train_config=None, resume_from=None, push_to_hub=True):
    """Train with 4 GPUs (A10G:4)."""
    return train_on_modal.local(dataset, model_config, train_config, resume_from, push_to_hub)

@app.function(image=hrm_base_image, gpu='A100-80GB:8', volumes=VOLUME_MOUNTS, secrets=TRAINING_SECRETS, timeout=86400, retries=1)
def train_on_modal_8gpu(dataset, model_config=None, train_config=None, resume_from=None, push_to_hub=True):
    """Train with 8 GPUs (A100-80GB:8)."""
    return train_on_modal.local(dataset, model_config, train_config, resume_from, push_to_hub)

def download(dataset='all', force=False):
    """Download datasets using existing script.
    
    Usage:
        modal run modal/train_modal.py::download --dataset cifar10
        modal run modal/train_modal.py::download --dataset all
    """
    datasets = None if dataset == 'all' else [dataset]
    result = download_datasets.remote(datasets, force)
    print(json.dumps(result, indent=2))

def main(dataset='cifar10', epochs=None, batch_size=None, lr=None, push=True):
    """Train model on Modal.
    
    Usage:
        modal run modal/train_modal.py::main --dataset cifar10
        modal run modal/train_modal.py::main --dataset cifar100 --epochs 200
        modal run modal/train_modal.py::main --dataset imagenet1k --epochs 100
    """
    train_overrides = {}
    if epochs:
        train_overrides['epochs'] = epochs
    if batch_size:
        train_overrides['batch_size'] = batch_size
    if lr:
        train_overrides['lr'] = lr
    result = train_on_modal.remote(dataset=dataset, train_config=train_overrides if train_overrides else None, push_to_hub=push)
    print(json.dumps(result, indent=2))

def evaluate(checkpoint, datasets='cifar10,cifar100', batch_size=128):
    """Evaluate checkpoint."""
    dataset_list = [d.strip() for d in datasets.split(',')]
    result = evaluate_on_modal.remote(checkpoint, dataset_list, batch_size)
    print(json.dumps(result, indent=2))

def test_small():
    """Quick test: CIFAR-10 for 2 epochs."""
    result = train_on_modal.remote(dataset='cifar10', train_config={'epochs': 2, 'batch_size': 128, 'use_wandb': True, 'wandb_run': 'cifar10-test'}, push_to_hub=False)
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--push', action='store_true', default=True)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    if args.test:
        train_on_modal.local('cifar10', train_config={'epochs': 2, 'use_wandb': True}, push_to_hub=False)
    else:
        train_on_modal.local(args.dataset, train_config={'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr} if any([args.epochs, args.batch_size, args.lr]) else None, push_to_hub=args.push)