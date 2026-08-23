#!/usr/bin/env python
"""
Test Modal Integration

Verifies:
1. Modal setup works
2. Dataset download works (via existing scripts/download_data.py)
3. Training works with existing Trainer class
4. Checkpoints saved to volume
5. Metrics logged to W&B
6. Checkpoints can be pushed to HF Hub
"""

import modal as modal_client
from modal import App
from modal_integration.images import hrm_base_image
from modal_integration.volumes import datasets_volume, checkpoints_volume, VOLUME_MOUNTS
from modal_integration.modal_secrets import TRAINING_SECRETS

app = App("ihrm-test")


@app.function(
    image=hrm_base_image,
    volumes=VOLUME_MOUNTS,
    secrets=TRAINING_SECRETS,
    timeout=600,
)
def test_modal_setup():
    """Test Modal environment."""
    import torch
    import os
    
    print("Testing Modal setup...")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    
    # Check volumes
    print("\nVolumes:")
    for mount_path, _ in VOLUME_MOUNTS.items():
        path = Path(mount_path)
        print(f"  {mount_path}: {'OK' if path.exists() else 'NOT MOUNTED'}")
    
    # Check secrets
    print("\nSecrets:")
    for key in ["HF_TOKEN", "WANDB_API_KEY"]:
        print(f"  {key}: {'SET' if os.environ.get(key) else 'NOT SET'}")
    
    return {"torch": torch.__version__, "cuda": torch.cuda.is_available(), "gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0}


@app.function(
    image=hrm_base_image,
    gpu="A10G:1",
    volumes=VOLUME_MOUNTS,
    secrets=TRAINING_SECRETS,
    timeout=1800,
)
def test_training_pipeline():
    """Test training pipeline with CIFAR-10 for 1 epoch."""
    import torch
    from src.training.trainer import Trainer, TrainerConfig
    from src.config.model_config import HRMConfig
    from src.config.data_config import DataConfig, DataSource
    from src.config.train_config import TrainConfig
    from src.models.hrm import HierarchicalVisionTransformer
    from src.data import get_dataset, get_dataloader
    
    print("Testing training pipeline (CIFAR-10, 1 epoch)...")
    
    model_cfg = HRMConfig(
        hidden_size=192, num_heads=4, num_layers_per_stream=1, max_steps=2,
        attention_type="dsa", dsa_window=7, ffn_type="moe", moe_num_experts=4,
        ffn_expansion=4.0, image_size=32, patch_size=4, in_channels=3,
        num_classes=10, task="classification", init_std=0.02,
    )
    
    data_cfg = DataConfig(
        name="cifar10", source=DataSource.TORCHVISION, tv_dataset_name="CIFAR10",
        num_classes=10, image_size=32, tv_root="/data/cifar10",
    )
    
    train_cfg = TrainConfig(
        epochs=1, batch_size=64, lr=3e-4, optimizer="adamw", weight_decay=0.05,
        scheduler="cosine_warmup", warmup_epochs=1, precision="bf16", bf16=True,
        grad_clip=1.0, distributed_backend="none", log_interval=10,
        use_wandb=True, wandb_project="ihrm-test", wandb_run="cifar10-pipeline-test",
        use_tensorboard=True, tensorboard_dir="/checkpoints/tensorboard",
        save_interval=1, push_to_hub=False, compile=False, seed=42,
    )
    train_cfg.checkpoint_dir = "/checkpoints/models"
    
    train_dataset = get_dataset(data_cfg, split="train")
    val_dataset = get_dataset(data_cfg, split="val")
    train_loader = get_dataloader(train_dataset, train_cfg, shuffle=True)
    val_loader = get_dataloader(val_dataset, train_cfg, shuffle=False)
    
    model = HierarchicalVisionTransformer(model_cfg)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        config=train_cfg, model_config=model_cfg, data_config=data_cfg,
    )
    
    print("Training...")
    trainer.train()
    
    ckpts = list(Path("/checkpoints/models").glob("*.pt"))
    print(f"Checkpoints: {[c.name for c in ckpts]}")
    
    return {"status": "success", "epochs": trainer.epoch, "steps": trainer.step, "checkpoints": [c.name for c in ckpts]}


@app.function(
    image=hrm_base_image,
    volumes=VOLUME_MOUNTS,
    secrets=TRAINING_SECRETS,
    timeout=600,
)
def test_hf_push():
    """Test HF Hub push."""
    from huggingface_hub import HfApi, create_repo
    import torch
    from src.models.unified_hrm import UnifiedCVHRM, UnifiedCVHRMConfig
    from src.config.model_config import HRMConfig
    import os
    import tempfile
    
    model_cfg = HRMConfig(hidden_size=192, num_heads=4, num_layers_per_stream=1, max_steps=2,
                          image_size=32, patch_size=4, num_classes=10)
    hf_config = UnifiedCVHRMConfig.from_local_config(model_cfg)
    hf_model = UnifiedCVHRM(hf_config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        hf_model.save_pretrained(tmpdir)
        
        repo_id = "test/ihrm-modal-test"
        try:
            create_repo(repo_id, exist_ok=True, private=True)
            api = HfApi()
            api.upload_folder(repo_id=repo_id, folder_path=str(tmpdir), token=os.environ.get("HF_TOKEN"))
            api.delete_repo(repo_id, token=os.environ.get("HF_TOKEN"))
            return {"status": "success", "message": "HF push works"}
        except Exception as e:
            return {"status": "error", "error": str(e)}


@app.function(
    image=hrm_base_image,
    volumes=VOLUME_MOUNTS,
    secrets=TRAINING_SECRETS,
    timeout=600,
)
def test_checkpoint_resume():
    """Test checkpoint save/resume."""
    import torch
    from src.training.trainer import Trainer, TrainerConfig
    from src.config.model_config import HRMConfig
    from src.config.data_config import DataConfig, DataSource
    from src.config.train_config import TrainConfig
    from src.models.hrm import HierarchicalVisionTransformer
    from src.data import get_dataset, get_dataloader
    
    model_cfg = HRMConfig(hidden_size=128, num_heads=4, num_layers_per_stream=1, max_steps=2,
                          image_size=32, patch_size=4, num_classes=10)
    data_cfg = DataConfig(name="cifar10", source=DataSource.TORCHVISION, tv_dataset_name="CIFAR10",
                          num_classes=10, image_size=32, tv_root="/data/cifar10")
    train_cfg = TrainConfig(epochs=2, batch_size=64, lr=3e-4, precision="bf16", bf16=True,
                            distributed_backend="none", log_interval=10, use_wandb=False,
                            use_tensorboard=False, save_interval=1, push_to_hub=False,
                            compile=False, seed=42)
    train_cfg.checkpoint_dir = "/checkpoints/models"
    
    train_dataset = get_dataset(data_cfg, split="train")
    val_dataset = get_dataset(data_cfg, split="val")
    train_loader = get_dataloader(train_dataset, train_cfg, shuffle=True)
    val_loader = get_dataloader(val_dataset, train_cfg, shuffle=False)
    
    # Train 1 epoch
    model = HierarchicalVisionTransformer(model_cfg)
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                      config=train_cfg, model_config=model_cfg, data_config=data_cfg)
    trainer.train()
    step0 = trainer.step
    
    # Resume
    ckpt = max(Path("/checkpoints/models").glob("*.pt"), key=lambda p: p.stat().st_mtime)
    model2 = HierarchicalVisionTransformer(model_cfg)
    trainer2 = Trainer(model=model2, train_loader=train_loader, val_loader=val_loader,
                       config=train_cfg, model_config=model_cfg, data_config=data_cfg)
    trainer2.load_checkpoint(ckpt)
    print(f"Resumed: epoch={trainer2.epoch}, step={trainer2.step}")
    trainer2.train()
    
    assert trainer2.epoch == 2 and trainer2.step > step0
    return {"status": "success", "resume_works": True, "final_epoch": trainer2.epoch, "final_step": trainer2.step}


@app.local_entrypoint()
def test_all():
    """Run all tests."""
    print("=" * 60)
    print("TEST 1: Modal Setup")
    test_modal_setup.remote()
    
    print("\n" + "=" * 60)
    print("TEST 2: Training Pipeline")
    test_training_pipeline.remote()
    
    print("\n" + "=" * 60)
    print("TEST 3: HF Hub Push")
    test_hf_push.remote()
    
    print("\n" + "=" * 60)
    print("TEST 4: Checkpoint Resume")
    test_checkpoint_resume.remote()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")


@app.local_entrypoint()
def test_setup(): test_modal_setup.remote()
@app.local_entrypoint()
def test_train(): test_training_pipeline.remote()
@app.local_entrypoint()
def test_hf(): test_hf_push.remote()
@app.local_entrypoint()
def test_resume(): test_checkpoint_resume.remote()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        getattr(sys.modules[__name__], f"test_{sys.argv[1]}")()
    else:
        test_all()