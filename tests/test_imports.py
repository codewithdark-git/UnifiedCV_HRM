"""
Test Suite for I-HRM

Run with: pytest tests/ -v
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestImports:
    """Test that all modules import correctly."""

    def test_config_imports(self):
        from src.config import HRMConfig, DataConfig, TrainConfig
        from src.config.loader import ConfigManager, load_config, build_configs

    def test_model_imports(self):
        from src.models import (
            HRMStream,
            AdaptiveHaltingModule,
            HierarchicalVisionTransformer,
            TransformerBlock,
            RMSNorm,
            SwiGLU,
            DenseRoutedMoE,
            DeltaNetFFN,
            MultiHeadAttention,
            DynamicSparseAttention,
            DeltaAttention,
            ImagePatchEmbed,
            VideoTokenizer3D,
            VideoPatchEmbed,
            create_tokenizer,
            ClassificationHead,
            SegmentationHead,
            DetectionHead,
            MultiTaskHead,
            UnifiedCVHRM,
            UnifiedCVHRMConfig,
            AttentionType,
            FFNType,
            TaskType,
        )

    def test_datasets_imports(self):
        from src.data import (
            UnifiedDataset,
            TVDatasetWrapper,
            LocalDataset,
            MedicalLocalDataset,
            MultiTaskLocalDataset,
            HFDataset,
            HFMultiTaskDataset,
            get_dataset,
            get_dataloader,
            build_dataset_from_registry,
            create_local_dataset,
            create_hf_dataset,
            create_medical_dataset,
            build_train_transform,
            build_val_transform,
        )

    def test_training_imports(self):
        from src.training import (
            Trainer,
            TrainerConfig,
            MultiTaskLoss,
            ClassificationLoss,
            SegmentationLoss,
            DetectionLoss,
            HaltingLoss,
            get_loss_fn,
            Accuracy,
            TopKAccuracy,
            IoU,
            DiceScore,
            COCOmAP,
            get_metrics,
            Callback,
            ModelCheckpoint,
            EarlyStopping,
            LearningRateMonitor,
            WandbCallback,
            TensorBoardCallback,
            create_optimizer,
            create_scheduler,
        )

    def test_evaluation_imports(self):
        from src.evaluation import (
            Evaluator,
            BenchmarkSuite,
            run_paper_benchmarks,
        )

    def test_utils_imports(self):
        from src.utils import (
            setup_logging,
            get_logger,
            ProgressLogger,
            is_main_process,
            get_rank,
            get_world_size,
            reduce_dict,
            all_gather,
            AverageMeter,
            MetricTracker,
            compute_flops,
            count_parameters,
            save_checkpoint,
            load_checkpoint,
            save_model,
            load_model,
            strip_ddp,
            load_partial,
            convert_to_hf_checkpoint,
            find_latest_checkpoint,
        )


class TestConfig:
    """Test configuration classes."""

    def test_hrm_config_default(self):
        from src.config.model_config import HRMConfig
        cfg = HRMConfig()
        assert cfg.hidden_size == 384
        assert cfg.num_heads == 8
        assert cfg.num_layers_per_stream == 2

    def test_hrm_config_presets(self):
        from src.config.model_config import HRMConfig
        for name in ["standard", "dsa", "moe", "dsa_moe", "paper_base"]:
            cfg = getattr(HRMConfig, name)()
            assert isinstance(cfg, HRMConfig)

    def test_data_config(self):
        from src.config.data_config import DataConfig, DataSource
        cfg = DataConfig(name="test", source=DataSource.LOCAL, num_classes=10)
        assert cfg.num_classes == 10

    def test_train_config(self):
        from src.config.train_config import TrainConfig
        cfg = TrainConfig(epochs=10, batch_size=32)
        assert cfg.epochs == 10


class TestModel:
    """Test model building and forward pass."""

    def test_hrm_config_to_local(self):
        from src.models.unified_hrm import UnifiedCVHRMConfig
        from src.config.model_config import HRMConfig

        hf_cfg = UnifiedCVHRMConfig(hidden_size=256, num_classes=100)
        local_cfg = hf_cfg.to_local_config()
        assert isinstance(local_cfg, HRMConfig)
        assert local_cfg.hidden_size == 256

    def test_hierarchical_vit_forward(self):
        from src.models.hrm import HierarchicalVisionTransformer, HRMConfig

        cfg = HRMConfig(
            hidden_size=128,
            num_heads=4,
            num_layers_per_stream=1,
            max_steps=2,
            image_size=32,
            patch_size=4,
            num_classes=10,
        )
        model = HierarchicalVisionTransformer(cfg)

        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out, steps = model(x, return_steps=True, return_logits=True)

        assert out.shape == (2, 10)  # (B, num_classes)
        assert "halt_probs" in steps
        assert "halt_steps" in steps

    def test_unified_cvhrm_forward(self):
        from src.models.unified_hrm import UnifiedCVHRM, UnifiedCVHRMConfig

        cfg = UnifiedCVHRMConfig(
            hidden_size=128,
            num_heads=4,
            num_layers_per_stream=1,
            max_steps=2,
            image_size=32,
            patch_size=4,
            num_classes=10,
        )
        model = UnifiedCVHRM(cfg)

        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(x, task="classification")

        assert hasattr(out, "logits")
        assert out.logits.shape == (2, 10)

    def test_generate(self):
        from src.models.unified_hrm import UnifiedCVHRM, UnifiedCVHRMConfig

        cfg = UnifiedCVHRMConfig(
            hidden_size=128,
            num_heads=4,
            num_layers_per_stream=1,
            max_steps=2,
            image_size=32,
            patch_size=4,
            num_classes=10,
        )
        model = UnifiedCVHRM(cfg)

        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            pred = model.generate(x, task="classification", top_k=3)

        assert "labels" in pred
        assert "scores" in pred
        assert len(pred["labels"][0]) == 3


class TestDataConfig:
    """Test data configuration and registry."""

    def test_dataset_registry(self):
        from src.config.data_config import DatasetRegistry

        # Check all paper datasets registered
        for ds in DatasetRegistry.list():
            cfg = DatasetRegistry.get(ds)
            assert cfg is not None
            assert cfg.name == ds

    def test_preset_configs(self):
        from src.config.data_config import DatasetRegistry
        cfg = DatasetRegistry.get("cifar10")
        assert cfg is not None
        assert cfg.num_classes == 10
        assert cfg.source.value == "torchvision"


class TestTraining:
    """Test training components."""

    def test_loss_functions(self):
        from src.training.losses import (
            ClassificationLoss,
            SegmentationLoss,
            DetectionLoss,
            HaltingLoss,
            MultiTaskLoss,
        )

        cls_loss = ClassificationLoss(label_smoothing=0.1)
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss = cls_loss(logits, targets)
        assert loss.item() >= 0

    def test_scheduler_creation(self):
        from src.training.optim import create_scheduler
        from src.training.optim import create_optimizer

        model = torch.nn.Linear(10, 10)
        opt = create_optimizer(model, "adamw", lr=1e-3)
        sched = create_scheduler(opt, "cosine_warmup", num_warmup_steps=10, num_training_steps=100)

        assert opt is not None
        assert sched is not None


class TestUtils:
    """Test utility functions."""

    def test_count_parameters(self):
        from src.utils.metrics import count_parameters
        model = torch.nn.Linear(100, 100)
        stats = count_parameters(model)
        assert stats["total"] == 10100

    def test_compute_flops(self):
        from src.utils.metrics import compute_flops
        model = torch.nn.Linear(10, 10)
        flops = compute_flops(model, (1, 10), unit="M")
        assert "flops" in flops


# Run with: python -m pytest tests/ -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])