"""Data Configuration

Unified dataset configuration supporting:
- Local datasets (folder-based)
- Hugging Face datasets
- TorchVision built-in datasets

Covers all 13 paper benchmarks (Table 3).
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
from pathlib import Path
import yaml


class DataSource(str, Enum):
    """Dataset source type"""
    LOCAL = "local"           # Folder-based (ImageFolder, custom)
    HF = "hf"                 # Hugging Face Hub
    TORCHVISION = "torchvision"  # TorchVision built-ins


class DatasetSplit(str, Enum):
    """Dataset splits"""
    TRAIN = "train"
    VAL = "val"
    VALIDATION = "validation"
    TEST = "test"


@dataclass
class DataConfig:
    """
    Unified dataset configuration for any source.

    All 13 Paper Benchmarks (Table 3) have predefined configs in configs/data/
    """
    # Identity
    name: str
    display_name: str = ""

    # Source
    source: DataSource = DataSource.LOCAL

    # Local dataset
    root_dir: Optional[str] = None
    train_subdir: str = "train"
    val_subdir: str = "val"
    test_subdir: str = "test"

    # HF dataset
    hf_repo_id: Optional[str] = None          # e.g., "ILSVRC/imagenet-1k"
    hf_config: Optional[str] = None           # e.g., "default"
    hf_split_mapping: Dict[str, str] = field(default_factory=lambda: {
        "train": "train",
        "val": "validation",
        "test": "test",
    })

    # TorchVision
    tv_dataset_name: Optional[str] = None     # e.g., "CIFAR100"
    tv_root: Optional[str] = None

    # Preprocessing
    image_size: int = 224
    image_size_val: Optional[int] = None      # Different val size
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Medical preprocessing (CLAHE)
    use_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)

    # Augmentation
    augment: bool = True
    aug_strength: str = "standard"            # "none" | "light" | "standard" | "strong"
    rand_augment: bool = False
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0

    # Dataset properties
    num_classes: int = 1000
    task: str = "classification"
    is_multi_label: bool = False
    class_names: Optional[List[str]] = None

    # Video (for video datasets)
    num_frames: int = 8
    frame_sample_rate: int = 1

    # Dataloader
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # Distributed
    drop_last: bool = True
    shuffle_train: bool = True

    def __post_init__(self):
        if self.image_size_val is None:
            self.image_size_val = self.image_size

        if isinstance(self.source, str):
            self.source = DataSource(self.source)

        if not self.display_name:
            self.display_name = self.name

        # Set default tv_root for torchvision datasets
        if self.source == DataSource.TORCHVISION and self.tv_root is None:
            self.tv_root = "./data"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["source"] = self.source.value
        # Convert tuples to lists for YAML
        data["mean"] = list(self.mean)
        data["std"] = list(self.std)
        data["clahe_tile_grid"] = list(self.clahe_tile_grid)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataConfig":
        if "source" in data and isinstance(data["source"], str):
            data["source"] = DataSource(data["source"])
        if "mean" in data and isinstance(data["mean"], list):
            data["mean"] = tuple(data["mean"])
        if "std" in data and isinstance(data["std"], list):
            data["std"] = tuple(data["std"])
        if "clahe_tile_grid" in data and isinstance(data["clahe_tile_grid"], list):
            data["clahe_tile_grid"] = tuple(data["clahe_tile_grid"])
        return cls(**data)

    def to_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "DataConfig":
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


class DatasetRegistry:
    """Registry of all 13 paper benchmark datasets"""

    _CONFIGS: Dict[str, DataConfig] = {}

    @classmethod
    def register(cls, name: str, config: DataConfig):
        cls._CONFIGS[name.lower()] = config

    @classmethod
    def get(cls, name: str) -> Optional[DataConfig]:
        return cls._CONFIGS.get(name.lower())

    @classmethod
    def list(cls) -> List[str]:
        return sorted(cls._CONFIGS.keys())

    @classmethod
    def load_all_from_dir(cls, config_dir: str):
        """Load all YAML configs from directory"""
        path = Path(config_dir)
        for yaml_file in path.glob("*.yaml"):
            name = yaml_file.stem
            cls._CONFIGS[name] = DataConfig.from_yaml(str(yaml_file))


# ─────────────────────────────────────────────────────────────────
# Paper Benchmark Presets (Table 3)
# ─────────────────────────────────────────────────────────────────

def create_paper_presets() -> Dict[str, DataConfig]:
    """Create configs for all 13 paper benchmark datasets"""

    presets = {}

    # 1. CIFAR-10
    presets["cifar10"] = DataConfig(
        name="cifar10",
        display_name="CIFAR-10",
        source=DataSource.TORCHVISION,
        tv_dataset_name="CIFAR10",
        num_classes=10,
        image_size=32,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
        batch_size=128,
    )

    # 2. CIFAR-100
    presets["cifar100"] = DataConfig(
        name="cifar100",
        display_name="CIFAR-100",
        source=DataSource.TORCHVISION,
        tv_dataset_name="CIFAR100",
        num_classes=100,
        image_size=32,
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
        batch_size=128,
    )

    # 3. SVHN
    presets["svhn"] = DataConfig(
        name="svhn",
        display_name="SVHN",
        source=DataSource.TORCHVISION,
        tv_dataset_name="SVHN",
        num_classes=10,
        image_size=32,
        mean=(0.4377, 0.4438, 0.4728),
        std=(0.1980, 0.2010, 0.1970),
        batch_size=128,
    )

    # 4. MNIST
    presets["mnist"] = DataConfig(
        name="mnist",
        display_name="MNIST",
        source=DataSource.TORCHVISION,
        tv_dataset_name="MNIST",
        num_classes=10,
        image_size=224,  # Resized to 224 for ViT
        mean=(0.1307, 0.1307, 0.1307),  # Duplicated for 3-channel
        std=(0.3081, 0.3081, 0.3081),
        batch_size=128,
    )

    # 5. Fashion-MNIST
    presets["fashion_mnist"] = DataConfig(
        name="fashion_mnist",
        display_name="Fashion-MNIST",
        source=DataSource.TORCHVISION,
        tv_dataset_name="FashionMNIST",
        num_classes=10,
        image_size=224,
        mean=(0.2860, 0.2860, 0.2860),
        std=(0.3530, 0.3530, 0.3530),
        batch_size=128,
    )

    # 6. STL-10
    presets["stl10"] = DataConfig(
        name="stl10",
        display_name="STL-10",
        source=DataSource.TORCHVISION,
        tv_dataset_name="STL10",
        num_classes=10,
        image_size=96,  # Native 96x96
        mean=(0.4467, 0.4398, 0.4066),
        std=(0.2603, 0.2566, 0.2713),
        batch_size=64,
    )

    # 7. TinyImageNet-200
    presets["tinyimagenet200"] = DataConfig(
        name="tinyimagenet200",
        display_name="TinyImageNet-200",
        source=DataSource.HF,
        hf_repo_id="Maysee/tiny-imagenet",
        num_classes=200,
        image_size=64,  # Native 64x64, resized to 224
        batch_size=64,
    )

    # 8. ImageNet-1K
    presets["imagenet1k"] = DataConfig(
        name="imagenet1k",
        display_name="ImageNet-1K",
        source=DataSource.HF,
        hf_repo_id="ILSVRC/imagenet-1k",
        num_classes=1000,
        image_size=224,
        batch_size=256,
        num_workers=8,
    )

    # 9. ChestX-Ray (Pneumonia) - Binary
    presets["chestxray_pneumonia"] = DataConfig(
        name="chestxray_pneumonia",
        display_name="ChestX-Ray (Pneumonia)",
        source=DataSource.LOCAL,
        num_classes=2,
        image_size=224,
        use_clahe=True,
        clahe_clip_limit=2.0,
        clahe_tile_grid=(8, 8),
        is_multi_label=False,
        batch_size=32,
    )

    # 10. Brain Tumor MRI
    presets["brain_tumor_mri"] = DataConfig(
        name="brain_tumor_mri",
        display_name="Brain Tumor MRI",
        source=DataSource.LOCAL,
        num_classes=3,  # Cheng 2016 Figshare: 3 classes (meningioma, glioma, pituitary)
        image_size=224,
        use_clahe=True,
        batch_size=32,
    )

    # 11. HAM10000 (Dermoscopy)
    presets["ham10000"] = DataConfig(
        name="ham10000",
        display_name="HAM10000",
        source=DataSource.HF,
        hf_repo_id="hkchengrex/HAM10000",
        num_classes=7,
        image_size=224,
        use_clahe=True,
        batch_size=32,
    )

    # 12. Intel Scenes
    presets["intel_scenes"] = DataConfig(
        name="intel_scenes",
        display_name="Intel Scenes",
        source=DataSource.HF,
        hf_repo_id="pcuenq/intel-image-classification",
        num_classes=6,
        image_size=224,
        batch_size=64,
    )

    # 13. COCO Multi-task (from original code)
    presets["coco_multitask"] = DataConfig(
        name="coco_multitask",
        display_name="COCO Multi-Task",
        source=DataSource.LOCAL,
        task="multi_task",
        num_classes=80,
        image_size=64,  # Original code uses 64
        batch_size=16,
    )

    # 14. Pose Action Recognition (video)
    presets["pose_action_recognition"] = DataConfig(
        name="pose_action_recognition",
        display_name="Pose Action Recognition",
        source=DataSource.HF,
        hf_repo_id="CristianLazoQuispe/pose-action-recognition",
        num_classes=20,  # Updated based on dataset metadata
        image_size=64,
        num_frames=8,
        batch_size=32,
        task="classification",
    )

    # 15. WLASL (video) - multiple configs
    presets["wlasl100"] = DataConfig(
        name="wlasl100",
        display_name="WLASL-100",
        source=DataSource.HF,
        hf_repo_id="aipieces/WLASL",
        hf_config="100",
        num_classes=100,
        image_size=64,
        num_frames=8,
        batch_size=32,
        task="classification",
    )

    presets["wlasl300"] = DataConfig(
        name="wlasl300",
        display_name="WLASL-300",
        source=DataSource.HF,
        hf_repo_id="aipieces/WLASL",
        hf_config="300",
        num_classes=300,
        image_size=64,
        num_frames=8,
        batch_size=32,
        task="classification",
    )

    presets["wlasl1000"] = DataConfig(
        name="wlasl1000",
        display_name="WLASL-1000",
        source=DataSource.HF,
        hf_repo_id="aipieces/WLASL",
        hf_config="1000",
        num_classes=1000,
        image_size=64,
        num_frames=8,
        batch_size=16,
        task="classification",
    )

    # 16. UCF101 (video)
    presets["ucf101"] = DataConfig(
        name="ucf101",
        display_name="UCF101",
        source=DataSource.HF,
        hf_repo_id="flwrlabs/ucf101",
        num_classes=101,
        image_size=64,
        num_frames=8,
        batch_size=32,
        task="classification",
    )

    # 17. UCF-Crime (video)
    presets["ucf_crime"] = DataConfig(
        name="ucf_crime",
        display_name="UCF-Crime",
        source=DataSource.HF,
        hf_repo_id="backseollgi/UCF-Crime",
        num_classes=14,
        image_size=64,
        num_frames=8,
        batch_size=32,
        task="classification",
    )

    return presets


# Auto-register presets
PRESETS = create_paper_presets()
for name, config in PRESETS.items():
    DatasetRegistry.register(name, config)