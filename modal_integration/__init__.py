"""
Modal Package for HRM Training

Minimal integration using existing infrastructure:
- scripts/download_data.py for dataset downloading
- src/training/trainer.py Trainer class for training
- src/data for dataset loading
"""

from . import volumes
from . import images
from . import modal_secrets as secrets

__all__ = [
    "volumes",
    "images",
    "modal_secrets",
]

__version__ = "1.0.0"