"""
Evaluator Module

Main evaluation class for running inference and computing metrics.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Any, Callable, Union
from pathlib import Path
import json
from collections import defaultdict

from src.utils.logging import get_logger

logger = get_logger(__name__)


class Evaluator:
    """
    Model evaluator for classification, segmentation, detection, and multi-task.
    """

    def __init__(
        self,
        model: nn.Module,
        task: str = "classification",
        device: Optional[torch.device] = None,
        metrics: Optional[Dict[str, Callable]] = None,
        use_amp: bool = True,
    ):
        """
        Args:
            model: PyTorch model (backbone or HF UnifiedCVHRM)
            task: "classification" | "segmentation" | "detection" | "multi_task"
            device: Compute device
            metrics: Dict of metric functions
            use_amp: Use mixed precision
        """
        self.model = model
        self.task = task
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = metrics or {}
        self.use_amp = use_amp and self.device.type == "cuda"

        self.model.to(self.device)
        self.model.eval()

        # Default metrics per task
        self._init_default_metrics()

    def _init_default_metrics(self):
        """Initialize default metrics for task."""
        from src.training.metrics import (
            Accuracy, TopKAccuracy, IoU, DiceScore, PixelAccuracy, COCOmAP
        )

        if self.task == "classification":
            self.metrics.setdefault("accuracy", Accuracy(topk=1))
            self.metrics.setdefault("top5", TopKAccuracy(k=5))
        elif self.task == "segmentation":
            self.metrics.setdefault("iou", IoU(num_classes=21, average="mean"))
            self.metrics.setdefault("dice", DiceScore(num_classes=21, average="mean"))
            self.metrics.setdefault("pixel_acc", PixelAccuracy())
        elif self.task == "detection":
            self.metrics.setdefault("coco_map", COCOmAP(num_classes=80))

    def evaluate(
        self,
        dataloader: DataLoader,
        metrics: Optional[List[str]] = None,
        prefix: str = "",
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on dataloader.

        Returns:
            Dict of metric names -> values
        """
        metrics_to_compute = metrics or list(self.metrics.keys())
        metric_sums = defaultdict(float)
        num_batches = 0

        logger.info(f"Evaluating {self.task} on {len(dataloader)} batches...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break

                batch = self._to_device(batch)
                outputs = self._forward(batch)

                # Compute metrics
                for name in metrics_to_compute:
                    if name in self.metrics:
                        if self.task == "classification":
                            val = self._compute_classification_metrics(outputs, batch, name)
                        elif self.task == "segmentation":
                            val = self._compute_segmentation_metrics(outputs, batch, name)
                        elif self.task == "detection":
                            val = self._compute_detection_metrics(outputs, batch, name)
                        else:
                            val = 0.0

                        metric_sums[name] += val
                num_batches += 1

        # Average results
        results = {f"{prefix}{k}": v / max(num_batches, 1) for k, v in metric_sums.items()}
        logger.info(f"Evaluation results: {results}")
        return results

    def _forward(self, batch) -> Dict[str, torch.Tensor]:
        """Forward pass through model."""
        if isinstance(batch, dict):
            pixel_values = batch["image"]
        elif isinstance(batch, (list, tuple)):
            pixel_values = batch[0]
        else:
            pixel_values = batch

        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self._model_forward(pixel_values)
        else:
            outputs = self._model_forward(pixel_values)

        return outputs

    def _model_forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Model-specific forward."""
        # Handle HF UnifiedCVHRM
        if hasattr(self.model, "backbone"):
            features, _ = self.model.backbone(pixel_values, return_steps=True)
            if self.task == "classification":
                logits = self.model.heads.classification_head(features)
            elif self.task == "segmentation":
                logits = self.model.heads.segmentation_head(features)
            elif self.task == "detection":
                cls_logits, bbox_pred = self.model.heads.detection_head(features)
                return {"det_cls": cls_logits, "det_bbox": bbox_pred}
            return {"cls_logits": logits}
        else:
            # Local HierarchicalVisionTransformer
            features, _ = self.model(pixel_values, return_steps=True)
            if self.task == "classification":
                logits = self.model.heads.classification_head(features)
                return {"cls_logits": logits}
            elif self.task == "segmentation":
                logits = self.model.heads.segmentation_head(features)
                return {"seg_logits": logits}
            elif self.task == "detection":
                cls_logits, bbox_pred = self.model.heads.detection_head(features)
                return {"det_cls": cls_logits, "det_bbox": bbox_pred}
            return {"features": features}

    def _compute_classification_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Any,
        metric_name: str,
    ) -> float:
        """Compute classification metrics."""
        logits = outputs.get("cls_logits")
        if logits is None:
            return 0.0

        if isinstance(batch, dict):
            targets = batch.get("label") or batch.get("labels")
        else:
            targets = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else None

        if targets is None:
            return 0.0

        targets = targets.to(self.device)
        metric_fn = self.metrics[metric_name]
        return metric_fn(logits, targets).item()

    def _compute_segmentation_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Any,
        metric_name: str,
    ) -> float:
        """Compute segmentation metrics."""
        logits = outputs.get("seg_logits")
        if logits is None:
            return 0.0

        if isinstance(batch, dict):
            targets = batch.get("seg_masks") or batch.get("segmentation")
        else:
            targets = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else None

        if targets is None:
            return 0.0

        targets = targets.to(self.device)
        metric_fn = self.metrics[metric_name]
        return metric_fn(logits, targets).item()

    def _compute_detection_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Any,
        metric_name: str,
    ) -> float:
        """Compute detection metrics (placeholder - uses COCO)."""
        # COCO mAP requires accumulating predictions
        # This is a simplified version
        return 0.0

    def _to_device(self, batch):
        """Move batch to device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, (list, tuple)):
            return [self._to_device(b) for b in batch]
        elif isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        return batch


# ─────────────────────────────────────────────────────────────────
# Inference Helpers
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(
    model: nn.Module,
    pixel_values: torch.Tensor,
    task: str = "classification",
    top_k: int = 5,
    device: Optional[torch.device] = None,
    use_amp: bool = True,
) -> Dict[str, Any]:
    """
    Run inference on single image/batch.

    Args:
        model: I-HRM model
        pixel_values: (B, C, H, W) or (B, C, T, H, W)
        task: Task type
        top_k: Top-K predictions for classification
        device: Device
        use_amp: Use AMP

    Returns:
        Dict with predictions
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    pixel_values = pixel_values.to(device)

    if use_amp and device.type == "cuda":
        with torch.cuda.amp.autocast():
            outputs = _predict_forward(model, pixel_values, task)
    else:
        outputs = _predict_forward(model, pixel_values, task)

    if task == "classification":
        logits = outputs.get("cls_logits")
        if logits is not None:
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_idx = probs.topk(top_k, dim=-1)
            return {
                "labels": top_idx.cpu().tolist(),
                "scores": top_probs.cpu().tolist(),
            }

    elif task == "segmentation":
        logits = outputs.get("seg_logits")
        if logits is not None:
            pred = logits.argmax(dim=1)
            return {"segmentation": pred.cpu().numpy()}

    elif task == "detection":
        cls_logits = outputs.get("det_cls")
        bbox_pred = outputs.get("det_bbox")
        if cls_logits is not None:
            probs = torch.softmax(cls_logits, dim=-1)
            top_probs, top_idx = probs.topk(top_k, dim=-1)
            return {
                "labels": top_idx.cpu().tolist(),
                "scores": top_probs.cpu().tolist(),
                "boxes": bbox_pred.cpu().tolist() if bbox_pred is not None else [],
            }

    return {}


def _predict_forward(model: nn.Module, pixel_values: torch.Tensor, task: str) -> Dict:
    """Internal forward for prediction."""
    if hasattr(model, "backbone"):  # HF UnifiedCVHRM
        features, _ = model.backbone(pixel_values, return_steps=True)
        if task == "classification":
            logits = model.heads.classification_head(features)
            return {"cls_logits": logits}
        elif task == "segmentation":
            logits = model.heads.segmentation_head(features)
            return {"seg_logits": logits}
        elif task == "detection":
            cls_logits, bbox_pred = model.heads.detection_head(features)
            return {"det_cls": cls_logits, "det_bbox": bbox_pred}
    else:  # Local model
        features, _ = model(pixel_values, return_steps=True)
        if task == "classification":
            logits = model.heads.classification_head(features)
            return {"cls_logits": logits}
        elif task == "segmentation":
            logits = model.heads.segmentation_head(features)
            return {"seg_logits": logits}
        elif task == "detection":
            cls_logits, bbox_pred = model.heads.detection_head(features)
            return {"det_cls": cls_logits, "det_bbox": bbox_pred}
    return {}


def create_evaluator(
    model: nn.Module,
    device: Optional[torch.device] = None,
    use_amp: bool = True,
    metrics: Optional[Dict[str, Any]] = None,
) -> Evaluator:
    """
    Factory function to create an Evaluator instance.

    Args:
        model: Model to evaluate
        device: Device (auto-detected if None)
        use_amp: Use automatic mixed precision
        metrics: Optional custom metrics dict

    Returns:
        Evaluator instance
    """
    return Evaluator(
        model=model,
        device=device,
        use_amp=use_amp,
        metrics=metrics,
    )