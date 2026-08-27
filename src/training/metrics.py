"""
Metrics Module

Classification, Segmentation, Detection metrics for evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
from collections import defaultdict
import numpy as np


# ─────────────────────────────────────────────────────────────────
# Classification Metrics
# ─────────────────────────────────────────────────────────────────

class Accuracy(nn.Module):
    """Top-1 accuracy."""

    def __init__(self, topk: int = 1):
        super().__init__()
        self.topk = topk

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() > 1:
            targets = targets.argmax(dim=1)

        if self.topk == 1:
            preds = logits.argmax(dim=1)
            return (preds == targets).float().mean()

        # Top-k accuracy
        _, preds = logits.topk(self.topk, dim=1, largest=True, sorted=True)
        correct = preds.eq(targets.view(-1, 1).expand_as(preds))
        return correct.any(dim=1).float().mean()


class TopKAccuracy(nn.Module):
    """Top-K accuracy (top-5, etc.)."""

    def __init__(self, k: int = 5):
        super().__init__()
        self.k = k

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() > 1:
            targets = targets.argmax(dim=1)

        _, preds = logits.topk(self.k, dim=1)
        correct = preds.eq(targets.view(-1, 1).expand_as(preds))
        return correct.any(dim=1).float().mean()


class BalancedAccuracy(nn.Module):
    """Balanced accuracy for imbalanced datasets."""

    def __init__(self, num_classes: Optional[int] = None):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() > 1:
            targets = targets.argmax(dim=1)

        preds = logits.argmax(dim=1)

        if self.num_classes is None:
            self.num_classes = logits.shape[1]

        accs = []
        for c in range(self.num_classes):
            mask = targets == c
            if mask.sum() > 0:
                accs.append((preds[mask] == c).float().mean())

        return torch.stack(accs).mean() if accs else torch.tensor(0.0)


class F1Score(nn.Module):
    """Macro F1 score."""

    def __init__(self, num_classes: int, average: str = "macro"):
        super().__init__()
        self.num_classes = num_classes
        self.average = average

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() > 1:
            targets = targets.argmax(dim=1)

        preds = logits.argmax(dim=1)

        f1s = []
        for c in range(self.num_classes):
            tp = ((preds == c) & (targets == c)).sum().float()
            fp = ((preds == c) & (targets != c)).sum().float()
            fn = ((preds != c) & (targets == c)).sum().float()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            f1s.append(f1)

        return torch.stack(f1s).mean()


# ─────────────────────────────────────────────────────────────────
# Segmentation Metrics
# ─────────────────────────────────────────────────────────────────

class IoU(nn.Module):
    """Intersection over Union (mIoU) for segmentation."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        average: str = "mean",  # "mean" | "per_class"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.average = average

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W]
            targets: [B, H, W] with class indices
        """
        preds = logits.argmax(dim=1)
        valid_mask = targets != self.ignore_index

        ious = []
        for c in range(self.num_classes):
            pred_c = (preds == c) & valid_mask
            target_c = (targets == c) & valid_mask

            intersection = (pred_c & target_c).sum().float()
            union = (pred_c | target_c).sum().float()

            if union > 0:
                ious.append(intersection / union)
            else:
                ious.append(torch.tensor(1.0, device=logits.device))  # No ground truth = perfect

        ious = torch.stack(ious)
        if self.average == "mean":
            return ious.mean()
        return ious


class DiceScore(nn.Module):
    """Dice coefficient (F1 for segmentation)."""

    def __init__(self, num_classes: int, ignore_index: int = 255, average: str = "mean"):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.average = average

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = logits.argmax(dim=1)
        valid_mask = targets != self.ignore_index

        dices = []
        for c in range(self.num_classes):
            pred_c = (preds == c) & valid_mask
            target_c = (targets == c) & valid_mask

            intersection = (pred_c & target_c).sum().float()
            pred_sum = pred_c.sum().float()
            target_sum = target_c.sum().float()

            dice = (2 * intersection) / (pred_sum + target_sum + 1e-8)
            dices.append(dice)

        dices = torch.stack(dices)
        if self.average == "mean":
            return dices.mean()
        return dices


class PixelAccuracy(nn.Module):
    """Pixel accuracy (ignoring ignore_index)."""

    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = logits.argmax(dim=1)
        valid_mask = targets != self.ignore_index
        correct = (preds == targets) & valid_mask
        return correct.sum().float() / valid_mask.sum().float()


# ─────────────────────────────────────────────────────────────────
# Detection Metrics
# ─────────────────────────────────────────────────────────────────

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """IoU between two sets of boxes (xyxy format)."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / union


class COCOmAP(nn.Module):
    """
    COCO-style mAP (mean Average Precision) for object detection.

    Computes AP@[IoU=0.50:0.95] over 10 IoU thresholds.
    """

    def __init__(self, num_classes: int = 80, iou_thresholds: Optional[List[float]] = None):
        super().__init__()
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [0.5 + i * 0.05 for i in range(10)]
        self.predictions = []
        self.targets = []

    def update(self, preds: List[Dict], targets: List[Dict]):
        """Accumulate predictions and targets."""
        self.predictions.extend(preds)
        self.targets.extend(targets)

    def compute(self) -> Dict[str, float]:
        """Compute COCO mAP metrics."""
        if not self.predictions:
            return {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0}

        # Simplified implementation - uses pycocotools if available
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except ImportError:
            logger.warning("pycocotools not available, returning dummy metrics")
            return {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0, "APs": 0.0, "APm": 0.0, "APl": 0.0}

        # Convert to COCO format
        gt_annots = self._to_coco_gt(self.targets)
        dt_annots = self._to_coco_dt(self.predictions)

        coco_gt = COCO()
        coco_gt.dataset = gt_annots
        coco_gt.createIndex()

        coco_dt = coco_gt.loadRes(dt_annots)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats  # 12 stats array
        return {
            "mAP": stats[0],     # AP @ IoU=0.50:0.95
            "AP50": stats[1],    # AP @ IoU=0.50
            "AP75": stats[2],    # AP @ IoU=0.75
            "APs": stats[3],     # AP for small objects
            "APm": stats[4],     # AP for medium objects
            "APl": stats[5],     # AP for large objects
        }

    def reset(self):
        self.predictions = []
        self.targets = []

    def _to_coco_gt(self, targets):
        """Convert targets to COCO ground truth format."""
        # Implementation would go here
        pass

    def _to_coco_dt(self, predictions):
        """Convert predictions to COCO detection format."""
        # Implementation would go here
        pass


class MeanAveragePrecision(nn.Module):
    """Simplified mAP without pycocotools."""

    def __init__(self, num_classes: int, iou_threshold: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.predictions = []
        self.targets = []

    def update(self, preds: Dict, targets: Dict):
        """Update with batch predictions."""
        # Simplified - would implement proper AP calculation
        pass

    def compute(self) -> Dict:
        return {"map": 0.0}

    def reset(self):
        self.predictions = []
        self.targets = []


# ─────────────────────────────────────────────────────────────────
# Metric Collections
# ─────────────────────────────────────────────────────────────────

class MetricCollection(nn.Module):
    """Collection of metrics updated together."""

    def __init__(self, metrics: Dict[str, nn.Module]):
        super().__init__()
        self.metrics = nn.ModuleDict(metrics)

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric(*args, **kwargs)
        return results

    def update(self, *args, **kwargs):
        for metric in self.metrics.values():
            if hasattr(metric, "update"):
                metric.update(*args, **kwargs)

    def compute(self) -> Dict[str, torch.Tensor]:
        results = {}
        for name, metric in self.metrics.items():
            if hasattr(metric, "compute"):
                results[name] = metric.compute()
            else:
                results[name] = 0.0
        return results


def get_classification_metrics(num_classes: int = 1000, topk: Tuple[int, ...] = (1, 5)) -> MetricCollection:
    """Get standard classification metrics."""
    metrics = {"accuracy": Accuracy()}
    for k in topk:
        metrics[f"top{k}_accuracy"] = TopKAccuracy(k=k)
    metrics["f1_macro"] = F1Score(num_classes)
    return MetricCollection(metrics)


def get_segmentation_metrics(num_classes: int, ignore_index: int = 255) -> MetricCollection:
    """Get standard segmentation metrics."""
    return MetricCollection({
        "mIoU": IoU(num_classes, ignore_index),
        "dice": DiceScore(num_classes, ignore_index),
        "pixel_acc": PixelAccuracy(ignore_index),
    })


def get_detection_metrics(num_classes: int) -> MetricCollection:
    """Get standard detection metrics."""
    return MetricCollection({
        "COCOmAP": COCOmAP(num_classes),
    })


def get_metrics(task: str, **kwargs) -> MetricCollection:
    """Factory for metric collections."""
    builders = {
        "classification": get_classification_metrics,
        "segmentation": get_segmentation_metrics,
        "detection": get_detection_metrics,
    }

    if task not in builders:
        raise ValueError(f"Unknown task: {task}")

    return builders[task](**kwargs)