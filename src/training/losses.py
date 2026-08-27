"""
Loss Functions for I-HRM

- Classification: CrossEntropy, Focal, LabelSmoothing
- Segmentation: CrossEntropy, Dice, Focal, Combined
- Detection: Focal + L1/GIoU
- Multi-task: Weighted combination
- Halting: ACT loss (Algorithm 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List


# ─────────────────────────────────────────────────────────────────
# Classification Losses
# ─────────────────────────────────────────────────────────────────

class ClassificationLoss(nn.Module):
    """Standard cross-entropy with optional label smoothing."""

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
            reduction=reduction,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets)


class FocalLoss(nn.Module):
    """Focal Loss for dense object detection (Lin et al., 2017)."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none", ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothed Cross Entropy (Szegedy et al., 2016)."""

    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        smooth_loss = -log_probs.mean(dim=-1)

        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            loss = loss[mask]

        return loss.mean()


# ─────────────────────────────────────────────────────────────────
# Segmentation Losses
# ─────────────────────────────────────────────────────────────────

class SegmentationLoss(nn.Module):
    """Combined CrossEntropy + Dice loss for segmentation."""

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_weight: float = 0.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.smooth = smooth
        self.focal = FocalLoss(alpha=0.25, gamma=2.0, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = {}

        # Cross Entropy
        if self.ce_weight > 0:
            losses["ce"] = self.ce_vector(logits, targets)

        # Dice
        if self.dice_weight > 0:
            losses["dice"] = self.dice_loss(logits, targets)

        # Focal
        if self.focal_weight > 0:
            losses["focal"] = self.focal(logits, targets)

        total = sum(w * l for (k, l), w in zip(losses.items(), [self.ce_weight, self.dice_weight, self.focal_weight]))
        return total

    def ce_vector(self, logits, targets):
        return self.ce(logits, targets)

    def dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Dice loss for multi-class segmentation."""
        probs = F.softmax(logits, dim=1)
        num_classes = probs.shape[1]

        # One-hot encode targets
        valid_mask = targets != self.ignore_index
        targets_onehot = F.one_hot(targets.clamp(min=0), num_classes).float()
        targets_onehot = targets_onehot.permute(0, 3, 1, 2)  # [B, C, H, W]

        # Apply valid mask
        valid_mask = valid_mask.unsqueeze(1).float()
        probs = probs * valid_mask
        targets_onehot = targets_onehot * valid_mask

        intersection = torch.sum(probs * targets_onehot, dim=(2, 3))
        union = torch.sum(probs + targets_onehot, dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class DiceLoss(nn.Module):
    """Pure Dice loss."""

    def __init__(self, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        num_classes = probs.shape[1]

        valid_mask = targets != self.ignore_index
        targets_onehot = F.one_hot(targets.clamp(min=0), num_classes).float().permute(0, 3, 1, 2)

        valid_mask = valid_mask.unsqueeze(1).float()
        probs = probs * valid_mask
        targets_onehot = targets_onehot * valid_mask

        intersection = torch.sum(probs * targets_onehot, dim=(2, 3))
        union = torch.sum(probs + targets_onehot, dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalSegmentationLoss(nn.Module):
    """Focal loss adapted for segmentation."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = 255):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal(logits, targets)


# ─────────────────────────────────────────────────────────────────
# Detection Losses
# ─────────────────────────────────────────────────────────────────

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes (xyxy format)."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Generalized IoU (Rezatofighi et al., 2019)."""
    iou = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area_c = wh[:, :, 0] * wh[:, :, 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - iou * (area1[:, None] * area2).sqrt()

    return iou - (area_c - union) / area_c


class DetectionLoss(nn.Module):
    """Detection loss: Focal (classification) + L1 + GIoU (bbox)."""

    def __init__(
        self,
        num_classes: int = 80,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        bbox_loss_type: str = "giou",  # "l1" | "giou" | "both"
        cls_weight: float = 1.0,
        bbox_weight: float = 5.0,
        giou_weight: float = 2.0,
        num_proposals: int = 100,
        eos_coef: float = 0.1,  # Background class weight
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight
        self.bbox_loss_type = bbox_loss_type

        # Classification: Focal Loss
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        # Class weights (background class gets lower weight)
        if eos_coef is not None:
            class_weights = torch.ones(num_classes + 1)
            class_weights[-1] = eos_coef  # Last class = background/no-object
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        cls_logits: torch.Tensor,      # [B, num_proposals, num_classes+1]
        bbox_pred: torch.Tensor,       # [B, num_proposals, 4] (cx, cy, w, h) normalized
        targets: List[Dict],           # List of dicts with 'labels', 'boxes' (xyxy normalized)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute detection loss with Hungarian matching (simplified - no matcher here).
        In practice, use a matcher like DETR's HungarianMatcher.
        """
        batch_size = cls_logits.shape[0]
        num_proposals = cls_logits.shape[1]

        # For simplicity, assume targets are padded to num_proposals
        # Real implementation uses Hungarian matching

        total_cls_loss = 0
        total_bbox_loss = 0
        total_giou_loss = 0

        for b in range(batch_size):
            tgt_labels = targets[b]["labels"]  # [N]
            tgt_boxes = targets[b]["boxes"]    # [N, 4] xyxy normalized

            # Get predictions for this batch
            pred_cls = cls_logits[b]  # [P, C+1]
            pred_boxes = bbox_pred[b]  # [P, 4] cxcywh

            # Simplified: match by min class cost (no Hungarian)
            # This is a placeholder - production uses proper matching
            if len(tgt_labels) > 0:
                # Classification loss
                cls_loss = F.cross_entropy(
                    pred_cls[:len(tgt_labels)],
                    tgt_labels,
                    weight=self.class_weights.to(cls_logits.device) if self.class_weights is not None else None,
                )

                # Bbox loss (convert to xyxy)
                pred_boxes_xyxy = self.cxcywh_to_xyxy(pred_boxes[:len(tgt_labels)])
                tgt_boxes_xyxy = tgt_boxes

                bbox_loss = F.l1_loss(pred_boxes_xyxy, tgt_boxes_xyxy)

                if self.bbox_loss_type == "giou" or self.bbox_loss_type == "both":
                    giou = generalized_box_iou(pred_boxes_xyxy, tgt_boxes_xyxy)
                    giou_loss = 1 - torch.diag(giou).mean()
                else:
                    giou_loss = torch.tensor(0.0, device=cls_logits.device)

                total_cls_loss += cls_loss
                total_bbox_loss += bbox_loss
                total_giou_loss += giou_loss

        # Average over batch
        losses = {}
        if batch_size > 0:
            losses["cls"] = total_cls_loss / batch_size
            losses["bbox"] = total_bbox_loss / batch_size
            losses["giou"] = total_giou_loss / batch_size

        total = (
            self.cls_weight * losses.get("cls", 0) +
            self.bbox_weight * losses.get("bbox", 0) +
            self.giou_weight * losses.get("giou", 0)
        )

        return total, losses

    def cxcywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert cx,cy,w,h to x1,y1,x2,y2 (normalized)."""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1).clamp(0, 1)


# ─────────────────────────────────────────────────────────────────
# Halting / ACT Losses (Algorithm 1 from paper)
# ─────────────────────────────────────────────────────────────────

class HaltingLoss(nn.Module):
    """
    Adaptive Computation Time (ACT) Halting Loss.

    Implements Algorithm 1 from the paper:
    - Each step k produces halt probability h_t^k
    - Compute expected halting step
    - Minimize difference from actual halting step
    """

    def __init__(
        self,
        max_steps: int = 6,
        loss_weight: float = 0.1,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.loss_weight = loss_weight
        self.epsilon = epsilon

    def forward(self, halt_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            halt_logits: [B, max_steps] or [B, seq_len, max_steps] - logits for halting at each step

        Returns:
            halting loss scalar
        """
        B = halt_logits.shape[0]
        halt_probs = torch.sigmoid(halt_logits)  # [B, K] or [B, L, K]

        if halt_probs.dim() == 3:
            # Sequence dimension present - average over tokens
            halt_probs = halt_probs.mean(dim=1)  # [B, K]

        # Compute cumulative distribution
        # Expected step = sum_{k=1}^K k * h_k where h_k is prob of halting exactly at k
        # h_k = p_k * prod_{j<k} (1 - p_j)

        p = halt_probs
        not_halted = torch.ones_like(p[:, :1])  # [B, 1]
        expected_steps = torch.zeros(B, device=p.device)

        for k in range(self.max_steps):
            h_k = p[:, k] * not_halted.squeeze(-1)  # Prob halting at step k
            expected_steps += (k + 1) * h_k
            not_halted = not_halted * (1 - p[:, k:k+1])

        # Remainder gets max_steps
        expected_steps += self.max_steps * not_halted.squeeze(-1)

        # Target: we want early halting but not too early
        # Paper uses λ * expected_steps as regularization
        loss = self.loss_weight * expected_steps.mean()

        return loss


class ACTLoss(nn.Module):
    """
    Full ACT Loss (Graves, 2016) adapted for HRM.

    Combines:
    1. Task loss (cross-entropy, etc.)
    2. Halting regularization (expected steps penalty)
    3. Ponder cost
    """

    def __init__(
        self,
        base_loss: nn.Module,
        max_steps: int = 6,
        halt_penalty: float = 0.1,
        ponder_weight: float = 0.01,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.max_steps = max_steps
        self.halt_penalty = halt_penalty
        self.ponder_weight = ponder_weight
        self.halting_loss = HaltingLoss(max_steps, halt_penalty)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        halt_logits: Optional[torch.Tensor] = None,
        step_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            logits: [B, C] or [B, L, C] - task predictions
            targets: [B] or [B, L] - labels
            halt_logits: [B, K] or [B, L, K] - halting logits per step
            step_probs: [B, K] - softmax probabilities per step (if available)
        """
        task_loss = self.base_loss(logits, targets)

        losses = {"task": task_loss}

        if halt_logits is not None:
            halt_loss = self.halting_loss(halt_logits)
            losses["halt"] = halt_loss

            # Ponder cost (sum of halting probabilities over steps)
            if step_probs is not None:
                ponder_cost = self.ponder_weight * step_probs.sum(dim=-1).mean()
                losses["ponder"] = ponder_cost

        total = sum(losses.values())
        losses["total"] = total

        return total, losses


# ─────────────────────────────────────────────────────────────────
# Multi-Task Loss
# ─────────────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    Weighted combination of multiple task losses.

    Tasks: classification, segmentation, detection
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        seg_weight: float = 1.0,
        det_weight: float = 1.0,
        halt_weight: float = 0.1,
        cls_loss_type: str = "ce",
        seg_loss_type: str = "ce_dice",
        det_loss_type: str = "focal_l1_giou",
        num_classes: int = 80,
        seg_num_classes: int = 80,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.seg_weight = seg_weight
        self.det_weight = det_weight
        self.halt_weight = halt_weight

        # Classification loss
        if cls_loss_type == "ce":
            self.cls_loss = ClassificationLoss()
        elif cls_loss_type == "focal":
            self.cls_loss = FocalLoss()
        elif cls_loss_type == "label_smoothing":
            self.cls_loss = LabelSmoothingCrossEntropy()
        else:
            raise ValueError(f"Unknown cls_loss_type: {cls_loss_type}")

        # Segmentation loss
        if seg_loss_type == "ce":
            self.seg_loss = ClassificationLoss(ignore_index=ignore_index)
        elif seg_loss_type == "ce_dice":
            self.seg_loss = SegmentationLoss(ce_weight=1.0, dice_weight=1.0, ignore_index=ignore_index)
        elif seg_loss_type == "dice":
            self.seg_loss = DiceLoss(ignore_index=ignore_index)
        elif seg_loss_type == "focal":
            self.seg_loss = FocalSegmentationLoss(ignore_index=ignore_index)
        else:
            raise ValueError(f"Unknown seg_loss_type: {seg_loss_type}")

        # Detection loss
        if det_loss_type == "focal_l1_giou":
            self.det_loss = DetectionLoss(num_classes=num_classes)
        else:
            self.det_loss = DetectionLoss(num_classes=num_classes)

        # Halting loss
        self.halt_loss = HaltingLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            outputs: Dict with keys:
                - 'cls_logits': [B, C] or [B, L, C]
                - 'seg_logits': [B, C, H, W]
                - 'det_cls': [B, P, C+1], 'det_bbox': [B, P, 4]
                - 'halt_logits': [B, K] (optional)
            targets: Dict with corresponding keys
        """
        losses = {}

        # Classification
        if "cls_logits" in outputs and "cls_labels" in targets:
            losses["cls"] = self.cls_loss(outputs["cls_logits"], targets["cls_labels"])
            losses["cls"] *= self.cls_weight

        # Segmentation
        if "seg_logits" in outputs and "seg_masks" in targets:
            losses["seg"] = self.seg_loss(outputs["seg_logits"], targets["seg_masks"])
            losses["seg"] *= self.seg_weight

        # Detection
        if "det_cls" in outputs and "det_labels" in targets:
            det_loss, det_losses = self.det_loss(
                outputs["det_cls"],
                outputs["det_bbox"],
                targets["det_labels"],
                targets.get("det_boxes"),
            )
            losses.update({f"det_{k}": v * self.det_weight for k, v in det_losses.items()})

        # Halting
        if "halt_logits" in outputs:
            losses["halt"] = self.halt_loss(outputs["halt_logits"])
            losses["halt"] *= self.halt_weight

        total_loss = sum(losses.values())
        losses["total"] = total_loss

        return total_loss, losses


# ─────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────

def get_loss_fn(task: str, **kwargs) -> nn.Module:
    """
    Get loss function for task.

    Args:
        task: "classification" | "focal" | "label_smoothing" | "segmentation" |
              "dice" | "focal_segmentation" | "detection" | "halting" | "act" | "multi_task"
        **kwargs: Loss-specific arguments
    """
    losses = {
        "classification": lambda: ClassificationLoss(**kwargs),
        "focal": lambda: FocalLoss(**kwargs),
        "label_smoothing": lambda: LabelSmoothingCrossEntropy(**kwargs),
        "segmentation": lambda: SegmentationLoss(**kwargs),
        "dice": lambda: DiceLoss(**kwargs),
        "focal_segmentation": lambda: FocalSegmentationLoss(**kwargs),
        "detection": lambda: DetectionLoss(**kwargs),
        "halting": lambda: HaltingLoss(**kwargs),
        "act": lambda: ACTLoss(**kwargs),
        "multi_task": lambda: MultiTaskLoss(**kwargs),
    }

    if task not in losses:
        raise ValueError(f"Unknown loss: {task}. Available: {list(losses.keys())}")

    return losses[task]()