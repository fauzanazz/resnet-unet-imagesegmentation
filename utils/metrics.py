import torch
import numpy as np
from typing import Dict, List, Tuple


class SegmentationMetrics:
    """Compute segmentation metrics: IoU, pixel accuracy, etc."""

    def __init__(self, num_classes: int, ignore_index: int = None):
        """
        Args:
            num_classes: Number of classes (excluding background if ignore_index used)
            ignore_index: Class index to ignore in metrics (e.g., void pixels)
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # Confusion matrix: [pred, true]
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)

    def reset(self):
        """Reset confusion matrix."""
        self.confusion_matrix.zero_()

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update confusion matrix with batch predictions and targets.

        Args:
            preds: Predicted class indices [N, H, W] or [N*H*W]
            targets: Ground truth class indices [N, H, W] or [N*H*W]
        """
        if preds.dim() == 3:  # [N, H, W]
            preds = preds.flatten()
        if targets.dim() == 3:  # [N, H, W]
            targets = targets.flatten()

        # Filter out ignore_index pixels
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            preds = preds[valid_mask]
            targets = targets[valid_mask]

        # Clamp predictions to valid range
        preds = torch.clamp(preds, 0, self.num_classes - 1)
        targets = torch.clamp(targets, 0, self.num_classes - 1)

        # Update confusion matrix
        indices = self.num_classes * targets + preds
        self.confusion_matrix += torch.bincount(indices, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)

    def compute_iou(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Intersection over Union (IoU) for each class and mean IoU.

        Returns:
            class_ious: IoU for each class [num_classes]
            miou: Mean IoU across classes
        """
        # IoU = TP / (TP + FP + FN)
        intersection = torch.diag(self.confusion_matrix)  # TP for each class
        union = self.confusion_matrix.sum(dim=0) + self.confusion_matrix.sum(dim=1) - intersection  # TP + FP + FN

        # Avoid division by zero
        union = torch.where(union == 0, torch.ones_like(union), union)
        class_ious = intersection.float() / union.float()

        # Mean IoU (ignore classes with no ground truth)
        valid_classes = union > 0
        if valid_classes.any():
            miou = class_ious[valid_classes].mean()
        else:
            miou = torch.tensor(0.0)

        return class_ious, miou

    def compute_pixel_accuracy(self) -> torch.Tensor:
        """Compute pixel-wise accuracy."""
        total_pixels = self.confusion_matrix.sum()
        correct_pixels = torch.diag(self.confusion_matrix).sum()

        if total_pixels == 0:
            return torch.tensor(0.0)

        return correct_pixels.float() / total_pixels.float()

    def compute_dice(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Dice coefficient for each class and mean Dice.

        Returns:
            class_dices: Dice for each class [num_classes]
            mdice: Mean Dice across classes
        """
        # Dice = 2 * TP / (2 * TP + FP + FN)
        intersection = torch.diag(self.confusion_matrix)  # TP
        denominator = 2 * intersection + self.confusion_matrix.sum(dim=0) + self.confusion_matrix.sum(dim=1) - 2 * intersection

        # Avoid division by zero
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        class_dices = 2 * intersection.float() / denominator.float()

        # Mean Dice
        valid_classes = denominator > 0
        if valid_classes.any():
            mdice = class_dices[valid_classes].mean()
        else:
            mdice = torch.tensor(0.0)

        return class_dices, mdice

    def get_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics and return as dictionary.

        Returns:
            dict with keys: 'miou', 'pixel_acc', 'mdice', 'class_ious', 'class_dices'
        """
        class_ious, miou = self.compute_iou()
        pixel_acc = self.compute_pixel_accuracy()
        class_dices, mdice = self.compute_dice()

        return {
            'miou': miou.item(),
            'pixel_acc': pixel_acc.item(),
            'mdice': mdice.item(),
            'class_ious': class_ious.tolist(),
            'class_dices': class_dices.tolist()
        }


def compute_segmentation_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = None
) -> Dict[str, float]:
    """
    Compute segmentation metrics for predictions and targets.

    Args:
        preds: Predicted class indices [N, H, W] or flattened
        targets: Ground truth class indices [N, H, W] or flattened
        num_classes: Number of classes
        ignore_index: Class index to ignore

    Returns:
        Dictionary with metrics
    """
    metrics = SegmentationMetrics(num_classes, ignore_index)
    metrics.update(preds, targets)
    return metrics.get_metrics()


def print_metrics_table(metrics: Dict[str, float], class_names: List[str] = None):
    """Print a formatted table of metrics."""
    print("\n" + "="*50)
    print("SEGMENTATION METRICS")
    print("="*50)

    print(".4f")
    print(".4f")

    if 'mdice' in metrics:
        print(".4f")

    if 'class_ious' in metrics and class_names:
        print("\nPer-Class IoU:")
        print("-"*30)
        for i, (iou, name) in enumerate(zip(metrics['class_ious'], class_names)):
            print("2d")
    elif 'class_ious' in metrics:
        print("\nPer-Class IoU:")
        print("-"*30)
        for i, iou in enumerate(metrics['class_ious']):
            print("2d")

    print("="*50)


# Convenience functions for common use cases
def iou_score(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index=None) -> float:
    """Compute mean IoU."""
    metrics = compute_segmentation_metrics(preds, targets, num_classes, ignore_index)
    return metrics['miou']

def pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index=None) -> float:
    """Compute pixel accuracy."""
    metrics = compute_segmentation_metrics(preds, targets, num_classes, ignore_index)
    return metrics['pixel_acc']

def dice_score(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index=None) -> float:
    """Compute mean Dice score."""
    metrics = compute_segmentation_metrics(preds, targets, num_classes, ignore_index)
    return metrics['mdice']
