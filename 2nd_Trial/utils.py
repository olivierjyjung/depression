"""
MDTD: Multimodal Depression Temporal Detector
Utility functions
"""

import os
import random
import logging
import numpy as np
import torch
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(save_dir: str, log_file: str = "train.log") -> logging.Logger:
    """Setup logging to file and console"""
    os.makedirs(save_dir, exist_ok=True)

    logger = logging.getLogger("MDTD")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(os.path.join(save_dir, log_file))
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        preds: (N,) or (N, 1) predicted probabilities
        labels: (N,) or (N, 1) ground truth labels
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    # Flatten if needed
    preds = preds.flatten()
    labels = labels.flatten()

    # Binary predictions
    pred_labels = (preds >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, pred_labels),
        "precision": precision_score(labels, pred_labels, zero_division=0),
        "recall": recall_score(labels, pred_labels, zero_division=0),
        "f1": f1_score(labels, pred_labels, zero_division=0),
    }

    # AUC metrics (need probability scores)
    try:
        metrics["auc_roc"] = roc_auc_score(labels, preds)
    except ValueError:
        metrics["auc_roc"] = 0.0

    try:
        metrics["auc_pr"] = average_precision_score(labels, preds)
    except ValueError:
        metrics["auc_pr"] = 0.0

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, pred_labels).ravel()
    metrics["true_positives"] = int(tp)
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return metrics


class EarlyStopping:
    """Early stopping handler"""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "min",
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.

        Args:
            value: Current metric value

        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True
            return True

        return False

    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_value = None
        self.should_stop = False


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str,
):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[int, Dict[str, float]]:
    """
    Load model checkpoint.

    Returns:
        epoch: Epoch number
        metrics: Validation metrics at checkpoint
    """
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dictionary as string"""
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


class AverageMeter:
    """Compute and store the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return 0.0
