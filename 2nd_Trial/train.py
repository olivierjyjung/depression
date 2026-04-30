"""
MDTD: Multimodal Depression Temporal Detector
Training pipeline
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional
import json
from datetime import datetime

from config import Config, get_config
from model import DepressionDetector, MultiTaskLoss, create_model, create_loss
from dataset import create_dataloaders
from utils import (
    set_seed,
    compute_metrics,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    setup_logging,
)


class Trainer:
    """Training manager for MDTD model"""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.train.device)

        # Setup logging
        self.logger = setup_logging(config.train.save_dir)
        self.logger.info(f"Using device: {self.device}")

        # Set seed for reproducibility
        set_seed(config.train.seed)

        # Create model
        self.model = create_model(config.model).to(self.device)
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Create loss function
        self.criterion = create_loss(
            lambda_cls=config.train.lambda_cls,
            lambda_contra=config.train.lambda_contra,
            temperature=config.train.temperature,
            label_smoothing=config.train.label_smoothing,
        )

        # Create optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )

        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            config.data, config.train
        )
        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Val samples: {len(self.val_loader.dataset)}")

        # Create learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.train.patience,
            min_delta=config.train.min_delta,
        )

        # Tracking
        self.best_val_f1 = 0.0
        self.global_step = 0

    def _create_scheduler(self):
        """Create learning rate scheduler"""
        num_training_steps = len(self.train_loader) * self.config.train.num_epochs
        num_warmup_steps = self.config.train.warmup_steps

        if self.config.train.lr_scheduler == "cosine":
            warmup = LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=num_warmup_steps,
            )
            cosine = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps - num_warmup_steps,
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[num_warmup_steps],
            )
        else:
            return LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=num_training_steps,
            )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_contra_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            # Move to device
            text_input_ids = batch["text_input_ids"].to(self.device)
            text_attention_mask = batch["text_attention_mask"].to(self.device)
            speech_mels = batch["speech_mels"].to(self.device)
            block_mask = batch["block_mask"].to(self.device)
            engagement_values = batch["engagement_values"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            outputs = self.model(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                speech_mels=speech_mels,
                block_mask=block_mask,
                engagement_values=engagement_values,
            )

            # Compute loss
            losses = self.criterion(outputs, labels, block_mask)

            # Backward pass
            losses["total"].backward()

            # Gradient clipping
            if self.config.train.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.train.gradient_clip,
                )

            self.optimizer.step()
            self.scheduler.step()

            # Track metrics
            total_loss += losses["total"].item()
            total_cls_loss += losses["classification"].item()
            total_contra_loss += losses["contrastive"].item()

            all_preds.extend(outputs.prob.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{losses['total'].item():.4f}",
                "cls": f"{losses['classification'].item():.4f}",
                "contra": f"{losses['contrastive'].item():.4f}",
            })

            self.global_step += 1

            # Logging
            if self.global_step % self.config.train.log_every == 0:
                self.logger.info(
                    f"Step {self.global_step}: loss={losses['total'].item():.4f}, "
                    f"cls={losses['classification'].item():.4f}, "
                    f"contra={losses['contrastive'].item():.4f}"
                )

        # Compute epoch metrics
        num_batches = len(self.train_loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        metrics = compute_metrics(all_preds, all_labels, self.config.eval.depression_threshold)

        return {
            "loss": total_loss / num_batches,
            "cls_loss": total_cls_loss / num_batches,
            "contra_loss": total_contra_loss / num_batches,
            **metrics,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_disagreements = []

        for batch in tqdm(self.val_loader, desc="Validating"):
            # Move to device
            text_input_ids = batch["text_input_ids"].to(self.device)
            text_attention_mask = batch["text_attention_mask"].to(self.device)
            speech_mels = batch["speech_mels"].to(self.device)
            block_mask = batch["block_mask"].to(self.device)
            engagement_values = batch["engagement_values"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = self.model(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                speech_mels=speech_mels,
                block_mask=block_mask,
                engagement_values=engagement_values,
            )

            # Compute loss
            losses = self.criterion(outputs, labels, block_mask)
            total_loss += losses["total"].item()

            # Collect predictions
            all_preds.extend(outputs.prob.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_disagreements.extend(outputs.mean_disagreement.cpu().numpy())

        # Compute metrics
        num_batches = len(self.val_loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        metrics = compute_metrics(all_preds, all_labels, self.config.eval.depression_threshold)

        # Add disagreement analysis
        all_disagreements = np.array(all_disagreements)
        metrics["mean_disagreement_dep"] = all_disagreements[all_labels.flatten() == 1].mean()
        metrics["mean_disagreement_ctrl"] = all_disagreements[all_labels.flatten() == 0].mean()

        return {
            "loss": total_loss / num_batches,
            **metrics,
        }

    def train(self):
        """Full training loop"""
        self.logger.info("Starting training...")

        for epoch in range(1, self.config.train.num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.logger.info(
                f"Epoch {epoch} Train: "
                f"loss={train_metrics['loss']:.4f}, "
                f"acc={train_metrics['accuracy']:.4f}, "
                f"f1={train_metrics['f1']:.4f}"
            )

            # Validate
            val_metrics = self.validate()
            self.logger.info(
                f"Epoch {epoch} Val: "
                f"loss={val_metrics['loss']:.4f}, "
                f"acc={val_metrics['accuracy']:.4f}, "
                f"f1={val_metrics['f1']:.4f}, "
                f"auc={val_metrics.get('auc_roc', 0):.4f}"
            )

            # Log disagreement analysis
            self.logger.info(
                f"Disagreement - Depression: {val_metrics.get('mean_disagreement_dep', 0):.4f}, "
                f"Control: {val_metrics.get('mean_disagreement_ctrl', 0):.4f}"
            )

            # Save best model
            if val_metrics["f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                    os.path.join(self.config.train.save_dir, "best_model.pt"),
                )
                self.logger.info(f"New best model saved! F1: {self.best_val_f1:.4f}")

            # Periodic checkpoint
            if epoch % self.config.train.save_every == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                    os.path.join(self.config.train.save_dir, f"checkpoint_epoch{epoch}.pt"),
                )

            # Early stopping
            if self.early_stopping(val_metrics["loss"]):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        self.logger.info(f"Training complete! Best validation F1: {self.best_val_f1:.4f}")

        # Final evaluation on test set
        if self.test_loader is not None:
            self.logger.info("Evaluating on test set...")
            # Load best model
            load_checkpoint(
                self.model,
                os.path.join(self.config.train.save_dir, "best_model.pt"),
            )
            test_metrics = self._evaluate_test()
            self.logger.info(f"Test metrics: {test_metrics}")

            # Save test results
            with open(os.path.join(self.config.train.save_dir, "test_results.json"), "w") as f:
                json.dump(test_metrics, f, indent=2)

    @torch.no_grad()
    def _evaluate_test(self) -> Dict[str, float]:
        """Evaluate on test set"""
        self.model.eval()

        all_preds = []
        all_labels = []

        for batch in tqdm(self.test_loader, desc="Testing"):
            text_input_ids = batch["text_input_ids"].to(self.device)
            text_attention_mask = batch["text_attention_mask"].to(self.device)
            speech_mels = batch["speech_mels"].to(self.device)
            block_mask = batch["block_mask"].to(self.device)
            engagement_values = batch["engagement_values"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                speech_mels=speech_mels,
                block_mask=block_mask,
                engagement_values=engagement_values,
            )

            all_preds.extend(outputs.prob.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        return compute_metrics(all_preds, all_labels, self.config.eval.depression_threshold)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Train MDTD model")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs override")
    args = parser.parse_args()

    # Create config with overrides
    overrides = {}
    if args.lr:
        overrides["train.learning_rate"] = args.lr
    if args.batch_size:
        overrides["train.batch_size"] = args.batch_size
    if args.epochs:
        overrides["train.num_epochs"] = args.epochs

    config = get_config(**overrides)

    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.train.save_dir = os.path.join(config.train.save_dir, f"run_{timestamp}")
    os.makedirs(config.train.save_dir, exist_ok=True)

    # Save config
    with open(os.path.join(config.train.save_dir, "config.json"), "w") as f:
        json.dump({
            "model": vars(config.model),
            "data": vars(config.data),
            "train": vars(config.train),
            "eval": vars(config.eval),
        }, f, indent=2, default=str)

    # Train
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
