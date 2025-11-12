import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.amp as amp
from pathlib import Path
import json
import csv
from datetime import datetime
from tqdm import tqdm
import numpy as np


def get_loss_fn(loss_type='ce', class_weights=None, ignore_index=None, label_smoothing=0.0):
    """Get loss function with specified parameters."""
    if loss_type == 'ce':
        loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index if ignore_index != -1 else -100,  # PyTorch uses -100
            label_smoothing=label_smoothing
        )
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    return loss_fn


def get_scheduler(scheduler_type, optimizer, epochs=None, T_max=None):
    """Get learning rate scheduler."""
    if scheduler_type == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=T_max or epochs)
    elif scheduler_type == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    else:
        return None


def save_checkpoint(model, optimizer, scheduler, epoch, best_miou, checkpoint_path, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_miou': best_miou
    }

    torch.save(checkpoint, checkpoint_path)
    if is_best:
        best_path = checkpoint_path.parent / 'best_miou.pt'
        torch.save(checkpoint, best_path)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model checkpoint."""
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint {checkpoint_path} not found, starting from scratch")
        return 0, 0.0

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    best_miou = checkpoint.get('best_miou', 0.0)

    print(f"Loaded checkpoint from epoch {epoch}, best mIoU: {best_miou:.4f}")
    return epoch, best_miou


def train_epoch(model, dataloader, loss_fn, optimizer, device, scaler=None, scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        optimizer.zero_grad()

        # Forward pass with optional AMP
        if scaler:
            with amp.autocast(device_type=device.type):
                outputs = model(images)
                loss = loss_fn(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches

    # Step scheduler if it's cosine (per epoch)
    if scheduler and isinstance(scheduler, CosineAnnealingLR):
        scheduler.step()

    return avg_loss


def validate_epoch(model, dataloader, loss_fn, device, metrics_fn):
    """Validate for one epoch and compute metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            total_loss += loss.item()

            # Get predictions
            preds = torch.argmax(outputs, dim=1)  # [B, H, W]

            # Collect for metrics
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

    avg_loss = total_loss / len(dataloader)

    # Compute metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = metrics_fn(all_preds, all_targets)

    return avg_loss, metrics


def train_model(
    model, train_loader, val_loader, loss_fn, optimizer, scheduler,
    device, num_epochs, output_dir, checkpoint_path=None, use_amp=False,
    metrics_fn=None, log_dir=None
):
    """
    Main training function.

    Args:
        model: PyTorch model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        loss_fn: Loss function
        optimizer: Optimizer
        scheduler: LR scheduler (optional)
        device: Device to train on
        num_epochs: Number of epochs
        output_dir: Directory to save checkpoints and logs
        checkpoint_path: Path to resume from (optional)
        use_amp: Whether to use automatic mixed precision
        metrics_fn: Function to compute validation metrics
        log_dir: Directory for TensorBoard logs (optional)
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize scaler for AMP
    scaler = amp.GradScaler() if use_amp else None

    # Resume from checkpoint if provided
    start_epoch = 0
    best_miou = 0.0

    if checkpoint_path:
        start_epoch, best_miou = load_checkpoint(
            model, optimizer, scheduler, checkpoint_path
        )

    # Setup logging
    metrics_file = output_dir / 'metrics.csv'
    fieldnames = ['epoch', 'train_loss', 'val_loss', 'miou', 'pixel_acc'] + \
                 [f'class_{i}_iou' for i in range(metrics_fn.num_classes)] if metrics_fn else []

    with open(metrics_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler, scheduler
        )

        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, loss_fn, device, metrics_fn
        )

        # Extract metrics
        miou = val_metrics.get('miou', 0.0)
        pixel_acc = val_metrics.get('pixel_acc', 0.0)
        class_ious = val_metrics.get('class_ious', [])

        # Update scheduler if plateau-based
        if scheduler and isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(miou)

        # Save best model
        is_best = miou > best_miou
        if is_best:
            best_miou = miou

        # Save checkpoint
        checkpoint_path = output_dir / 'last.pt'
        save_checkpoint(model, optimizer, scheduler, epoch + 1, best_miou, checkpoint_path, is_best)

        # Log metrics
        metrics_row = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'miou': miou,
            'pixel_acc': pixel_acc
        }

        for i, iou in enumerate(class_ious):
            metrics_row[f'class_{i}_iou'] = iou

        with open(metrics_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(metrics_row)

        # Print progress
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"mIoU: {miou:.4f}, Pixel Acc: {pixel_acc:.4f}")

        # Log to TensorBoard if available
        if log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(log_dir)
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Metrics/mIoU', miou, epoch)
                writer.add_scalar('Metrics/Pixel_Acc', pixel_acc, epoch)
                writer.close()
            except ImportError:
                pass  # TensorBoard not available

    print(f"\nTraining complete! Best mIoU: {best_miou:.4f}")
    return best_miou
