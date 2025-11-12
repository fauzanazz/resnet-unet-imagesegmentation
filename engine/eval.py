import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
import json
from tqdm import tqdm

from utils.metrics import SegmentationMetrics, print_metrics_table


def load_model_for_eval(model, checkpoint_path, device):
    """Load model from checkpoint for evaluation."""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    return model


def evaluate_model(
    model, dataloader, device, num_classes, ignore_index=None,
    save_predictions=False, output_dir=None
):
    """
    Evaluate model on a dataset.

    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        num_classes: Number of classes
        ignore_index: Class index to ignore in metrics
        save_predictions: Whether to save prediction images
        output_dir: Directory to save predictions (if save_predictions=True)

    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    metrics = SegmentationMetrics(num_classes, ignore_index)

    if save_predictions and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_dir = output_dir / 'predictions'
        pred_dir.mkdir(exist_ok=True)

    all_preds = []
    all_targets = []
    sample_paths = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)  # [B, H, W]

            # Update metrics
            metrics.update(preds, masks)

            # Store for potential saving
            if save_predictions:
                all_preds.append(preds.cpu())
                all_targets.append(masks.cpu())
                sample_paths.extend(batch['image_path'])

    # Get final metrics
    results = metrics.get_metrics()

    # Save predictions if requested
    if save_predictions and output_dir and all_preds:
        print(f"Saving predictions to {pred_dir}")
        for i, (pred, target, img_path) in enumerate(zip(
            torch.cat(all_preds, dim=0),
            torch.cat(all_targets, dim=0),
            sample_paths
        )):
            # Save prediction mask
            pred_img = Image.fromarray(pred.numpy().astype(np.uint8))
            pred_filename = f"{Path(img_path).stem}_pred.png"
            pred_img.save(pred_dir / pred_filename)

            # Optionally save ground truth too
            if i < 10:  # Save GT for first 10 samples
                gt_img = Image.fromarray(target.numpy().astype(np.uint8))
                gt_filename = f"{Path(img_path).stem}_gt.png"
                gt_img.save(pred_dir / gt_filename)

    return results


def save_evaluation_results(metrics, output_path, class_names=None):
    """Save evaluation results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Also save a human-readable summary
    summary_path = output_path.parent / 'evaluation_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("EVALUATION SUMMARY\n")
        f.write("="*50 + "\n")
        f.write(".4f")
        f.write(".4f")
        f.write(".4f")
        f.write(f"Number of classes: {len(metrics.get('class_ious', []))}\n")

        if class_names and 'class_ious' in metrics:
            f.write("\nPer-Class IoU:\n")
            for i, (iou, name) in enumerate(zip(metrics['class_ious'], class_names)):
                f.write("2d")

    print(f"Results saved to {output_path}")
    print(f"Summary saved to {summary_path}")


def run_evaluation(
    checkpoint_path, data_dir, split, num_classes, image_size,
    batch_size=8, num_workers=4, device='cuda', ignore_index=None,
    save_predictions=False, output_dir=None, class_names=None
):
    """
    Run complete evaluation pipeline.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Directory containing data
        split: 'val' or 'test'
        num_classes: Number of classes
        image_size: Tuple (H, W) for resizing
        batch_size: Batch size for evaluation
        num_workers: Number of workers for dataloader
        device: Device to run on
        ignore_index: Class index to ignore
        save_predictions: Whether to save prediction masks
        output_dir: Directory to save results
        class_names: List of class names for reporting
    """

    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Import here to avoid circular imports
    from models.resnet_unet import create_resnet_unet
    from data.segmentation_dataset import get_train_val_datasets, get_test_dataset

    # Create model
    model = create_resnet_unet(num_classes)

    # Load checkpoint
    model = load_model_for_eval(model, checkpoint_path, device)

    # Create dataset
    if split == 'test':
        dataset = get_test_dataset(data_dir, image_size, ignore_index)
    else:  # val
        _, dataset = get_train_val_datasets(data_dir, image_size, ignore_index=ignore_index)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Run evaluation
    print(f"Evaluating on {split} set ({len(dataset)} samples)")
    metrics = evaluate_model(
        model, dataloader, device, num_classes, ignore_index,
        save_predictions, output_dir
    )

    # Print results
    print_metrics_table(metrics, class_names)

    # Save results if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / f'{split}_results.json'
        save_evaluation_results(metrics, results_path, class_names)

    return metrics
