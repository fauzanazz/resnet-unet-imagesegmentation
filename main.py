import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json

from models.resnet_unet import create_resnet_unet
from data.segmentation_dataset import get_train_val_datasets
from engine.train import train_model, get_loss_fn, get_scheduler
from engine.eval import run_evaluation
from utils.common import (
    set_seed, get_device, count_parameters, create_experiment_dir,
    create_distinct_colormap, create_high_contrast_colormap, apply_colormap,
    get_cityscapes_class_names, save_colored_prediction_with_legend
)
from utils.metrics import SegmentationMetrics


def add_train_args(subparsers):
    """Add arguments for train subcommand."""
    parser = subparsers.add_parser('train', help='Train the model')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing train/val data')
    parser.add_argument('--num-classes', type=int, required=True,
                       help='Number of classes for segmentation')
    parser.add_argument('--img-size', type=int, default=512,
                       help='Image size for training (square)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')
    parser.add_argument('--amp', action='store_true',
                       help='Use automatic mixed precision')
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'plateau'], default='cosine',
                       help='Learning rate scheduler')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--ignore-index', type=int, default=None,
                       help='Class index to ignore in loss/metrics')
    parser.add_argument('--class-weights', type=str, default=None,
                       help='Path to class weights JSON or "balanced" for auto-compute')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                       help='Label smoothing factor')
    parser.add_argument('--output-dir', type=str, default='runs/exp',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to train on (auto/cuda/cpu)')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='TensorBoard log directory')


def add_evaluate_args(subparsers):
    """Add arguments for evaluate subcommand."""
    parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing evaluation data')
    parser.add_argument('--split', type=str, choices=['val', 'test'], default='val',
                       help='Data split to evaluate on')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num-classes', type=int, required=True,
                       help='Number of classes')
    parser.add_argument('--img-size', type=int, default=512,
                       help='Image size for evaluation')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--ignore-index', type=int, default=None,
                       help='Class index to ignore in metrics')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save prediction masks')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save results and predictions')
    parser.add_argument('--class-names', type=str, default=None,
                       help='Path to class names JSON')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to evaluate on')


def add_predict_args(subparsers):
    """Add arguments for predict subcommand."""
    parser = subparsers.add_parser('predict', help='Run inference on images')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image file or directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num-classes', type=int, required=True,
                       help='Number of classes')
    parser.add_argument('--img-size', type=int, default=512,
                       help='Image size for prediction')
    parser.add_argument('--output-dir', type=str, default='predictions',
                       help='Directory to save predictions')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for prediction')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to run prediction on')
    parser.add_argument('--colored', action='store_true',
                       help='Save colored predictions with distinct class colors')
    parser.add_argument('--colormap', type=str, choices=['standard', 'high-contrast'], 
                       default='high-contrast',
                       help='Colormap type: standard (HSV-based) or high-contrast (predefined)')
    parser.add_argument('--class-names', type=str, default=None,
                       help='Path to JSON file with class names mapping (e.g., {"0": "road", "1": "sidewalk"})')


def train_command(args):
    """Handle train subcommand."""
    # Set seed and device
    set_seed(args.seed)
    device = get_device(args.device)

    print(f"Training on device: {device}")
    print(f"Random seed: {args.seed}")

    # Create datasets
    train_dataset, val_dataset = get_train_val_datasets(
        args.data_dir, (args.img_size, args.img_size),
        augment=True, ignore_index=args.ignore_index
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Create model
    model = create_resnet_unet(args.num_classes)
    model.to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Setup loss function
    class_weights = None
    if args.class_weights == 'balanced':
        from utils.common import get_class_weights_from_dataset
        class_weights = get_class_weights_from_dataset(train_dataset, args.num_classes, args.ignore_index)
        class_weights = class_weights.to(device)
        print(f"Using balanced class weights: {class_weights}")
    elif args.class_weights:
        with open(args.class_weights, 'r') as f:
            weights_dict = json.load(f)
            class_weights = torch.tensor(list(weights_dict.values())).to(device)
        print(f"Using custom class weights: {class_weights}")

    loss_fn = get_loss_fn(
        loss_type='ce',
        class_weights=class_weights,
        ignore_index=args.ignore_index,
        label_smoothing=args.label_smoothing
    )

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(args.scheduler, optimizer, args.epochs)

    # Setup metrics function
    def metrics_fn(preds, targets):
        metrics = SegmentationMetrics(args.num_classes, args.ignore_index)
        metrics.update(preds, targets)
        return metrics.get_metrics()

    # Create output directory
    output_dir = create_experiment_dir(args.output_dir)

    # Save training config
    config = vars(args)
    config['output_dir'] = str(output_dir)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Output directory: {output_dir}")

    # Train model
    best_miou = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        output_dir=output_dir,
        checkpoint_path=args.resume,
        use_amp=args.amp,
        metrics_fn=metrics_fn,
        log_dir=args.log_dir
    )

    print(f"Training completed! Best mIoU: {best_miou:.4f}")


def evaluate_command(args):
    """Handle evaluate subcommand."""
    device = get_device(args.device)

    # Load class names if provided
    class_names = None
    if args.class_names:
        with open(args.class_names, 'r') as f:
            class_names = json.load(f)

    # Run evaluation
    metrics = run_evaluation(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        num_classes=args.num_classes,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        ignore_index=args.ignore_index,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir,
        class_names=class_names
    )

    return metrics


def predict_command(args):
    """Handle predict subcommand."""
    import torchvision.transforms as T
    from PIL import Image
    import numpy as np

    device = get_device(args.device)

    # Load model
    model = create_resnet_unet(args.num_classes)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Create colormap if colored output is requested
    colormap = None
    class_names = None
    if args.colored:
        if args.colormap == 'high-contrast':
            colormap = create_high_contrast_colormap(args.num_classes)
            print(f"Using high-contrast colormap for {args.num_classes} classes")
        else:
            colormap = create_distinct_colormap(args.num_classes)
            print(f"Using standard colormap for {args.num_classes} classes")
        
        # Get class names
        if args.class_names:
            with open(args.class_names, 'r') as f:
                names_dict = json.load(f)
                class_names = [names_dict.get(str(i), f"Class {i}") for i in range(args.num_classes)]
        elif args.num_classes == 19:
            # Default to Cityscapes class names
            class_names = get_cityscapes_class_names()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get input files
    input_path = Path(args.input)
    if input_path.is_file():
        image_files = [input_path]
    else:
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(ext))

    print(f"Found {len(image_files)} images to process")

    # Process images
    transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
    ])

    with torch.no_grad():
        for img_path in image_files:
            # Load and preprocess image
            image = Image.open(img_path).convert('RGB')
            original_size = image.size  # (W, H)

            input_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, H, W]

            # Run inference
            output = model(input_tensor)  # [1, num_classes, H, W]
            pred = torch.argmax(output, dim=1).squeeze(0)  # [H, W]

            # Resize prediction back to original size
            pred_pil = T.ToPILImage()(pred.byte())
            pred_resized = pred_pil.resize(original_size, Image.NEAREST)

            # Save raw prediction (class indices)
            output_path = output_dir / f"{img_path.stem}_pred.png"
            pred_resized.save(output_path)
            print(f"Saved prediction: {output_path}")

            # Save colored prediction if requested
            if args.colored and colormap is not None:
                pred_np = np.array(pred_resized)
                colored_mask = apply_colormap(pred_np, colormap)
                
                # Save with legend
                colored_output_path = output_dir / f"{img_path.stem}_pred_colored.png"
                save_colored_prediction_with_legend(
                    colored_mask, colormap, colored_output_path, class_names
                )
                print(f"Saved colored prediction with legend: {colored_output_path}")

    print(f"Predictions saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='ResNet50-UNet for Semantic Segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Add subcommands
    add_train_args(subparsers)
    add_evaluate_args(subparsers)
    add_predict_args(subparsers)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute command
    if args.command == 'train':
        train_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    elif args.command == 'predict':
        predict_command(args)


if __name__ == "__main__":
    main()
