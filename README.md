# ResNet50-UNet for Semantic Segmentation

A PyTorch implementation of ResNet50-UNet for multi-class semantic segmentation with CLI training/evaluation and a self-contained Jupyter notebook.

## Features

- **ResNet50 Encoder**: Pretrained on ImageNet for better feature extraction
- **UNet Decoder**: Skip connections for precise localization
- **Multi-class Segmentation**: Support for any number of classes
- **CLI Interface**: Easy training, evaluation, and inference
- **Mixed Precision**: Automatic mixed precision training (AMP)
- **Comprehensive Metrics**: mIoU, pixel accuracy, per-class IoU
- **Data Augmentation**: Random flips and rotations
- **Self-contained Notebook**: Complete training code in one notebook

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

### Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pillow tqdm matplotlib tensorboard
```

Or using the provided pyproject.toml:

```bash
pip install -e .
```

## Data Preparation

### Supported Dataset Formats

This implementation supports multiple dataset formats:

#### 1. Generic Format (Default)

Organize your data as follows:

```
data/
├── train/
│   ├── images/
│   │   ├── 0001.png
│   │   ├── 0002.jpg
│   │   └── ...
│   └── masks/
│       ├── 0001.png
│       ├── 0002.png
│       └── ...
└── val/
    ├── images/
    │   ├── 0001.png
    │   ├── 0002.jpg
    │   └── ...
    └── masks/
        ├── 0001.png
        ├── 0002.png
        └── ...
```

#### 2. Cityscapes Dataset

Download the Cityscapes dataset from https://www.cityscapes-dataset.net/

The expected structure is:

```
cityscapes/
├── leftImg8bit/
│   ├── train/
│   │   ├── city1/
│   │   │   ├── city1_000001_leftImg8bit.png
│   │   │   ├── city1_000002_leftImg8bit.png
│   │   │   └── ...
│   │   └── city2/
│   │       └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
└── gtFine/
    ├── train/
    │   ├── city1/
    │   │   ├── city1_000001_gtFine_labelIds.png
    │   │   ├── city1_000002_gtFine_labelIds.png
    │   │   └── ...
    │   └── city2/
    │       └── ...
    ├── val/
    │   └── ...
    └── test/
        └── ...
```

For Cityscapes training, use:

```bash
python main.py train --data-dir cityscapes --num-classes 19 --img-size 512 --batch-size 4 --epochs 50 --ignore-index 255 --output-dir runs/cityscapes_exp1
```

**Note**: Cityscapes has 19 semantic classes (plus void class 255). The model will be trained to predict 19 classes. The system automatically detects Cityscapes structure when `leftImg8bit` and `gtFine` directories are present.

### Data Format

- **Images**: RGB images in PNG, JPG, JPEG, BMP, or TIFF format
- **Masks**: Single-channel masks with integer class IDs
  - Generic: (0, 1, 2, ..., num_classes-1)
  - Cityscapes: Standard Cityscapes label IDs (0-18 for valid classes, 255 for void)
- **Naming**: Image and mask filenames must match exactly (stem should be identical)
- **Size**: Images and masks can be any size; they will be resized during training

### Optional: Test Set

For evaluation on a separate test set:

```
data/
└── test/
    ├── images/
    └── masks/
```

### Cityscapes Class Information

Cityscapes has 19 semantic classes:

| ID | Class Name | Trainable |
|----|------------|-----------|
| 0  | road       | ✓         |
| 1  | sidewalk   | ✓         |
| 2  | building   | ✓         |
| 3  | wall       | ✓         |
| 4  | fence      | ✓         |
| 5  | pole       | ✓         |
| 6  | traffic light | ✓       |
| 7  | traffic sign | ✓        |
| 8  | vegetation | ✓         |
| 9  | terrain    | ✓         |
| 10 | sky        | ✓         |
| 11 | person     | ✓         |
| 12 | rider      | ✓         |
| 13 | car        | ✓         |
| 14 | truck      | ✓         |
| 15 | bus        | ✓         |
| 16 | train      | ✓         |
| 17 | motorcycle | ✓         |
| 18 | bicycle    | ✓         |
| 255| void       | ✗         |

For Cityscapes training, use `--ignore-index 255` to ignore void pixels.

## Training

### Basic Training

```bash
python main.py train --data-dir data --num-classes 4 --img-size 512 --batch-size 8 --epochs 50 --output-dir runs/exp1
```

### Advanced Training Options

```bash
python main.py train \\
    --data-dir data \\
    --num-classes 4 \\
    --img-size 512 \\
    --batch-size 8 \\
    --epochs 100 \\
    --lr 3e-4 \\
    --weight-decay 1e-4 \\
    --amp \\
    --scheduler cosine \\
    --class-weights balanced \\
    --label-smoothing 0.1 \\
    --output-dir runs/exp1 \\
    --seed 42 \\
    --device cuda
```

### Training with Custom Class Weights

Create a JSON file with class weights:

```json
{
    "0": 1.0,
    "1": 2.5,
    "2": 1.8,
    "3": 3.2
}
```

Then use:

```bash
python main.py train --data-dir data --num-classes 4 --class-weights weights.json --output-dir runs/exp1
```

## Evaluation

### Evaluate on Validation Set

```bash
python main.py evaluate \\
    --data-dir data \\
    --split val \\
    --checkpoint runs/exp1/best_miou.pt \\
    --num-classes 4 \\
    --img-size 512 \\
    --output-dir results/val
```

### Evaluate on Test Set

```bash
python main.py evaluate \\
    --data-dir data \\
    --split test \\
    --checkpoint runs/exp1/best_miou.pt \\
    --num-classes 4 \\
    --img-size 512 \\
    --save-predictions \\
    --output-dir results/test
```

### Evaluation Output

- Console output with metrics table
- `results.json`: Detailed metrics in JSON format
- `evaluation_summary.txt`: Human-readable summary
- `predictions/`: Predicted masks (if `--save-predictions` used)

## Inference

### Predict on Single Image

```bash
python main.py predict \\
    --input image.jpg \\
    --checkpoint runs/exp1/best_miou.pt \\
    --num-classes 4 \\
    --img-size 512 \\
    --output-dir predictions
```

### Predict on Directory

```bash
python main.py predict \\
    --input images/ \\
    --checkpoint runs/exp1/best_miou.pt \\
    --num-classes 4 \\
    --img-size 512 \\
    --output-dir predictions
```

## Training Tips

### Memory Issues

- Reduce `--batch-size` if you get CUDA out of memory errors
- Use `--img-size 256` or smaller for limited GPU memory
- Disable AMP with `--no-amp` if issues persist

### Training Speed

- Use `--amp` for 2x faster training with minimal accuracy loss
- Increase `--num-workers` if you have multiple CPU cores
- Use larger batch sizes if you have more GPU memory

### Improving Performance

- Use data augmentation by ensuring your train/val split has augment=True
- Try `--class-weights balanced` for imbalanced datasets
- Use `--label-smoothing 0.1` for better generalization
- Experiment with different learning rates (1e-4 to 1e-3)

### Monitoring Training

- Check `runs/exp1/metrics.csv` for training curves
- Use `--log-dir runs/exp1/tensorboard` for TensorBoard logging
- Best model is saved as `best_miou.pt`

## Using the Training Notebook

For a more interactive experience, use the self-contained Jupyter notebook:

```bash
jupyter notebook notebooks/ResNet50_UNet_Training.ipynb
```

The notebook includes:
- All model and training code in one place
- Interactive configuration
- Training visualization
- Evaluation and inference examples

## Model Architecture

- **Encoder**: ResNet50 with pretrained ImageNet weights
- **Decoder**: 4 upsampling blocks with skip connections
- **Output**: Logits for each class (no softmax)
- **Input Size**: Any size, outputs same size as input after final upsampling

## Metrics

- **mIoU**: Mean Intersection over Union across all classes
- **Pixel Accuracy**: Overall pixel-wise accuracy
- **Per-class IoU**: IoU for each individual class

## Troubleshooting

### Common Issues

1. **"Image-mask mismatch"**: Ensure image and mask filenames match exactly
2. **CUDA out of memory**: Reduce batch size or image size
3. **Poor performance**: Check data format, try balanced class weights
4. **Training doesn't start**: Ensure data directory structure is correct

### Getting Help

- Check the metrics.csv file for training progress
- Verify your data format with a small test
- Use the notebook for debugging data loading

## License

This project is open source. Feel free to use and modify.

## Contributing

Contributions welcome! Please ensure new code follows the clean code principles used in this project.
