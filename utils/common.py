import torch
import random
import numpy as np
import os
import colorsys
from pathlib import Path


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(device_str='auto'):
    """Get PyTorch device."""
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_str.startswith('cuda'):
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            return torch.device('cpu')
        return torch.device(device_str)
    else:
        return torch.device(device_str)


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """Get model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def save_config(config, path):
    """Save configuration to JSON file."""
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(path):
    """Load configuration from JSON file."""
    import json
    with open(path, 'r') as f:
        return json.load(f)


def format_number(num):
    """Format large numbers with K/M/B suffixes."""
    if num < 1000:
        return str(num)
    elif num < 1000000:
        return ".1f"
    elif num < 1000000000:
        return ".1f"
    else:
        return ".1f"


def create_experiment_dir(base_dir, exp_name=None):
    """Create experiment directory with timestamp."""
    from datetime import datetime

    base_dir = Path(base_dir)
    if exp_name:
        exp_dir = base_dir / exp_name
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir = base_dir / f'exp_{timestamp}'

    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def get_class_weights_from_dataset(dataset, num_classes, ignore_index=None):
    """Compute class weights from dataset for balanced loss."""
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    # Create a temporary dataloader to sample the dataset
    temp_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    class_counts = torch.zeros(num_classes)
    total_pixels = 0

    for batch in temp_loader:
        mask = batch['mask'].squeeze()
        if ignore_index is not None:
            mask = mask[mask != ignore_index]

        # Count pixels per class
        counts = torch.bincount(mask.flatten(), minlength=num_classes)
        class_counts += counts
        total_pixels += mask.numel()

    # Compute weights as inverse frequency
    class_weights = total_pixels / (class_counts * num_classes)
    class_weights = class_weights / class_weights.sum() * num_classes  # Normalize

    # Handle classes with zero count
    class_weights[class_counts == 0] = 0

    return class_weights


def create_distinct_colormap(num_classes):
    """
    Create a colormap with distinct, contrasting colors for each class.
    Uses HSV color space to maximize perceptual distance between colors.
    
    Args:
        num_classes: Number of classes
        
    Returns:
        numpy array of shape (num_classes, 3) with RGB values [0-255]
    """
    colors = []
    
    # Generate colors using HSV space for better distribution
    for i in range(num_classes):
        # Distribute hues evenly around the color wheel
        hue = i / num_classes
        
        # Vary saturation and value to create more distinct colors
        # Use different patterns for better visual separation
        saturation = 0.75 + 0.25 * (i % 3) / 2  # 0.75, 0.875, or 1.0
        value = 0.7 + 0.3 * ((i // 3) % 2)  # 0.7 or 1.0
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        # Convert to 0-255 range
        colors.append([int(c * 255) for c in rgb])
    
    # Ensure class 0 (background) is black for better contrast
    if num_classes > 0:
        colors[0] = [0, 0, 0]
    
    return np.array(colors, dtype=np.uint8)


def create_high_contrast_colormap(num_classes):
    """
    Create a colormap optimized for maximum contrast between adjacent classes.
    Uses predefined high-contrast colors for better visual distinction.
    
    Args:
        num_classes: Number of classes
        
    Returns:
        numpy array of shape (num_classes, 3) with RGB values [0-255]
    """
    # Predefined high-contrast color palette (first 20 colors)
    # These are manually selected for maximum visual distinction
    high_contrast_palette = [
        [0, 0, 0],        # 0: Black (background)
        [255, 0, 0],      # 1: Red
        [0, 255, 0],      # 2: Green
        [0, 0, 255],      # 3: Blue
        [255, 255, 0],    # 4: Yellow
        [255, 0, 255],    # 5: Magenta
        [0, 255, 255],    # 6: Cyan
        [255, 128, 0],    # 7: Orange
        [128, 0, 255],    # 8: Purple
        [255, 192, 203],  # 9: Pink
        [0, 128, 0],      # 10: Dark Green
        [128, 128, 0],    # 11: Olive
        [0, 0, 128],      # 12: Navy
        [128, 0, 0],      # 13: Maroon
        [192, 192, 192],  # 14: Silver
        [128, 128, 128],  # 15: Gray
        [255, 165, 0],    # 16: Orange Red
        [0, 255, 127],    # 17: Spring Green
        [255, 20, 147],   # 18: Deep Pink
        [70, 130, 180],   # 19: Steel Blue
    ]
    
    # Use predefined palette if we have enough colors
    if num_classes <= len(high_contrast_palette):
        colors = high_contrast_palette[:num_classes]
    else:
        # Start with predefined colors
        colors = high_contrast_palette.copy()
        
        # Generate additional colors using HSV with better spacing
        for i in range(len(high_contrast_palette), num_classes):
            # Use golden ratio for better color distribution
            golden_ratio = 0.618033988749895
            hue = (i * golden_ratio) % 1.0
            saturation = 0.6 + 0.4 * (i % 2)
            value = 0.5 + 0.5 * ((i // 2) % 2)
            
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append([int(c * 255) for c in rgb])
    
    return np.array(colors, dtype=np.uint8)


def apply_colormap(mask, colormap):
    """
    Apply colormap to segmentation mask.
    
    Args:
        mask: numpy array of shape (H, W) with class indices
        colormap: numpy array of shape (num_classes, 3) with RGB colors
        
    Returns:
        numpy array of shape (H, W, 3) with RGB values [0-255]
    """
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Handle ignore_index (-100) by making it gray
    valid_mask = mask >= 0
    mask_clamped = np.clip(mask, 0, len(colormap) - 1)
    
    for class_id in range(len(colormap)):
        class_pixels = (mask_clamped == class_id) & valid_mask
        colored_mask[class_pixels] = colormap[class_id]
    
    # Set ignored pixels to gray
    ignored_pixels = ~valid_mask
    colored_mask[ignored_pixels] = [128, 128, 128]
    
    return colored_mask


def get_cityscapes_class_names():
    """Get Cityscapes class names."""
    return [
        "road", "sidewalk", "building", "wall", "fence",
        "pole", "traffic light", "traffic sign", "vegetation", "terrain",
        "sky", "person", "rider", "car", "truck",
        "bus", "train", "motorcycle", "bicycle"
    ]


def save_colored_prediction_with_legend(
    colored_mask, colormap, output_path, class_names=None, legend_width=200
):
    """
    Save colored prediction with a legend showing class colors and names.
    
    Args:
        colored_mask: numpy array (H, W, 3) with RGB values [0-255]
        colormap: numpy array (num_classes, 3) with RGB colors
        output_path: Path to save the image
        class_names: Optional list of class names (default: "Class 0", "Class 1", etc.)
        legend_width: Width of the legend in pixels
    """
    from PIL import Image, ImageDraw, ImageFont
    
    # Convert to PIL Image
    pred_img = Image.fromarray(colored_mask)
    img_width, img_height = pred_img.size
    
    # Create legend
    num_classes = len(colormap)
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    # Calculate legend height
    item_height = 25
    legend_height = num_classes * item_height + 20  # padding
    
    # Create new image with space for legend
    new_width = img_width + legend_width + 10  # 10px gap
    new_height = max(img_height, legend_height)
    combined_img = Image.new('RGB', (new_width, new_height), color='white')
    
    # Paste prediction image
    combined_img.paste(pred_img, (0, 0))
    
    # Draw legend
    draw = ImageDraw.Draw(combined_img)
    
    # Try to use a nice font, fallback to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
    
    legend_x = img_width + 10
    legend_y = 10
    
    # Draw title
    draw.text((legend_x, legend_y), "Class Legend", fill='black', font=font)
    legend_y += 20
    
    # Draw each class
    for i, (color, name) in enumerate(zip(colormap, class_names)):
        y_pos = legend_y + i * item_height
        
        # Draw color box
        box_size = 20
        draw.rectangle(
            [legend_x, y_pos, legend_x + box_size, y_pos + box_size],
            fill=tuple(color.tolist())
        )
        
        # Draw class name
        text_y = y_pos + (box_size - 12) // 2  # Center text vertically
        draw.text(
            (legend_x + box_size + 5, text_y),
            name,
            fill='black',
            font=font
        )
    
    # Save combined image
    combined_img.save(output_path)
