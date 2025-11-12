import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
import random
from torchvision import transforms as T
from torchvision.transforms import functional as TF


class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation with images and masks."""

    def __init__(self, image_dir, mask_dir, image_size=None, augment=False, ignore_index=None, dataset_type='generic'):
        """
        Args:
            image_dir: Path to directory containing images
            mask_dir: Path to directory containing masks
            image_size: Tuple (H, W) to resize images/masks to, or None for no resize
            augment: Whether to apply data augmentations
            ignore_index: Class index to ignore in loss/metrics (e.g., void/background)
            dataset_type: 'generic' or 'cityscapes'
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.augment = augment
        self.ignore_index = ignore_index
        self.dataset_type = dataset_type

        # Find matching image/mask pairs
        self.image_paths = []
        self.mask_paths = []

        if dataset_type == 'cityscapes':
            self._load_cityscapes_pairs()
        else:
            self._load_generic_pairs()

        print(f"Found {len(self.image_paths)} image-mask pairs")

    def _load_generic_pairs(self):
        """Load image-mask pairs for generic dataset format."""
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        mask_extensions = {'.png', '.tiff'}  # Masks typically PNG or TIFF

        # Get all image files
        for ext in image_extensions:
            self.image_paths.extend(self.image_dir.glob(f'**/*{ext}'))

        # Get all mask files
        for ext in mask_extensions:
            self.mask_paths.extend(self.mask_dir.glob(f'**/*{ext}'))

        # Sort for consistent ordering
        self.image_paths.sort()
        self.mask_paths.sort()

        # Verify matching pairs
        image_names = {p.stem for p in self.image_paths}
        mask_names = {p.stem for p in self.mask_paths}

        if image_names != mask_names:
            missing_masks = image_names - mask_names
            missing_images = mask_names - image_names
            raise ValueError(
                f"Image-mask mismatch! Missing masks for: {missing_masks}, "
                f"Missing images for: {missing_images}"
            )

    def _load_cityscapes_pairs(self):
        """Load image-mask pairs for Cityscapes dataset format."""
        # Cityscapes structure:
        # Images: leftImg8bit/{split}/{city}/{city}_{frame}_leftImg8bit.png
        # Masks: gtFine/{split}/{city}/{city}_{frame}_gtFine_labelIds.png

        # Find all image files
        self.image_paths = list(self.image_dir.glob('**/*_leftImg8bit.png'))

        # Create corresponding mask paths
        for img_path in self.image_paths:
            # Convert image path to mask path
            # e.g., leftImg8bit/train/city1/city1_000001_leftImg8bit.png
            # -> gtFine/train/city1/city1_000001_gtFine_labelIds.png
            parts = img_path.parts
            # Find 'leftImg8bit' in path and replace with 'gtFine'
            leftimg_idx = parts.index('leftImg8bit')
            mask_parts = list(parts)
            mask_parts[leftimg_idx] = 'gtFine'

            # Replace filename suffix
            mask_filename = parts[-1].replace('_leftImg8bit.png', '_gtFine_labelIds.png')
            mask_parts[-1] = mask_filename

            mask_path = Path(*mask_parts)
            self.mask_paths.append(mask_path)

        # Verify all mask files exist
        missing_masks = []
        for mask_path in self.mask_paths:
            if not mask_path.exists():
                missing_masks.append(mask_path)

        if missing_masks:
            raise ValueError(f"Missing mask files: {missing_masks[:5]}...")  # Show first 5

        # Sort for consistent ordering
        combined = sorted(zip(self.image_paths, self.mask_paths))
        self.image_paths, self.mask_paths = zip(*combined)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)

        # Convert mask to numpy for processing
        mask = np.array(mask)

        # Apply augmentations if enabled
        if self.augment:
            image, mask = self._apply_augmentations(image, mask)

        # Resize if specified
        if self.image_size is not None:
            image = TF.resize(image, self.image_size, interpolation=T.InterpolationMode.BILINEAR)
            # Use nearest neighbor for masks to preserve class boundaries
            mask_pil = Image.fromarray(mask)
            mask_pil = TF.resize(mask_pil, self.image_size, interpolation=T.InterpolationMode.NEAREST)
            mask = np.array(mask_pil)

        # Convert to tensors
        image = TF.to_tensor(image)  # [3, H, W], float32, 0-1
        mask = torch.from_numpy(mask).long()  # [H, W], int64

        # Apply ignore_index if specified
        if self.ignore_index is not None:
            mask[mask == self.ignore_index] = -1  # Will be ignored in loss

        return {
            'image': image,
            'mask': mask,
            'image_path': str(image_path),
            'mask_path': str(mask_path)
        }

    def _apply_augmentations(self, image, mask):
        """Apply basic augmentations to image and mask."""
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = np.flip(mask, axis=1)

        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = np.flip(mask, axis=0)

        # Random rotation (90, 180, 270 degrees to preserve mask integrity)
        rotations = [0, 90, 180, 270]
        rotation = random.choice(rotations)
        if rotation > 0:
            image = TF.rotate(image, rotation)
            mask = np.rot90(mask, k=rotation//90)

        return image, mask


def get_train_val_datasets(data_dir, image_size, val_split=0.2, augment=True, ignore_index=None):
    """
    Create train and validation datasets from a data directory.

    Args:
        data_dir: Path containing 'train' and 'val' subdirs, or just 'train' for split
        image_size: Tuple (H, W) for resizing
        val_split: Fraction of train data to use for validation if no val dir
        augment: Whether to apply augmentations to training set
        ignore_index: Class index to ignore

    Returns:
        train_dataset, val_dataset
    """
    data_dir = Path(data_dir)

    # Auto-detect dataset type
    dataset_type = _detect_dataset_type(data_dir)

    # Check for existing train/val split
    if dataset_type == 'cityscapes':
        # Cityscapes structure
        train_image_dir = data_dir / 'leftImg8bit' / 'train'
        train_mask_dir = data_dir / 'gtFine' / 'train'
        val_image_dir = data_dir / 'leftImg8bit' / 'val'
        val_mask_dir = data_dir / 'gtFine' / 'val'

        if train_image_dir.exists() and val_image_dir.exists():
            train_dataset = SegmentationDataset(
                train_image_dir, train_mask_dir,
                image_size=image_size, augment=augment, ignore_index=ignore_index,
                dataset_type='cityscapes'
            )
            val_dataset = SegmentationDataset(
                val_image_dir, val_mask_dir,
                image_size=image_size, augment=False, ignore_index=ignore_index,
                dataset_type='cityscapes'
            )
        else:
            raise ValueError(f"Cityscapes train/val directories not found in {data_dir}")
    else:
        # Generic structure
        train_dir = data_dir / 'train'
        val_dir = data_dir / 'val'

        if train_dir.exists() and val_dir.exists():
            # Use existing split
            train_dataset = SegmentationDataset(
                train_dir / 'images', train_dir / 'masks',
                image_size=image_size, augment=augment, ignore_index=ignore_index,
                dataset_type='generic'
            )
            val_dataset = SegmentationDataset(
                val_dir / 'images', val_dir / 'masks',
                image_size=image_size, augment=False, ignore_index=ignore_index,
                dataset_type='generic'
            )
        elif train_dir.exists():
            # Split existing train data
            train_dataset_full = SegmentationDataset(
                train_dir / 'images', train_dir / 'masks',
                image_size=image_size, augment=False, ignore_index=ignore_index,
                dataset_type='generic'
            )

            # Split indices
            n_total = len(train_dataset_full)
            n_val = int(n_total * val_split)
            n_train = n_total - n_val

            indices = list(range(n_total))
            random.seed(42)  # For reproducible splits
            random.shuffle(indices)

            train_indices = indices[:n_train]
            val_indices = indices[n_train:]

            # Create subset datasets
            train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
            val_dataset = torch.utils.data.Subset(train_dataset_full, val_indices)

            # Re-enable augmentations for training subset
            train_dataset.dataset.augment = augment
        else:
            raise ValueError(f"No train directory found in {data_dir}")

    return train_dataset, val_dataset


def _detect_dataset_type(data_dir):
    """Auto-detect dataset type based on directory structure."""
    data_dir = Path(data_dir)

    # Check for Cityscapes structure
    if (data_dir / 'leftImg8bit').exists() and (data_dir / 'gtFine').exists():
        return 'cityscapes'

    # Default to generic
    return 'generic'


def get_test_dataset(data_dir, image_size, ignore_index=None):
    """Create test dataset."""
    data_dir = Path(data_dir)

    # Auto-detect dataset type
    dataset_type = _detect_dataset_type(data_dir)

    if dataset_type == 'cityscapes':
        test_image_dir = data_dir / 'leftImg8bit' / 'test'
        test_mask_dir = data_dir / 'gtFine' / 'test'

        if not test_image_dir.exists():
            raise ValueError(f"No Cityscapes test directory found in {data_dir}")

        return SegmentationDataset(
            test_image_dir, test_mask_dir,
            image_size=image_size, augment=False, ignore_index=ignore_index,
            dataset_type='cityscapes'
        )
    else:
        test_dir = data_dir / 'test'

        if not test_dir.exists():
            raise ValueError(f"No test directory found in {data_dir}")

        return SegmentationDataset(
            test_dir / 'images', test_dir / 'masks',
            image_size=image_size, augment=False, ignore_index=ignore_index,
            dataset_type='generic'
        )
