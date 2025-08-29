import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import re

# ----------------------
# Natural sorting helper
# ----------------------
def natural_key(filename):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', filename)]

# ----------------------
# Utility Functions
# ----------------------
def load_image(path):
    """Load an image in RGB format."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_density_map(path):
    """Load density map from .npy or image file."""
    if path.endswith(".npy"):
        dmap = np.load(path).astype(np.float32)
    else:
        dmap = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if dmap is None:
            raise FileNotFoundError(f"Density map not found: {path}")
        dmap = dmap.astype(np.float32)
    return dmap

def resize_half(img, dmap):
    """Resize image and density map by 0.5 (keeping density sum same)."""
    h, w = img.shape[:2]
    new_size = (w // 2, h // 2)

    img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    dmap_resized = cv2.resize(dmap, new_size, interpolation=cv2.INTER_LINEAR)

    # Adjust density map values so total count remains correct
    dmap_resized = dmap_resized * (dmap.sum() / (dmap_resized.sum() + 1e-6))
    return img_resized, dmap_resized

# ----------------------
# Dataset Class
# ----------------------
class PlotCounterDataset(Dataset):
    """
    Dataset for PlotCounter
    - Training: returns random patches from resized images
    - Validation/Testing: returns full resized plot
    """
    def __init__(self, images_dir, density_dir, mode="train", patch_size=512, patches_per_plot=20):
        """
        Args:
            images_dir (str): Path to plot images
            density_dir (str): Path to density maps (.npy or image files)
            mode (str): "train" | "val" | "test"
            patch_size (int): Size of cropped patch (default: 512)
            patches_per_plot (int): Number of patches per plot (train mode only)
        """
        self.images_dir = images_dir
        self.density_dir = density_dir
        self.mode = mode
        self.patch_size = patch_size
        self.patches_per_plot = patches_per_plot

        # Collect files
        self.image_files = sorted(
            [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png"))],
            key=natural_key
        )
        self.density_files = sorted(
            [f for f in os.listdir(density_dir) if f.endswith(".npy") or f.endswith((".png", ".jpg"))],
            key=natural_key
        )

        assert len(self.image_files) == len(self.density_files), \
            f"Mismatch between images ({len(self.image_files)}) and density maps ({len(self.density_files)})"

    def __len__(self):
        if self.mode == "train":
            # Each epoch generates (plots × patches_per_plot) samples
            return len(self.image_files) * self.patches_per_plot
        else:
            return len(self.image_files)

    def __getitem__(self, idx):
        if self.mode == "train":
            # Select which plot this idx corresponds to
            plot_idx = idx // self.patches_per_plot
            img_path = os.path.join(self.images_dir, self.image_files[plot_idx])
            dmap_path = os.path.join(self.density_dir, self.density_files[plot_idx])

            img = load_image(img_path)
            dmap = load_density_map(dmap_path)

            # Resize by 0.5
            img, dmap = resize_half(img, dmap)
            h, w = img.shape[:2]

            # Random crop
            x = random.randint(0, w - self.patch_size)
            y = random.randint(0, h - self.patch_size)

            img_patch = img[y:y+self.patch_size, x:x+self.patch_size]
            dmap_patch = dmap[y:y+self.patch_size, x:x+self.patch_size]

            # Convert to tensors
            img_patch = torch.from_numpy(img_patch.transpose(2, 0, 1)).float() / 255.0
            dmap_patch = torch.from_numpy(dmap_patch).unsqueeze(0)

            return img_patch, dmap_patch

        else:  # validation or testing -> full plot
            img_path = os.path.join(self.images_dir, self.image_files[idx])
            dmap_path = os.path.join(self.density_dir, self.density_files[idx])

            img = load_image(img_path)
            dmap = load_density_map(dmap_path)

            img, dmap = resize_half(img, dmap)

            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            dmap_tensor = torch.from_numpy(dmap).unsqueeze(0)

            return img_tensor, dmap_tensor, self.image_files[idx]
