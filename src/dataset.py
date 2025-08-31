import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import re
from PIL import Image


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
    def __init__(self, images_dir, density_dir, mode="train", patch_size=512, patches_per_plot=20, transform=None):
        self.images_dir = images_dir
        self.density_dir = density_dir
        self.mode = mode
        self.patch_size = patch_size
        self.patches_per_plot = patches_per_plot
        self.transform = transform

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
            return len(self.image_files) * self.patches_per_plot
        else:
            return len(self.image_files)

    def __getitem__(self, idx):
        from PIL import Image
        if self.mode == "train":
            plot_idx = idx // self.patches_per_plot
            img_path = os.path.join(self.images_dir, self.image_files[plot_idx])
            dmap_path = os.path.join(self.density_dir, self.density_files[plot_idx])

            img = load_image(img_path)
            dmap = load_density_map(dmap_path)

            # Resize
            img, dmap = resize_half(img, dmap)
            h, w = img.shape[:2]

            # Random crop
            x = random.randint(0, w - self.patch_size)
            y = random.randint(0, h - self.patch_size)

            img_patch = img[y:y+self.patch_size, x:x+self.patch_size]
            dmap_patch = dmap[y:y+self.patch_size, x:x+self.patch_size]

            # Convert to PIL for transforms
            img_patch = Image.fromarray(img_patch)

            if self.transform:
                img_patch = self.transform(img_patch)
            else:
                img_patch = T.ToTensor()(img_patch)

            dmap_patch = torch.from_numpy(dmap_patch).unsqueeze(0).float()

            return img_patch, dmap_patch

        else:  # val / test
            img_path = os.path.join(self.images_dir, self.image_files[idx])
            dmap_path = os.path.join(self.density_dir, self.density_files[idx])

            img = load_image(img_path)
            dmap = load_density_map(dmap_path)
            img, dmap = resize_half(img, dmap)

            img = Image.fromarray(img)

            if self.transform:
                img_tensor = self.transform(img)
            else:
                img_tensor = T.ToTensor()(img)

            dmap_tensor = torch.from_numpy(dmap).unsqueeze(0).float()

            return img_tensor, dmap_tensor, self.image_files[idx]
