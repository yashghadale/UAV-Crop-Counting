# src/train.py
import os
import argparse
import time
import json
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import SGD
import torchvision.transforms as T
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR



# project imports (assumes src is in python path or run from project root)
from dataset import PlotCounterDataset
from model import create_model
from utils import tiled_infer_density, evaluate_fullplots, save_checkpoint, append_log

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="/content/plotcounter", help="project root")
    p.add_argument("--train-images", type=str, default=None)
    p.add_argument("--train-dens", type=str, default=None)
    p.add_argument("--val-images", type=str, default=None)
    p.add_argument("--val-dens", type=str, default=None)
    p.add_argument("--test-images", type=str, default=None)
    p.add_argument("--test-dens", type=str, default=None)

    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--milestones", nargs="+", type=int, default=[200, 400])
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--patches-per-plot", type=int, default=20)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save-dir", type=str, default="runs")
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    n = 0
    for i, (imgs, dens) in enumerate(tqdm(loader, desc="Training", leave=False)):
        imgs = imgs.to(device)
        dens = dens.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        if preds.shape[-2:] != dens.shape[-2:]:
            preds = torch.nn.functional.interpolate(preds, size=dens.shape[-2:], mode='bilinear', align_corners=False)
        loss = criterion(preds, dens)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return running_loss / max(n, 1)

def main():
    args = parse_args()

    # Configure paths (if not provided, use defaults inside root)
    def default_join(rel): return os.path.join(args.root, rel)
    train_img = args.train_images or default_join("data/images/train")
    train_den = args.train_dens or default_join("data/density_maps/train")
    val_img = args.val_images or default_join("data/images/val")
    val_den = args.val_dens or default_join("data/density_maps/val")
    test_img = args.test_images or default_join("data/images/test")
    test_den = args.test_dens or default_join("data/density_maps/test")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # Datasets & loaders
        # Datasets & loaders
    train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),   # <-- must be last
    ])

# Validation: no augmentation, just tensor conversion
    val_transform = T.Compose([
    T.ToTensor()
    ])

# Testing: same as validation
    test_transform = T.Compose([
    T.ToTensor()
    ])

    train_ds = PlotCounterDataset(train_img, train_den, mode="train",
                              patch_size=args.patch_size,
                              patches_per_plot=args.patches_per_plot,
                              transform=train_transform)

    val_ds = PlotCounterDataset(val_img, val_den, mode="val", transform=val_transform)
    test_ds = PlotCounterDataset(test_img, test_den, mode="test", transform=test_transform)


    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)

    # Model, loss, optimizer, scheduler
    # 1. Create model
    model = create_model(in_ch=3, base=32, device=device).to(device)

# 2. Define loss function
    criterion = nn.L1Loss(reduction="mean")

# 3. Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

# 4. Scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)  # use args.epochs instead of hard-coded 50



    best_mae = float("inf")
    log_path = os.path.join(args.save_dir, "train_log.json")
    patience = 20
    counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        print(f"[Epoch {epoch}] LR: {scheduler.get_last_lr()[0]:.6f}")  # log current LR
        scheduler.step()

        # Validation
        val_metrics, _, _ = evaluate_fullplots(model, val_loader, device=device, tile=1024, overlap=64)
        val_mae = float(val_metrics["MAE"])

        is_best = val_mae < best_mae
        if is_best:
            best_mae = val_mae
            counter = 0

            # Save only best checkpoint
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_mae": best_mae
            }
            save_checkpoint(ckpt, is_best, args.save_dir, filename="best_model.pth")
            print(f"✅ Saved new best model at epoch {epoch} with MAE={best_mae:.2f}")
        else:
            counter += 1
            if counter >= patience:
                print(f"⏹ Early stopping triggered at epoch {epoch}!")
                break

        # Log entry (always log, even if not best)
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mae": val_mae,
            "best_mae": best_mae,
            "time": time.time() - t0
        }
        append_log(log_path, entry)
        print(f"[E{epoch}] train_loss={train_loss:.4f} val_MAE={val_mae:.2f} best_MAE={best_mae:.2f} time={entry['time']:.1f}s")

    # After training: evaluate test set with best model if exists
    best_path = os.path.join(args.save_dir, "best_model.pth")
    if os.path.exists(best_path):
        print("Loading best model:", best_path)
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])

        test_metrics, gts, preds = evaluate_fullplots(model, test_loader, device=device, tile=1024, overlap=64)
        print("Test metrics:", test_metrics)
        # Save final test results
        with open(os.path.join(args.save_dir, "test_results.json"), "w") as f:
            json.dump({"test_metrics": test_metrics, "gts": gts.tolist(), "preds": preds.tolist()}, f, indent=2)
    else:
        print("No best model found (best_model.pth missing).")

if __name__ == "__main__":
    main()
