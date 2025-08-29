# src/utils.py
import os
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ------------------------
# Tiled inference (memory safe)
# ------------------------
def tiled_infer_density(model, img_t, tile=1024, overlap=64, device="cuda"):
    """
    Memory-safe tiled inference for a single image tensor.
    Args:
        model: nn.Module, returns [1, H', W'] or [B, 1, H', W']
        img_t: torch.Tensor [3, H, W] (float, 0..1)
        tile: int tile size (pixels)
        overlap: int overlap pixels between tiles
        device: device string
    Returns:
        out: torch.Tensor shape [1, H, W] (float) predicted density map
    """
    model.eval()
    _, H, W = img_t.shape
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    out = torch.zeros((1, H, W), device=dev)
    weight = torch.zeros((1, H, W), device=dev)

    ys = list(range(0, max(H - tile, 0) + 1, tile - overlap))
    xs = list(range(0, max(W - tile, 0) + 1, tile - overlap))
    if len(ys) == 0: ys = [0]
    if len(xs) == 0: xs = [0]
    if ys[-1] + tile < H: ys.append(max(H - tile, 0))
    if xs[-1] + tile < W: xs.append(max(W - tile, 0))

    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = img_t[:, y:y+tile, x:x+tile].unsqueeze(0).to(dev)  # [1,3,h,w]
                pred = model(patch)  # [1,1,h',w'] expected
                # bring pred to the patch size
                pred_resized = F.interpolate(pred, size=(patch.shape[-2], patch.shape[-1]), mode='bilinear', align_corners=False)
                out[:, y:y+tile, x:x+tile] += pred_resized.squeeze(0)
                weight[:, y:y+tile, x:x+tile] += 1.0

    out = out / torch.clamp(weight, min=1e-6)
    return out.cpu()  # [1, H, W]

# ------------------------
# Full-plot evaluation (valid/test)
# ------------------------
def evaluate_fullplots(model, dataloader, device="cuda", tile=1024, overlap=64):
    """
    Run full-plot inference on a dataloader that yields (img_tensor, den_tensor, name)
    Returns:
        mae, gts_list, preds_list
    """
    gts, preds = [], []
    for img_t, den_t, name in tqdm(dataloader, desc="Eval", ncols=80):
        img_t = img_t.squeeze(0)  # [3,H,W]
        den_t = den_t.squeeze(0)  # [1,H,W]
        pred_den = tiled_infer_density(model, img_t, tile=tile, overlap=overlap, device=device)  # [1,H,W]
        gt = float(den_t.sum().item())
        pr = float(pred_den.sum().item())
        gts.append(gt)
        preds.append(pr)
    gts = np.array(gts, dtype=np.float64)
    preds = np.array(preds, dtype=np.float64)
    mae, rmse, rmae, rrmse, r2 = compute_metrics(gts, preds)
    return {"MAE": mae, "RMSE": rmse, "rMAE": rmae, "rRMSE": rrmse, "R2": r2}, gts, preds

# ------------------------
# Metrics
# ------------------------
def compute_metrics(gts, preds):
    """
    gts, preds: numpy arrays (same shape)
    Returns: MAE, RMSE, rMAE, rRMSE, R2
    rMAE/rRMSE are fractional (0..1)
    """
    gts = np.array(gts, dtype=np.float64)
    preds = np.array(preds, dtype=np.float64)
    errs = preds - gts
    mae = float(np.mean(np.abs(errs)))
    rmse = float(np.sqrt(np.mean(errs**2)))
    # relative: divide by gt (avoid divide-by-zero by adding eps)
    eps = 1e-8
    rel = np.abs(errs) / (gts + eps)
    rmae = float(np.mean(rel))
    rrmse = float(np.sqrt(np.mean((errs / (gts + eps))**2)))
    # R^2
    ss_res = float(np.sum((preds - gts)**2))
    ss_tot = float(np.sum((gts - np.mean(gts))**2)) + eps
    r2 = float(1.0 - ss_res / ss_tot)
    return mae, rmse, rmae, rrmse, r2

# ------------------------
# Checkpoint save / load
# ------------------------
def save_checkpoint(state, is_best, out_dir, filename="checkpoint.pth"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(out_dir, "best_model.pth")
        torch.save(state, best_path)

def load_checkpoint(path, model=None, optimizer=None, scheduler=None, device="cuda"):
    ckpt = torch.load(path, map_location=(device if torch.cuda.is_available() else "cpu"))
    if model is not None:
        model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt

# ------------------------
# Simple JSON logger
# ------------------------
def append_log(log_path, entry):
    logs = []
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.append(entry)
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)
