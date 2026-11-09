from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F


def fit_log_affine(predictions: list[np.ndarray], targets: list[np.ndarray]) -> tuple[float, float]:
    if not predictions or len(predictions) != len(targets):
        raise ValueError("validation predictions and targets must be non-empty and paired")
    x = np.concatenate([np.log(np.clip(value.ravel(), 1e-4, 1.0)) for value in predictions])
    y = np.concatenate([np.log(np.clip(value.ravel(), 1e-4, 1.0)) for value in targets])
    if x.size > 2_000_000:
        indices = np.linspace(0, x.size - 1, 2_000_000, dtype=np.int64)
        x, y = x[indices], y[indices]
    matrix = np.stack([x, np.ones_like(x)], axis=-1)
    slope, intercept = np.linalg.lstsq(matrix, y, rcond=None)[0]
    return float(np.clip(slope, 0.1, 10.0)), float(np.clip(intercept, -10.0, 10.0))


def apply_log_affine(value: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    corrected = np.exp(slope * np.log(np.clip(value, 1e-4, 1.0)) + intercept)
    return np.clip(corrected, 0.0, 1.0)


def psnr(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    error = (prediction.astype(np.float64) - target.astype(np.float64)) ** 2
    if mask is not None:
        values = error[mask.astype(bool)]
        if values.size == 0:
            return float("nan")
        mse = float(values.mean())
    else:
        mse = float(error.mean())
    return float(-10.0 * math.log10(max(mse, 1e-12)))


def ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    x = torch.as_tensor(prediction, dtype=torch.float32)[None, None]
    y = torch.as_tensor(target, dtype=torch.float32)[None, None]
    coords = torch.arange(11, dtype=torch.float32) - 5
    kernel_1d = torch.exp(-(coords.square()) / (2 * 1.5**2))
    kernel_1d /= kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :])[None, None]
    mu_x = F.conv2d(x, kernel, padding=5)
    mu_y = F.conv2d(y, kernel, padding=5)
    sigma_x = F.conv2d(x * x, kernel, padding=5) - mu_x.square()
    sigma_y = F.conv2d(y * y, kernel, padding=5) - mu_y.square()
    sigma_xy = F.conv2d(x * y, kernel, padding=5) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    value = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return float(value.mean())


def nearest_distances(
    source: np.ndarray, target: np.ndarray, device: torch.device, chunk: int = 512
) -> np.ndarray:
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
    result = []
    for start in range(0, len(source), chunk):
        source_tensor = torch.as_tensor(
            source[start : start + chunk], dtype=torch.float32, device=device
        )
        result.append(torch.cdist(source_tensor, target_tensor).min(dim=1).values.cpu().numpy())
    return np.concatenate(result)


def geometry_metrics(predicted: np.ndarray, target: np.ndarray, device: torch.device) -> dict:
    accuracy = nearest_distances(predicted, target, device)
    completeness = nearest_distances(target, predicted, device)
    result = {
        "accuracy_mean_m": float(accuracy.mean()),
        "accuracy_median_m": float(np.median(accuracy)),
        "completeness_mean_m": float(completeness.mean()),
        "completeness_median_m": float(np.median(completeness)),
        "chamfer_l1_m": float(0.5 * (accuracy.mean() + completeness.mean())),
        "chamfer_l2_m2": float(0.5 * ((accuracy**2).mean() + (completeness**2).mean())),
    }
    for threshold in (0.1, 0.25, 0.5):
        precision = float(np.mean(accuracy <= threshold))
        recall = float(np.mean(completeness <= threshold))
        fscore = 2 * precision * recall / max(precision + recall, 1e-12)
        key = str(threshold).replace(".", "p")
        result[f"precision_{key}m"] = precision
        result[f"recall_{key}m"] = recall
        result[f"fscore_{key}m"] = fscore
    return result | {"accuracy_distances": accuracy, "completeness_distances": completeness}
