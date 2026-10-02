"""Image-only skyline with local linear RGB tracking; research, not semantics."""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from automatic_sky_mask import Config, skyline_polygon
from smartphone_gate import require


@dataclass(frozen=True)
class GradientConfig(Config):
    history_rows: int = 12
    seed_fit_iterations: int = 3


def seed_plane(seed, config):
    """Robust plane inside the seed band only; never extrapolate it to the horizon."""
    height, width = seed.shape[:2]
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    design = np.stack([np.ones_like(x), x, y], axis=-1).reshape(-1, 3)
    values = seed.reshape(-1, 3)
    selected = np.ones(len(values), dtype=bool)
    for _ in range(config.seed_fit_iterations):
        coefficients, _, rank, _ = np.linalg.lstsq(design[selected], values[selected], rcond=None)
        require(rank == 3, 'degenerate seed plane')
        residual = values-design@coefficients
        center = np.median(residual[selected], axis=0)
        sigma = np.maximum(config.noise_floor_rgb,
                           1.4826*np.median(np.abs(residual[selected]-center), axis=0))
        candidate = np.max(np.abs(residual-center)/sigma, axis=1) <= config.color_sigma
        # Avoid an underdetermined fit on a pathological seed band.
        if candidate.sum() < len(values)//2:
            break
        selected = candidate
    prediction = (design@coefficients+center).reshape(seed.shape)
    incompatible = np.max(np.abs(seed-prediction)/sigma, axis=2) > config.color_sigma
    return prediction, incompatible, sigma


def track_columns(rgb, seed_prediction, seed_bad, config):
    """Rejected rows do not update history; three consecutive rejections stop it."""
    height, width = rgb.shape[:2]
    seed_rows = len(seed_prediction)
    n = config.consecutive_rows
    sustained = np.ones((seed_rows-n+1, width), dtype=bool)
    for offset in range(n):
        sustained &= seed_bad[offset:seed_rows-n+1+offset]
    boundary = np.where(sustained.any(axis=0), sustained.argmax(axis=0), height)
    active = boundary == height
    window = config.history_rows
    # Only isolated seed outliers are imputed; sustained ones already stop a column.
    clean_seed = np.where(seed_bad[..., None], seed_prediction, rgb[:seed_rows])
    history = clean_seed[-window:].copy()
    ys = np.broadcast_to(np.arange(seed_rows-window, seed_rows)[:, None], (window, width)).copy()
    failures = np.zeros(width, dtype=int)
    for row in range(seed_rows, height):
        if not active.any():
            break
        mean_y = ys.mean(axis=0)
        centered_y = ys-mean_y
        mean_rgb = history.mean(axis=0)
        denominator = np.sum(centered_y**2, axis=0)
        slopes = np.sum(centered_y[..., None]*(history-mean_rgb), axis=0)/denominator[:, None]
        fitted = mean_rgb+centered_y[..., None]*slopes
        residual = history-fitted
        residual_center = np.median(residual, axis=0)
        sigma = np.maximum(config.noise_floor_rgb,
                           1.4826*np.median(np.abs(residual-residual_center), axis=0))
        prediction = np.clip(mean_rgb+(row-mean_y)[:, None]*slopes+residual_center, 0, 255)
        good = active & (np.max(np.abs(rgb[row]-prediction)/sigma, axis=1) <= config.color_sigma)
        failures = np.where(good, 0, failures+active)
        stopped = active & (failures >= n)
        boundary[stopped] = row-n+1
        active[stopped] = False
        history[:-1, good] = history[1:, good]
        history[-1, good] = rgb[row, good]
        ys[:-1, good] = ys[1:, good]
        ys[-1, good] = row
    return boundary


def estimate_gradient(image, config=GradientConfig()):
    require(image.width >= 16 and image.height >= 16, 'image too small for skyline')
    small = image.convert('RGB')
    small.thumbnail((config.max_side, config.max_side), Image.Resampling.BOX)
    rgb = np.asarray(small.filter(ImageFilter.MedianFilter(config.median_size)), dtype=float)
    height, width = rgb.shape[:2]
    seed_rows = max(config.history_rows, int(height*config.seed_fraction))
    require(3 <= config.consecutive_rows <= config.history_rows <= seed_rows < height,
            'invalid tracking window for image dimensions')
    prediction, bad, sigma = seed_plane(rgb[:seed_rows], config)
    diagnostics = {'working_dimensions': [width, height], 'seed_rows': seed_rows,
                   'seed_residual_sigma_rgb': sigma.tolist(), 'seed_rejected_fraction': float(bad.mean())}
    if sigma.max() > config.max_seed_sigma_rgb:
        return None, {**diagnostics, 'decision': 'abstain_heterogeneous_seed'}
    boundary = track_columns(rgb, prediction, bad, config)
    padded = np.pad(boundary, 1, mode='edge')
    boundary = np.minimum.reduce([padded[:-2], padded[1:-1], padded[2:]])
    boundary = np.where(boundary < height, np.maximum(0, boundary-config.margin_pixels), height)
    fraction = float(boundary.mean()/height)
    diagnostics['selected_fraction'] = fraction
    if fraction < config.min_area_fraction:
        return None, {**diagnostics, 'decision': 'abstain_small_region'}
    return skyline_polygon(boundary, height), {**diagnostics, 'decision': 'mask'}
