"""Experimental RGB skyline baseline, independent of annotations and solver output.

Assumes sky touches the top and each column contains one sky/foreground boundary.
This is a deliberately limited image heuristic, not a semantic sky classifier.
"""
from dataclasses import asdict, dataclass
import time

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from smartphone_gate import require, sha256


@dataclass(frozen=True)
class Config:
    max_side: int = 320
    median_size: int = 5
    seed_fraction: float = 0.125
    noise_floor_rgb: float = 3.0
    color_sigma: float = 4.0
    consecutive_rows: int = 3
    margin_pixels: int = 2
    min_area_fraction: float = 0.15
    max_seed_sigma_rgb: float = 20.0


def skyline_polygon(boundary, height):
    """Simple x-monotone ring, including zero-height columns at either edge.

    A tiny positive height prevents a self-touching ring where sky is absent.
    It is smaller than a pixel center at any supported input resolution.
    """
    width = len(boundary)
    ys = np.maximum(np.asarray(boundary, dtype=float) / height, 1e-10)
    bottom = [[1.0, float(ys[-1])]]
    bottom += [[(x + 0.5) / width, float(ys[x])] for x in range(width-1, -1, -1)]
    bottom += [[0.0, float(ys[0])]]
    return [[0.0, 0.0], [1.0, 0.0]] + bottom


def estimate(image, config=Config()):
    """Return polygon or None (abstain), and diagnostics; input is oriented RGB."""
    require(image.width >= 16 and image.height >= 16, 'image too small for skyline')
    small = image.convert('RGB')
    small.thumbnail((config.max_side, config.max_side), Image.Resampling.BOX)
    rgb = np.asarray(small.filter(ImageFilter.MedianFilter(config.median_size)), dtype=float)
    height, width = rgb.shape[:2]
    seed_rows = max(1, int(height * config.seed_fraction))
    seed = rgb[:seed_rows].reshape(-1, 3)
    center = np.median(seed, axis=0)
    sigma = np.maximum(config.noise_floor_rgb, 1.4826 * np.median(np.abs(seed-center), axis=0))
    diagnostics = {'working_dimensions': [width, height], 'seed_rgb': center.tolist(),
                   'seed_sigma_rgb': sigma.tolist()}
    if float(sigma.max()) > config.max_seed_sigma_rgb:
        return None, {**diagnostics, 'decision': 'abstain_heterogeneous_seed'}
    incompatible = np.max(np.abs(rgb-center)/sigma, axis=2) > config.color_sigma
    # Isolated stars must not terminate an entire sky column.
    n = config.consecutive_rows
    sustained = np.ones((height-n+1, width), dtype=bool)
    for offset in range(n):
        sustained &= incompatible[offset:height-n+1+offset]
    boundary = np.where(sustained.any(axis=0), sustained.argmax(axis=0), height)
    # Expand foreground sideways by one working pixel, then leave a sky margin.
    padded = np.pad(boundary, 1, mode='edge')
    boundary = np.minimum.reduce([padded[:-2], padded[1:-1], padded[2:]])
    boundary = np.where(boundary < height, np.maximum(0, boundary-config.margin_pixels), height)
    fraction = float(boundary.mean()/height)
    diagnostics['selected_fraction'] = fraction
    if fraction < config.min_area_fraction:
        return None, {**diagnostics, 'decision': 'abstain_small_region'}
    return skyline_polygon(boundary, height), {**diagnostics, 'decision': 'mask'}


def algorithm_settings(algorithm):
    if algorithm == 'rgb_skyline_v1':
        return Config(), estimate, 'Automatic RGB skyline v1'
    if algorithm == 'rgb_gradient_v2':
        from gradient_sky_mask import GradientConfig, estimate_gradient
        return GradientConfig(), estimate_gradient, 'Automatic local RGB gradient v2'
    raise ValueError(f'unknown mask algorithm: {algorithm}')


def generate(manifest_path, entries, config=None, *, algorithm='rgb_skyline_v1'):
    """Only source pixels/identity enter estimation; no masks, IDs or solve hints."""
    default_config, estimator, label = algorithm_settings(algorithm)
    config = default_config if config is None else config
    masks, diagnostics = [], {}
    for entry in entries:
        started = time.monotonic()
        source = manifest_path.parent / entry['file']
        require(sha256(source) == entry['sha256'], f"{entry['id']}: input hash mismatch")
        with Image.open(source) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
        expected = [entry['width'], entry['height']]
        if entry['orientation'] in (5, 6, 7, 8):
            expected.reverse()
        require(list(image.size) == expected, f"{entry['id']}: oriented dimensions mismatch")
        polygon, diagnostic = estimator(image, config)
        diagnostics[entry['id']] = {**diagnostic, 'width': image.width, 'height': image.height,
                                  'source_sha256': entry['sha256'],
                                  'generation_ms': (time.monotonic()-started)*1000}
        if polygon is not None:
            masks.append({'id': entry['id'], 'source_sha256': entry['sha256'],
                          'width': image.width, 'height': image.height, 'sky_polygon': polygon,
                          'note': f'{label}; not segmentation ground truth.'})
    return {'schema_version': 1, 'coordinates': 'exif_oriented_normalized_image_edges',
            'description': f'{label}. Derived from photographs by Igor Lazarev, CC BY 4.0.',
            'masks': masks}, {'algorithm': algorithm, 'config': asdict(config), 'frames': diagnostics}
