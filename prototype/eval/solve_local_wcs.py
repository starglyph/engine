#!/usr/bin/env python3
"""Obtain independent WCS candidates locally; never upload images.

Requires Astrometry.net (solve-field plus installed indices), Astropy, NumPy,
and Pillow. Output is tied to EXIF-oriented pixels, with zero-based x/y and
row zero at the top. A successful external solve is a candidate for review,
not an assertion of absolute ground truth.
"""
import argparse
import hashlib
import json
import os
import signal
import shutil
import subprocess
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image, ImageOps


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def solve_one(source, output, args):
    dest = output / source.stem
    dest.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as raw:
        orientation = raw.getexif().get(274, 1)
        oriented = ImageOps.exif_transpose(raw)
        # Preserve the dynamic range of sensor TIFFs; convert('L') would clip
        # their background and stars alike to 255 before source extraction.
        pixels = np.asarray(oriented if oriented.mode in ("I", "F", "I;16")
                            else oriented.convert("L"), dtype=np.float32)
    if not np.isfinite(pixels).all():
        raise ValueError(f"non-finite image pixels: {source}")
    height, width = pixels.shape
    input_fits = dest / "input.fits"
    fits.writeto(input_fits, pixels, overwrite=True)
    result = {
        "source": str(source), "source_sha256": sha256(source),
        "orientation_applied": orientation, "width": width, "height": height,
        "pixel_convention": "top_left_zero_based",
        "input_fits_sha256": sha256(input_fits),
        "status": "unresolved", "attempts": [],
    }
    for factor in (4, 2):
        attempt = dest / f"downsample-{factor}"
        attempt.mkdir(exist_ok=True)
        for name in ("field.solved", "field.wcs", "field.corr"):
            (attempt / name).unlink(missing_ok=True)
        command = [
            args.solve_field, str(input_fits), "--dir", str(attempt),
            "--out", "field", "--overwrite", "--no-plots", "--no-verify",
            "--new-fits", "none", "--uniformize", "0", "--resort",
            "--pixel-error", "4", "--depth", "10,20,30,50",
            "--downsample", str(factor), "--cpulimit", str(args.cpu_limit),
            "--tweak-order", "2", "--crpix-center",
        ]
        if args.config:
            command += ["--config", str(args.config)]
        with (attempt / "solve.log").open("w") as log:
            try:
                with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True) as run:
                    try:
                        status = {"exit_code": run.wait(timeout=args.cpu_limit + 30)}
                    except subprocess.TimeoutExpired:
                        os.killpg(run.pid, signal.SIGKILL)
                        run.wait()
                        raise
            except subprocess.TimeoutExpired:
                status = {"wall_timeout": True}
        result["attempts"].append({"command": command, **status})
        marker, wcs_path = attempt / "field.solved", attempt / "field.wcs"
        if status.get("exit_code") != 0 or not marker.exists() or marker.read_bytes() != b"\x01" or not wcs_path.exists():
            continue
        wcs = WCS(fits.getheader(wcs_path))
        # Sample the full nonlinear WCS, not pixscale * image width.
        points = np.array([(x, y) for y in np.linspace(0, height - 1, 9)
                           for x in np.linspace(0, width - 1, 9)])
        world = wcs.all_pix2world(points, 0)
        center = wcs.all_pix2world([[width / 2, height / 2]], 0)[0]
        if not np.isfinite(world).all() or not np.isfinite(center).all():
            result["attempts"][-1]["invalid_wcs"] = "non-finite coordinates"
            continue
        result.update(status="solved_candidate", review_status="pending",
                      wcs_file=str(wcs_path), wcs_sha256=sha256(wcs_path),
                      center_ra_deg=float(center[0]), center_dec_deg=float(center[1]),
                      reference_points=[{"x": float(x), "y": float(y),
                                         "ra_deg": float(ra), "dec_deg": float(dec)}
                                        for (x, y), (ra, dec) in zip(points, world)])
        corr_path = attempt / "field.corr"
        if corr_path.exists():
            corr = fits.getdata(corr_path)
            result["correspondences_file"] = str(corr_path)
            result["correspondences_sha256"] = sha256(corr_path)
            result["n_correspondences"] = len(corr)
            if all(k in corr.names for k in ("field_x", "field_y", "index_x", "index_y")):
                residual = np.hypot(corr["field_x"] - corr["index_x"],
                                    corr["field_y"] - corr["index_y"])
                result["fit_rms_px"] = float(np.sqrt(np.mean(residual ** 2)))
                result["fit_p95_px"] = float(np.percentile(residual, 95))
                trusted = corr[corr["match_weight"] >= 0.95] if "match_weight" in corr.names else corr
                result["n_high_weight_correspondences"] = len(trusted)
                if len(trusted):
                    result["matched_extent_fraction"] = [
                        float(np.ptp(trusted["field_x"]) / width),
                        float(np.ptp(trusted["field_y"]) / height),
                    ]
        break
    (dest / "reference.json").write_text(json.dumps(result, indent=2) + "\n")
    print(source.name, result["status"], result.get("n_correspondences", ""), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--solve-field", default="solve-field")
    parser.add_argument("--cpu-limit", type=int, default=30)
    parser.add_argument("--index-dir", type=Path, required=True,
                        help="directory containing configured index FITS files (hashed for provenance)")
    args = parser.parse_args()
    if args.cpu_limit < 1:
        parser.error("--cpu-limit must be positive")
    if not shutil.which(args.solve_field):
        parser.error("solve-field is not installed or not on PATH")
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.config:
        args.config = args.config.resolve()
    indices = sorted(args.index_dir.resolve().glob("index-*.fits"))
    if not indices:
        parser.error("--index-dir contains no index-*.fits files")
    provenance = {
        "config": ({"path": str(args.config), "sha256": sha256(args.config),
                    "text": args.config.read_text()} if args.config else None),
        "indices": [{"path": str(p), "sha256": sha256(p)} for p in indices],
    }
    sources = sorted(p.resolve() for p in args.input_dir.iterdir()
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    if not sources:
        parser.error("no supported images found")
    if len({p.stem for p in sources}) != len(sources):
        parser.error("input image stems must be unique")
    version = subprocess.check_output([args.solve_field, "--version"], text=True).strip()
    records = []
    for source in sources:
        records.append(solve_one(source, args.out_dir, args))
        (args.out_dir / "summary.json").write_text(json.dumps({
            "solver": "Astrometry.net", "version": version,
            "provenance": provenance,
            "input_count": len(sources), "completed": len(records), "frames": records,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
