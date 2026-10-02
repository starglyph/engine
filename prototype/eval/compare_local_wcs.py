#!/usr/bin/env python3
"""Compare eval/solve-reports with independent, EXIF-oriented local WCS.

Reports full-field projection differences separately from errors at external
star correspondences. Pending reference candidates are never called ground truth.
Requires NumPy and Astropy. Does not run or fit either solver.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project(camera, radec):
    angles = np.deg2rad(np.asarray(radec, dtype=float))
    ra, dec = angles[:, 0], angles[:, 1]
    world = np.column_stack((np.cos(dec) * np.cos(ra),
                             np.cos(dec) * np.sin(ra), np.sin(dec)))
    rotation = np.asarray(camera["world_to_camera"], dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("invalid camera rotation shape")
    cam = world @ rotation.T
    if np.any(cam[:, 2] <= 0.05):
        raise ValueError("reference directions fall outside the solved camera")
    uv = cam[:, :2] / cam[:, 2, None]
    factor = 1 + camera["k1"] * np.sum(uv ** 2, axis=1)
    offsets = camera["focal_px"] * uv * factor[:, None]
    pixels = offsets * [1, -1] + [camera["width"] / 2, camera["height"] / 2]
    if not np.isfinite(pixels).all():
        raise ValueError("non-finite projected coordinates")
    return pixels


def stats(residual):
    return {"count": len(residual), "rms_px": float(np.sqrt(np.mean(residual ** 2))),
            "median_px": float(np.median(residual)),
            "p95_px": float(np.percentile(residual, 95)),
            "max_px": float(np.max(residual))}


def compare(reference, artifact):
    if reference["source_sha256"] != artifact["source_sha256"]:
        raise ValueError("source SHA-256 mismatch")
    for key in ("width", "height", "pixel_convention"):
        if reference[key] != artifact[key]:
            raise ValueError(f"{key} mismatch")
    if reference["pixel_convention"] != "top_left_zero_based":
        raise ValueError("unsupported pixel convention")
    result = {"reference_status": reference["status"],
              "review_status": reference.get("review_status", "unresolved"),
              "solver_status": artifact["report"]["status"]}
    if reference["status"] != "solved_candidate" or artifact["report"]["status"] != "solved":
        return result
    wcs_path = Path(reference["wcs_file"])
    if sha256(wcs_path) != reference["wcs_sha256"]:
        raise ValueError("WCS SHA-256 mismatch")
    camera = artifact["camera"]
    if (camera["width"], camera["height"]) != (artifact["width"], artifact["height"]):
        raise ValueError("camera dimensions mismatch")
    points = reference["reference_points"]
    expected = np.array([[p["x"], p["y"]] for p in points])
    world = WCS(fits.getheader(wcs_path)).all_pix2world(expected, 0)
    predicted = project(camera, world)
    result["full_field_wcs_difference"] = stats(np.linalg.norm(predicted - expected, axis=1))
    # Use the external catalog's stars and measured centroids, not the engine's
    # inliers. FITS table coordinates are one-based; both axes keep row order.
    if "correspondences_file" in reference:
        corr_path = Path(reference["correspondences_file"])
        if sha256(corr_path) != reference["correspondences_sha256"]:
            raise ValueError("correspondences SHA-256 mismatch")
        corr = fits.getdata(corr_path)
        if "match_weight" in corr.names:
            corr = corr[corr["match_weight"] >= 0.95]
        if len(corr):
            expected = np.column_stack((corr["field_x"], corr["field_y"])) - 1
            predicted = project(camera, np.column_stack((corr["index_ra"], corr["index_dec"])))
            result["external_star_error"] = stats(np.linalg.norm(predicted - expected, axis=1))
            result["external_wcs_fit"] = stats(np.hypot(corr["field_x"] - corr["index_x"],
                                                       corr["field_y"] - corr["index_y"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True, help="local WCS summary.json")
    parser.add_argument("--reports-dir", type=Path, required=True, help="eval/solve-reports directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for reference in json.loads(args.references.read_text())["frames"]:
        frame_id = Path(reference["source"]).stem
        path = args.reports_dir / f"{frame_id}.json"
        try:
            record = compare(reference, json.loads(path.read_text()))
        except (ValueError, KeyError, OSError) as error:
            record = {"error": str(error)}
        records.append({"id": frame_id, **record})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"frames": records}, indent=2, allow_nan=False) + "\n")
    if any("error" in record for record in records):
        raise SystemExit("comparison failed for some frames; see output")


if __name__ == "__main__":
    main()
