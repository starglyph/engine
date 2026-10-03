#!/usr/bin/env python3
"""Export complete frozen runs as a portable, compact collection measurement record."""
import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
import platform

from collection_run import COLLECTION, ROOT, digest, write_json


def read_run(directory, manifest_sha, inputs):
    plan = json.loads((directory / "plan.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if plan["manifest_sha256"] != manifest_sha or plan["inputs"] != inputs:
        raise ValueError("run does not match frozen manifest")
    if summary["plan_sha256"] != digest(directory / "plan.json"):
        raise ValueError("run plan changed after measurement")
    ids = [r["id"] for r in summary["frames"]]
    if (summary["completed"] != len(inputs) or summary["input_count"] != len(inputs)
            or len(ids) != len(set(ids)) or set(ids) != {r["id"] for r in inputs}):
        raise ValueError("incomplete or duplicated run; cannot publish partial rates")
    return plan, {r["id"]: r for r in summary["frames"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/samples/sky-samples/manifest.json")
    parser.add_argument("--starglyph-run", type=Path, required=True)
    parser.add_argument("--wcs-run", type=Path, required=True)
    parser.add_argument("--comparison-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [r for r in json.loads(args.manifest.read_text())
               if r.get("research", {}).get("collection") == COLLECTION]
    inputs = [{"id": r["id"], "sha256": r["clean_sha256"], "split": r["research"]["split"]}
              for r in records]
    manifest_sha = digest(args.manifest)
    solver_plan, solver = read_run(args.starglyph_run, manifest_sha, inputs)
    wcs_plan, wcs = read_run(args.wcs_run, manifest_sha, inputs)
    compare_plan, comparisons = read_run(args.comparison_run, manifest_sha, inputs)
    continuations = [json.loads(path.read_text()) for path in sorted(args.wcs_run.glob("resume-*.json"))]
    interrupted_ids = {rid for run in continuations for rid in run["quarantined_incomplete_ids"]}
    rows = []
    for rec in records:
        rid = rec["id"]
        s, w, c = solver[rid], wcs[rid], comparisons[rid]
        run = {k: s[k] for k in ("status", "solve_status", "elapsed_s", "reason", "exit_code", "report_sha256") if k in s}
        if "report" in s:
            report_path = Path(s["report"])
            if digest(report_path) != s["report_sha256"]:
                raise ValueError("Starglyph report changed: " + rid)
            artifact = json.loads(report_path.read_text())
            if artifact["source_sha256"] != rec["clean_sha256"]:
                raise ValueError("report input mismatch: " + rid)
            report = artifact["report"]
            run.update({k: report[k] for k in ("quality", "timing_ms", "reason") if k in report})
        external = {k: w[k] for k in ("status", "review_status", "elapsed_s", "reason", "error") if k in w}
        if rid in interrupted_ids:
            external["prior_incomplete_attempt"] = "Executor terminated externally; partial artifacts retained under interrupted/. This ID was restarted with unchanged settings."
        if "reference" in w:
            reference = json.loads(Path(w["reference"]).read_text())
            if reference["source_sha256"] != rec["clean_sha256"]:
                raise ValueError("WCS input mismatch: " + rid)
            external.update({k: reference[k] for k in (
                "wcs_sha256", "input_fits_sha256", "n_correspondences", "n_high_weight_correspondences",
                "fit_rms_px", "fit_p95_px", "matched_extent_fraction") if k in reference})
            external["attempts"] = [
                {"downsample": (4, 2)[i], **{k: a[k] for k in ("exit_code", "wall_timeout", "invalid_wcs") if k in a}}
                for i, a in enumerate(reference["attempts"])]
        rows.append({"id": rid, "source_sha256": rec["clean_sha256"], "track": rec["track"],
                     "split": rec["research"]["split"], "negative": rec["research"]["negative"],
                     "starglyph": run, "external_wcs": external,
                     "comparison": {k: v for k, v in c.items() if k not in ("id", "split")}})
    provenance = {
        "manifest_sha256": manifest_sha, "git_commit": solver_plan["git_commit"],
        "binary_sha256": solver_plan["binary_sha256"], "catalog_sha256": solver_plan["catalog_sha256"],
        "runner_sha256": solver_plan["runner_sha256"],
        "wcs_initial_runner_sha256": wcs_plan["runner_sha256"],
        "wcs_continuations": continuations,
        "comparison_runner_sha256": compare_plan["runner_sha256"],
        "solve_local_wcs_sha256": digest(Path(__file__).with_name("solve_local_wcs.py")),
        "compare_local_wcs_sha256": digest(Path(__file__).with_name("compare_local_wcs.py")),
        "run_plan_sha256": {label: digest(directory / "plan.json") for label, directory in (
            ("starglyph", args.starglyph_run), ("wcs", args.wcs_run), ("comparison", args.comparison_run))},
        "wcs_version": wcs_plan["solver_version"], "indices": wcs_plan["indices"],
        "wcs_config_sha256": wcs_plan["config_sha256"],
        "wcs_config_template": ["add_path ${ASTROMETRY_INDEX_DIR}" if line.strip().startswith("add_path ") else line
                                for line in wcs_plan["config_text"].splitlines()],
        "python_version": platform.python_version(),
        "wcs_python_packages": {name: version(name) for name in ("numpy", "astropy", "pillow")},
        "starglyph_wall_limit_s": solver_plan["wall_limit_s"], "wcs_cpu_limit_s": wcs_plan["cpu_limit_s"],
        "wcs_attempt_wall_limit_s": wcs_plan["attempt_wall_limit_s"], "wcs_downsample": wcs_plan["downsample"],
        "cache": solver_plan["cache"],
        "execution": "Two independent sequential queues initially overlapped on one host. WCS resumed in bounded batches after external termination; times are diagnostic, not isolated performance benchmarks.",
    }
    summaries = []
    for split in ("development", "holdout"):
        for track in ("solver", "stress", "scene"):
            group = [r for r in rows if r["split"] == split and r["track"] == track]
            summaries.append({"split": split, "track": track, "count": len(group),
                              "starglyph_status": dict(Counter(r["starglyph"].get("solve_status", r["starglyph"]["status"]) for r in group)),
                              "external_wcs_status": dict(Counter(r["external_wcs"]["status"] for r in group))})
    write_json(args.output, {"collection": COLLECTION, "date": "2026-10-02", "provenance": provenance,
                            "accepted_ground_truth_count": 0,
                            "interpretation": "External solutions are pending candidates. Neither solver success nor mutual agreement certifies a correct full-field solution.",
                            "summaries": summaries, "frames": rows})


if __name__ == "__main__":
    main()
