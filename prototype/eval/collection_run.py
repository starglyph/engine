#!/usr/bin/env python3
"""Frozen, per-ID collection measurements. No solver tuning or external uploads.

Run starglyph and wcs into separate fresh directories, then compare. Heavy logs,
images and WCS products belong in artifacts/, not the dataset manifest.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
COLLECTION = "robustness-2026-10"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def resume_rows(prior, current, summary, plan_sha):
    """Resume the measured prefix, allowing only a recorded executor revision."""
    if {k: v for k, v in prior.items() if k != "runner_sha256"} != {
            k: v for k, v in current.items() if k != "runner_sha256"}:
        raise ValueError("resume settings or frozen inputs changed")
    rows = summary["frames"]
    if (summary["plan_sha256"] != plan_sha or summary["completed"] != len(rows)
            or summary["input_count"] != prior["input_count"]
            or [r["id"] for r in rows] != [r["id"] for r in prior["inputs"][:len(rows)]]):
        raise ValueError("resume requires an intact completed prefix")
    for row, source in zip(rows, prior["inputs"]):
        if "report" in row and digest(Path(row["report"])) != row["report_sha256"]:
            raise ValueError("completed report changed")
        if "reference" in row:
            ref = json.loads(Path(row["reference"]).read_text())
            if ref["source_sha256"] != source["sha256"]:
                raise ValueError("completed reference input changed")
            for key in ("wcs", "correspondences"):
                if key + "_file" in ref and digest(Path(ref[key + "_file"])) != ref[key + "_sha256"]:
                    raise ValueError("completed WCS products changed")
    return rows


def bounded(command, log, timeout):
    start = time.monotonic()
    with log.open("w") as stream:
        with subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                              start_new_session=True) as process:
            try:
                code = process.wait(timeout=timeout)
                result = {"status": "completed" if code == 0 else "process_error", "exit_code": code}
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                result = {"status": "wall_timeout"}
    return {**result, "elapsed_s": time.monotonic() - start, "command": command}


def eligible(rec, stage):
    if rec["track"] == "scene":
        return False, "projection outside the single-pose evaluation"
    if stage == "wcs" and rec["research"].get("negative"):
        return False, "negative example without identified stellar field"
    if stage == "wcs" and rec["track"] == "stress":
        return False, "extended trails outside point-source WCS protocol"
    return True, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("starglyph", "wcs", "compare"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/samples/sky-samples/manifest.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "prototype/target/release/starglyph")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalogs/hyg_v42.csv.gz")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--solve-field", default="solve-field")
    parser.add_argument("--starglyph-run", type=Path)
    parser.add_argument("--wcs-run", type=Path)
    parser.add_argument("--resume", action="store_true", help="continue an intact prefix with identical measurement settings")
    parser.add_argument("--batch-size", type=int, help="stop cleanly after this many additional records")
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.resume and args.stage == "compare":
        parser.error("comparison requires a fresh directory")
    manifest = args.manifest.resolve()
    records = [r for r in json.loads(manifest.read_text())
               if r.get("research", {}).get("collection") == COLLECTION]
    if not records:
        parser.error("no collection records")
    for rec in records:
        source = manifest.parent / rec["file"]
        if digest(source) != rec["clean_sha256"]:
            parser.error(f"input hash mismatch: {rec['id']}")
    if args.out_dir.exists() and not args.resume:
        parser.error("out-dir must be fresh; refusing to mix runs")
    if args.resume and not (args.out_dir / "summary.json").exists():
        parser.error("resume requires an existing summary")
    args.out_dir.mkdir(parents=True, exist_ok=args.resume)
    run_lock = (args.out_dir / ".run.lock").open("a")
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("another executor owns this run directory")
    plan = {"stage": args.stage, "manifest_sha256": digest(manifest),
            "collection": COLLECTION, "input_count": len(records),
            "inputs": [{"id": r["id"], "sha256": r["clean_sha256"],
                        "split": r["research"]["split"]} for r in records],
            "runner_sha256": digest(Path(__file__)),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    if args.stage == "starglyph":
        plan.update(binary_sha256=digest(args.binary), catalog_sha256=digest(args.catalog),
                    wall_limit_s=120, cache="prewarmed; individual CLI processes; no masks")
    if args.stage == "wcs":
        if not args.config or not args.index_dir:
            parser.error("wcs requires config and index-dir")
        from solve_local_wcs import solve_one
        plan.update(measurement_code_sha256=digest(Path(__file__).with_name("solve_local_wcs.py")),
                    config_sha256=digest(args.config),
                    config_text=args.config.read_text(),
                    solver_version=subprocess.check_output([args.solve_field, "--version"], text=True).strip(),
                    indices={p.name: digest(p) for p in sorted(args.index_dir.glob("index-*.fits"))},
                    cpu_limit_s=30, attempt_wall_limit_s=60, downsample=[4, 2])
        if not plan["indices"]:
            parser.error("no indices")
    if args.stage == "compare":
        if not args.starglyph_run or not args.wcs_run:
            parser.error("compare requires both run directories")
        from compare_local_wcs import compare
        for directory in (args.starglyph_run, args.wcs_run):
            prior = json.loads((directory / "plan.json").read_text())
            if prior["manifest_sha256"] != plan["manifest_sha256"] or prior["inputs"] != plan["inputs"]:
                parser.error("runs do not use the same frozen inputs")
        solver_rows = {r["id"]: r for r in json.loads((args.starglyph_run / "summary.json").read_text())["frames"]}
        wcs_rows = {r["id"]: r for r in json.loads((args.wcs_run / "summary.json").read_text())["frames"]}
    rows = []
    if args.resume:
        prior = json.loads((args.out_dir / "plan.json").read_text())
        # The first collection run predates resumable execution. Its tracked WCS
        # helper must still match the recorded commit before admitting a resume.
        if args.stage == "wcs" and "measurement_code_sha256" not in prior:
            committed = subprocess.check_output(
                ["git", "show", prior["git_commit"] + ":prototype/eval/solve_local_wcs.py"], cwd=ROOT)
            prior["measurement_code_sha256"] = hashlib.sha256(committed).hexdigest()
        rows = resume_rows(prior, plan, json.loads((args.out_dir / "summary.json").read_text()),
                           digest(args.out_dir / "plan.json"))
        number = len(list(args.out_dir.glob("resume-*.json"))) + 1
        continuation = {"completed_before": len(rows), "runner_sha256": plan["runner_sha256"],
                        "plan_sha256": digest(args.out_dir / "plan.json"), "batch_size": args.batch_size,
                        "measurement_code_sha256": plan.get("measurement_code_sha256"),
                        "quarantined_incomplete_ids": []}
        if len(rows) < len(records):
            rid = records[len(rows)]["id"]
            partial = args.out_dir / rid
            if partial.exists():
                target = args.out_dir / "interrupted" / f"{rid}-{number}"
                target.parent.mkdir(exist_ok=True)
                partial.rename(target)
                continuation["quarantined_incomplete_ids"].append(rid)
        write_json(args.out_dir / f"resume-{number:04d}.json", continuation)
    else:
        write_json(args.out_dir / "plan.json", plan)
    remaining = records[len(rows):]
    if args.batch_size:
        remaining = remaining[:args.batch_size]
    for rec in remaining:
        rid = rec["id"]
        source = (manifest.parent / rec["file"]).resolve()
        row = {"id": rid, "split": rec["research"]["split"]}
        allowed, reason = eligible(rec, args.stage)
        if not allowed:
            row.update(status="not_attempted", reason=reason)
        elif args.stage == "starglyph":
            dest = args.out_dir / rid
            command = [str(args.binary.resolve()), "eval", "--manifest", str(manifest),
                       "--ids", rid, "--tracks", "solver,stress", "--catalog", str(args.catalog.resolve()),
                       "--out-dir", str(dest.resolve())]
            row.update(bounded(command, args.out_dir / f"{rid}.log", 120))
            report = dest / "solve-reports" / f"{rid}.json"
            if row["status"] == "completed":
                if report.exists():
                    data = json.loads(report.read_text())
                    row.update(report=str(report), report_sha256=digest(report),
                               solve_status=data["report"]["status"])
                else:
                    row.update(status="missing_report")
        elif args.stage == "wcs":
            settings = SimpleNamespace(solve_field=args.solve_field, config=args.config.resolve(), cpu_limit=30)
            started = time.monotonic()
            try:
                reference = solve_one(source, args.out_dir, settings)
                row.update(status=reference["status"], review_status=reference.get("review_status", "unresolved"),
                           reference=str(args.out_dir / rid / "reference.json"))
                attempts = reference.get("attempts", [])
                if reference["status"] == "unresolved" and attempts:
                    if all(a.get("wall_timeout") for a in attempts):
                        row["status"] = "wall_timeout"
                    elif all(a.get("exit_code", 0) != 0 for a in attempts):
                        row["status"] = "process_error"
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                row.update(status="process_error", error=str(error))
            # This grayscale scratch image is reproducible from the hashed source;
            # keep logs, correspondences and WCS, without accumulating multi-GB FITS.
            (args.out_dir / rid / "input.fits").unlink(missing_ok=True)
            row["elapsed_s"] = time.monotonic() - started
        else:
            solver, external = solver_rows[rid], wcs_rows[rid]
            if solver["status"] != "completed" or "reference" not in external:
                row.update(status="not_comparable", starglyph_status=solver["status"], wcs_status=external["status"])
            else:
                try:
                    row.update(status="compared", metrics=compare(
                        json.loads(Path(external["reference"]).read_text()),
                        json.loads(Path(solver["report"]).read_text())))
                except (OSError, ValueError, KeyError) as error:
                    row.update(status="comparison_error", error=str(error))
        rows.append(row)
        write_json(args.out_dir / "summary.json", {"plan_sha256": digest(args.out_dir / "plan.json"),
                   "input_count": len(records), "completed": len(rows), "frames": rows})
        print(rid, row["status"], flush=True)
    print(f"recorded {len(rows)}/{len(records)}; complete={len(rows) == len(records)}", flush=True)
    failures = {"process_error", "missing_report", "comparison_error"}
    return any(r["status"] in failures for r in rows)


if __name__ == "__main__":
    raise SystemExit(main())
