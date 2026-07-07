#!/usr/bin/env python3
"""Roll up starglyph-serve telemetry (Stage 0 · D1) into key KPIs.

Usage:
    python3 scripts/telemetry-rollup.py [solve-log.jsonl ...]

With no arguments reads the default log at prototype/artifacts/telemetry/
solve-log.jsonl. Prints solve-rate, outcome/failure breakdowns and latency
percentiles. Self-host / operator view only: the log is anonymous by design,
so there is no per-user dimension here (that belongs to closed wrappers).
"""

import json
import pathlib
import sys
from collections import Counter


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def load(paths: list[pathlib.Path]) -> list[dict]:
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"warning: {path}:{n}: bad JSON line skipped ({e})", file=sys.stderr)
    return records


def fov_bucket(record: dict) -> str:
    fov = (record.get("result") or {}).get("fov_x_deg")
    if fov is None:
        return "unknown"
    for hi, label in ((5, "<5°"), (15, "5–15°"), (30, "15–30°"), (50, "30–50°"), (80, "50–80°")):
        if fov < hi:
            return label
    return "≥80°"


def main() -> int:
    if len(sys.argv) > 1:
        paths = [pathlib.Path(p) for p in sys.argv[1:]]
    else:
        root = pathlib.Path(__file__).resolve().parents[1]
        paths = [root / "prototype/artifacts/telemetry/solve-log.jsonl"]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"error: no such log: {', '.join(map(str, missing))}", file=sys.stderr)
        return 1

    records = load(paths)
    if not records:
        print("no records")
        return 0

    outcomes = Counter(r["outcome"] for r in records)
    solved = outcomes.get("solved", 0)
    failed = outcomes.get("failed", 0)
    completed = solved + failed

    print(f"records: {len(records)}  (span {records[0]['ts']} … {records[-1]['ts']})")
    print(f"outcomes: " + ", ".join(f"{k}={v}" for k, v in outcomes.most_common()))
    if completed:
        print(f"solve-rate: {solved}/{completed} = {solved / completed:.1%} (of completed solves)")

    rejects = Counter(r.get("reject_code", "?") for r in records if r["outcome"] == "rejected")
    if rejects:
        print("rejects: " + ", ".join(f"{k}={v}" for k, v in rejects.most_common()))
    failures = Counter(r.get("failure_code", "?") for r in records if r["outcome"] == "failed")
    if failures:
        print("failures: " + ", ".join(f"{k}={v}" for k, v in failures.most_common()))

    for name, key in (("wall_ms", "wall_ms"), ("total_ms", "total_ms")):
        values = [
            float(r["timing"][key])
            for r in records
            if r["outcome"] != "rejected" and r["timing"].get(key) is not None
        ]
        if values:
            print(
                f"{name}: p50={percentile(values, 0.50):.0f} "
                f"p95={percentile(values, 0.95):.0f} max={max(values):.0f} (n={len(values)})"
            )

    queues = [float(r["timing"]["queue_ms"]) for r in records if r["outcome"] != "rejected"]
    if queues:
        print(f"queue_ms: p50={percentile(queues, 0.50):.0f} p95={percentile(queues, 0.95):.0f}")

    solved_fovs = Counter(fov_bucket(r) for r in records if r["outcome"] == "solved")
    if solved_fovs:
        print("solved FOV: " + ", ".join(f"{k}={v}" for k, v in solved_fovs.most_common()))

    exif_present = sum(1 for r in records if (r.get("exif") or {}).get("present"))
    print(f"exif present: {exif_present}/{len(records)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
