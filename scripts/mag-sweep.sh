#!/usr/bin/env bash
# B4: sweep the pattern-database magnitude depth (STARGLYPH_DB_MAG_LIMIT) and
# report the cost/quality curve on the sky-samples eval set.
#
# For each magnitude the eval runs twice into a dedicated scratch cache
# (removed at the start so every sweep is reproducible): the cold pass pays
# database generation (bootstrap + whatever blind-ladder bands the frames
# demand), the warm pass measures pure solving. Depth-tagged cache file names
# keep the sweep away from the default mag65 cache.
#
#   scripts/mag-sweep.sh                 # sweep 6.5 7.0 7.5
#   MAG_SWEEP_MAGS="6.5 8.0" scripts/mag-sweep.sh
#
# Output: prototype/artifacts/eval/mag-sweep/report.md (+ per-run eval dirs).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO="$ROOT/prototype"
read -r -a MAGS <<<"${MAG_SWEEP_MAGS:-6.5 7.0 7.5}"
CACHE="$PROTO/artifacts/cache-mag-sweep"
OUT="$PROTO/artifacts/eval/mag-sweep"

cd "$PROTO"
cargo build --release -p starglyph-cli
rm -rf "$CACHE"
mkdir -p "$OUT"

for mag in "${MAGS[@]}"; do
  tag="mag$(python3 -c "print(round(float('$mag') * 10))")"
  for pass in cold warm; do
    echo "[mag-sweep] $tag $pass pass…"
    t0=$(date +%s)
    STARGLYPH_DB_MAG_LIMIT="$mag" ./target/release/starglyph eval \
      --manifest ../data/samples/sky-samples/manifest.json \
      --catalog ../data/catalogs/hyg_v42.csv.gz \
      --cache-dir "$CACHE" \
      --out-dir "$OUT/$tag-$pass"
    echo "$(($(date +%s) - t0))" >"$OUT/$tag-$pass/wall_seconds"
  done
done

python3 - "$OUT" "$CACHE" "${MAGS[@]}" <<'EOF'
import json, pathlib, sys

out, cache = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
lines = [
    "# B4 mag-depth sweep (sky-samples, blind)",
    "",
    "| mag | solve-rate | axis med/p95 (deg) | fov err med (rel) | solve med (ms) "
    "| DB total | gen cost (s) |",
    "|---|---|---|---|---|---|---|",
]
for mag in sys.argv[3:]:
    tag = f"mag{round(float(mag) * 10)}"
    warm = json.load(open(out / f"{tag}-warm/summary.json"))
    cold_s = int(open(out / f"{tag}-cold/wall_seconds").read())
    warm_s = int(open(out / f"{tag}-warm/wall_seconds").read())
    st = warm["solver_track"]
    pe = warm["pose_errors"]
    axis = pe.get("axis_angle_deg") or {}
    fov = pe.get("fov_error_rel") or {}
    solve = (warm.get("timing_ms") or {}).get("solve") or {}
    size = sum(f.stat().st_size for f in cache.glob(f"*-{tag}-*.bin"))
    lines.append(
        f"| {mag} | {st['solved']}/{st['attempted']} = {st['solve_rate']:.3f} "
        f"| {axis.get('median', float('nan')):.3f} / {axis.get('p95', float('nan')):.3f} "
        f"| {fov.get('median', float('nan')):.4f} "
        f"| {solve.get('median', float('nan')):.0f} "
        f"| {size / 1e6:.0f} MB | ~{max(cold_s - warm_s, 0)} |"
    )
report = "\n".join(lines) + "\n"
(out / "report.md").write_text(report)
print(report)
EOF
