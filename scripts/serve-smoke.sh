#!/usr/bin/env bash
# End-to-end smoke test for starglyph-serve (Stage 0 · C1 acceptance):
#   1. build and start the service, wait for /readyz to turn green;
#   2. solve one real frame: 200 + SolveReport JSON, overlay PNG accessible;
#   3. run N parallel solves: all succeed and no pattern database is rebuilt
#      (cache-file mtimes/sizes unchanged across the parallel phase).
#
# On a warm cache (repo default) the whole run takes well under a minute; on a
# cold cache the first request additionally generates the dense band once.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO="$ROOT/prototype"
ADDR="${SERVE_SMOKE_ADDR:-127.0.0.1:18080}"
# CD_2011-09-19_0000: known to solve blind in ~0.1 s on a warm cache.
FRAME="${SERVE_SMOKE_FRAME:-$ROOT/data/input/CD_2011-09-19_0000.bmp}"
PARALLEL="${SERVE_SMOKE_PARALLEL:-6}"
CACHE="$PROTO/artifacts/cache"
OUT="$(mktemp -d)"

[ -f "$FRAME" ] || { echo "[smoke] frame '$FRAME' not found"; exit 1; }

cd "$PROTO"
cargo build --release -p starglyph-serve

./target/release/starglyph-serve --addr "$ADDR" --pool-size 2 --prewarm-dense "" \
  --telemetry-log "$OUT/telemetry.jsonl" \
  >"$OUT/serve.log" 2>&1 &
SERVE_PID=$!
cleanup() {
  kill "$SERVE_PID" 2>/dev/null || true
  wait "$SERVE_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ── 1. readiness ──────────────────────────────────────────────────────────────
for _ in $(seq 1 300); do
  curl -fsS "http://$ADDR/readyz" >/dev/null 2>&1 && break
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "[smoke] server died during warmup:"; cat "$OUT/serve.log"; exit 1
  fi
  sleep 1
done
curl -fsS "http://$ADDR/readyz" >/dev/null \
  || { echo "[smoke] /readyz never turned green"; tail -20 "$OUT/serve.log"; exit 1; }
curl -fsS "http://$ADDR/healthz" >/dev/null
echo "[smoke] ready"

# ── 2. single solve: JSON report + overlay PNG ────────────────────────────────
curl -fsS -F "image=@$FRAME" "http://$ADDR/solve" -o "$OUT/report.json"
python3 - "$OUT/report.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
assert r["status"] == "solved", r.get("failure", r)
assert r["pose"] and r["fov"] and r["quality"], r
print(f"[smoke] solved: ra={r['pose']['ra_deg']:.2f} dec={r['pose']['dec_deg']:.2f} "
      f"fov={r['fov']['fov_x_deg']:.2f} inliers={r['quality']['n_inliers']} "
      f"total={r['timing_ms']['total']}ms")
EOF

curl -fsS -F "image=@$FRAME" "http://$ADDR/solve?overlay=png" -o "$OUT/overlay.png"
head -c8 "$OUT/overlay.png" | cmp -s - <(printf '\x89PNG\r\n\x1a\n') \
  || { echo "[smoke] overlay response is not a PNG"; exit 1; }
echo "[smoke] overlay PNG ok ($(stat -c%s "$OUT/overlay.png") bytes)"

# ── 3. parallel solves must not rebuild databases ─────────────────────────────
before="$(stat -c '%n %Y %s' "$CACHE"/*.bin | sort)"
pids=()
for i in $(seq 1 "$PARALLEL"); do
  curl -fsS -F "image=@$FRAME" "http://$ADDR/solve" -o "$OUT/par-$i.json" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
after="$(stat -c '%n %Y %s' "$CACHE"/*.bin | sort)"
if [ "$before" != "$after" ]; then
  echo "[smoke] FAIL: cache files changed during parallel solves (duplicate rebuild?)"
  diff <(echo "$before") <(echo "$after") || true
  exit 1
fi
python3 - "$OUT" "$PARALLEL" <<'EOF'
import json, pathlib, statistics, sys
out, n = pathlib.Path(sys.argv[1]), int(sys.argv[2])
totals = []
for i in range(1, n + 1):
    r = json.load(open(out / f"par-{i}.json"))
    assert r["status"] == "solved", (i, r.get("failure"))
    totals.append(r["timing_ms"]["total"])
print(f"[smoke] {n} parallel solves ok; total_ms median={statistics.median(totals):.0f} max={max(totals)}")
EOF

# ── 4. telemetry: one anonymous record per request (Stage 0 · D1) ────────────
python3 - "$OUT/telemetry.jsonl" "$((2 + PARALLEL))" <<'EOF'
import json, sys
path, expected = sys.argv[1], int(sys.argv[2])
records = [json.loads(line) for line in open(path)]
assert len(records) == expected, f"expected {expected} records, got {len(records)}"
assert all(r["outcome"] == "solved" for r in records), [r["outcome"] for r in records]
last = records[-1]
assert last["source"] and last["image"]["width"] > 0, last
assert last["timing"]["wall_ms"] > 0 and last["timing"]["total_ms"] > 0, last["timing"]
forbidden = {"user", "user_id", "ip", "client", "gps"}
assert all(not (forbidden & set(r)) for r in records), "telemetry must stay anonymous"
print(f"[smoke] telemetry ok: {len(records)} records, "
      f"last wall={last['timing']['wall_ms']}ms queue={last['timing']['queue_ms']}ms")
EOF

if grep -qi "generating" "$OUT/serve.log"; then
  echo "[smoke] NOTE: this run generated databases (cold cache); rerun for warm-latency numbers"
fi
echo "[smoke] PASS (log: $OUT/serve.log)"
