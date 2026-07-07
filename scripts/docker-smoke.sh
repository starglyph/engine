#!/usr/bin/env bash
# Container smoke for starglyph-serve (Stage 0 · C3 acceptance):
#   1. build the image (context = repo root);
#   2. cold start on a fresh volume under the documented resource floor
#      (--cpus 2 --memory 2g): /readyz turns green once the bootstrap database
#      is generated into the volume, then a solve round-trips;
#   3. restart on the same volume: ready again within seconds and the log has
#      zero "generating" lines — the databases survived the restart.
#
# With a real frame present (repo default: data/input) the solve must report
# "solved"; without one (e.g. a trimmed checkout) a synthetic starless PNG
# must still complete with status "failed" — the endpoint contract holds.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${DOCKER_SMOKE_IMAGE:-starglyph-serve:smoke}"
PORT="${DOCKER_SMOKE_PORT:-18090}"
FRAME="${DOCKER_SMOKE_FRAME:-$ROOT/data/input/CD_2011-09-19_0000.bmp}"
NAME="starglyph-smoke-$$"
VOL="$NAME-data"
OUT="$(mktemp -d)"

# --network=host: build-container DNS is broken on some sandboxes/runners;
# host networking lets apt/cargo resolve through the host (where the daemon
# already pulls the base images anyway). Harmless where bridge DNS works.
docker build --network=host -t "$IMAGE" "$ROOT"

cleanup() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker volume rm "$VOL" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Dense prewarm is disabled so the phases stay deterministic (a half-finished
# background prewarm at restart time would legitimately regenerate bands and
# defeat the no-regeneration assertion); the blind solve below builds its own
# band once, inside the request, hence the raised solve timeout.
start_and_wait_ready() { # $1 = phase label, $2 = readyz timeout in seconds
  docker run -d --name "$NAME" --cpus 2 --memory 2g \
    -p "127.0.0.1:$PORT:8080" -v "$VOL:/var/lib/starglyph" \
    -e STARGLYPH_SERVE_PREWARM_DENSE= \
    -e STARGLYPH_SERVE_SOLVE_TIMEOUT_S=600 \
    "$IMAGE" >/dev/null
  for _ in $(seq 1 "$2"); do
    curl -fsS "http://127.0.0.1:$PORT/readyz" >/dev/null 2>&1 && return 0
    if [ -z "$(docker ps -q -f "name=$NAME")" ]; then
      echo "[docker-smoke] $1: container died during warmup:"
      docker logs "$NAME"
      exit 1
    fi
    sleep 1
  done
  echo "[docker-smoke] $1: /readyz never turned green"
  docker logs --tail 30 "$NAME"
  exit 1
}

echo "[docker-smoke] cold start (first ever run generates the bootstrap database)…"
COLD_T0=$(date +%s)
start_and_wait_ready cold 600
echo "[docker-smoke] cold ready in $(($(date +%s) - COLD_T0))s"

if [ -f "$FRAME" ]; then
  curl -fsS -F "image=@$FRAME" "http://127.0.0.1:$PORT/solve" -o "$OUT/report.json"
  python3 - "$OUT/report.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
assert r["status"] == "solved", r.get("failure", r)
print(f"[docker-smoke] solved: fov={r['fov']['fov_x_deg']:.2f} deg, "
      f"total={r['timing_ms']['total']}ms")
EOF
else
  echo "[docker-smoke] no real frame available; posting a synthetic starless PNG"
  python3 - "$OUT/synthetic.png" <<'EOF'
import struct, sys, zlib
w = h = 64
raw = b"".join(b"\x00" + bytes(w) for _ in range(h))
def chunk(tag, data):
    body = tag + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))
png = (b"\x89PNG\r\n\x1a\n"
       + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
       + chunk(b"IDAT", zlib.compress(raw))
       + chunk(b"IEND", b""))
open(sys.argv[1], "wb").write(png)
EOF
  curl -fsS -F "image=@$OUT/synthetic.png" "http://127.0.0.1:$PORT/solve" -o "$OUT/report.json"
  python3 - "$OUT/report.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
assert r["status"] == "failed" and r["failure"]["code"], r
print(f"[docker-smoke] endpoint round-trip ok (failed as expected: {r['failure']['code']})")
EOF
fi

docker rm -f "$NAME" >/dev/null
echo "[docker-smoke] restart on the same volume…"
WARM_T0=$(date +%s)
start_and_wait_ready warm 120
WARM_SECS=$(($(date +%s) - WARM_T0))
if docker logs "$NAME" 2>&1 | grep -qi "generating"; then
  echo "[docker-smoke] FAIL: databases were regenerated after the restart"
  docker logs "$NAME"
  exit 1
fi
curl -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null
echo "[docker-smoke] PASS: warm restart ready in ${WARM_SECS}s, no regeneration"
