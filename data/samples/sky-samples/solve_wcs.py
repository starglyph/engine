#!/usr/bin/env python3
"""Fetch WCS ground truth for a local sky frame via the astrometry.net API.

WHY THIS AND NOT SCRAPING nova's gallery:
  The per-image license on nova.astrometry.net is JS-rendered, so bulk-harvesting
  "someone else's solved frames" cleanly is fragile AND license-risky. The clean
  path is to solve frames whose license we ALREADY control (the CC0/Apache/CC-BY
  frames in this sample). The resulting WCS is factual celestial coordinates,
  not a copyrightable work — no license entanglement.

  (Confirmed 2026-07-05: `GET nova.astrometry.net/api/jobs/<jobid>/calibration/`
   returns ra/dec/pixscale/orientation/parity WITHOUT auth — but you still need a
   job id, which requires a submission. Hence this submit-and-poll tool.)

REQUIRES: a free astrometry.net API key -> https://nova.astrometry.net/api_help
  export ASTROMETRY_API_KEY=xxxxxxxx

USAGE:
  python3 solve_wcs.py images/B_amateur_widefield_cc0__flickr_orion_rahn.jpg [more...]

OUTPUT (sidecars next to each frame):
  <frame>.wcs.json  -> {ra, dec, pixscale, orientation, parity, radius, fields...}
  <frame>.wcs.fits  -> full WCS header (if solved)

ALTERNATIVE (canonical for this project): run the engine's own blind solver to
  produce pose/WCS in-house — no external dependency, no rate limits. See the
  engine eval-harness (Epic A). This script is a bootstrap / cross-check.
"""
import os, sys, json, time, urllib.request, urllib.parse, mimetypes, io

BASE = "https://nova.astrometry.net/api"
KEY = os.environ.get("ASTROMETRY_API_KEY")

def _post(url, fields):
    data = urllib.parse.urlencode({"request-json": json.dumps(fields)}).encode()
    with urllib.request.urlopen(url, data=data, timeout=60) as r:
        return json.load(r)

def _post_file(url, fields, path):
    boundary = "----starglyphboundary"
    body = io.BytesIO()
    def w(s): body.write(s.encode() if isinstance(s, str) else s)
    w(f"--{boundary}\r\nContent-Disposition: form-data; name=\"request-json\"\r\n\r\n{json.dumps(fields)}\r\n")
    fn = os.path.basename(path)
    ct = mimetypes.guess_type(fn)[0] or "application/octet-stream"
    w(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{fn}\"\r\nContent-Type: {ct}\r\n\r\n")
    w(open(path, "rb").read()); w(f"\r\n--{boundary}--\r\n")
    req = urllib.request.Request(url, data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)

def login():
    r = _post(f"{BASE}/login", {"apikey": KEY})
    if r.get("status") != "success":
        raise SystemExit(f"login failed: {r}")
    return r["session"]

def solve(path, session, poll=5, timeout=600):
    print(f"[submit] {path}")
    sub = _post_file(f"{BASE}/upload", {"session": session, "publicly_visible": "n",
                     "allow_modifications": "d", "allow_commercial_use": "d"}, path)
    if sub.get("status") != "success":
        print(f"  upload failed: {sub}"); return None
    subid = sub["subid"]; t0 = time.time()
    jobid = None
    while time.time() - t0 < timeout:
        s = _post(f"{BASE}/submissions/{subid}", {})
        jobs = [j for j in s.get("jobs", []) if j]
        if jobs:
            jobid = jobs[0]
            js = _post(f"{BASE}/jobs/{jobid}", {})
            st = js.get("status")
            if st == "success": break
            if st == "failure": print(f"  solve failed (job {jobid})"); return None
        time.sleep(poll)
    if not jobid:
        print("  timed out waiting for job"); return None
    calib = _post(f"{BASE}/jobs/{jobid}/calibration", {})
    if not isinstance(calib, dict) or calib.get("ra") is None:
        print(f"  [no-solution] job {jobid}: astrometry.net could not calibrate this frame")
        return None
    info = _post(f"{BASE}/jobs/{jobid}/info", {})
    out = {"jobid": jobid, "calibration": calib,
           "objects_in_field": info.get("objects_in_field"), "tags": info.get("tags")}
    json.dump(out, open(path + ".wcs.json", "w"), indent=2, ensure_ascii=False)
    try:
        with urllib.request.urlopen(f"https://nova.astrometry.net/wcs_file/{jobid}", timeout=60) as r:
            open(path + ".wcs.fits", "wb").write(r.read())
    except Exception as e:
        print(f"  (wcs.fits fetch skipped: {e})")
    c = calib
    print(f"  [OK] job {jobid}: RA={c.get('ra'):.4f} Dec={c.get('dec'):.4f} "
          f"scale={c.get('pixscale'):.3f}\"/px orient={c.get('orientation'):.2f} -> {path}.wcs.json")
    return out

if __name__ == "__main__":
    if not KEY:
        raise SystemExit("Set ASTROMETRY_API_KEY (free: https://nova.astrometry.net/api_help)")
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    sess = login()
    for p in sys.argv[1:]:
        try: solve(p, sess)
        except Exception as e: print(f"[FAIL] {p}: {e}")
