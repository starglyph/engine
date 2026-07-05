#!/usr/bin/env python3
"""Reconstruct the sky-sample images from manifest.json (nothing binary in git).

The repo stores only provenance (manifest.json + docs). This script rebuilds the
actual frames on demand: download from the recorded source, verify integrity
against orig_sha256, strip EXIF/GPS, resize to the recorded dimensions, and write
into images/ and images-stress-tier-b/.

  python3 fetch_sample.py            # fetch all
  python3 fetch_sample.py --list     # show what would be fetched
  python3 fetch_sample.py <id> ...   # fetch specific ids

Rationale: keeps the repo tiny, sidesteps any redistribution question (we
reference sources, not re-host them), and lets sha256 catch source drift.
"""
import io, os, sys, json, hashlib, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = json.load(open(os.path.join(HERE, "manifest.json")))
UA = "starglyph-dataset-collector/1.0 (https://github.com/starglyph; research)"
IMG_MAGIC = (b"\xff\xd8\xff", b"\x89PNG", b"II*\x00", b"MM\x00*", b"RIFF", b"GIF8")
# tetra3 (Apache-2.0) frames come from the repo, not a media URL:
TETRA3 = {
 "tetra3_alt60": "examples/data/2019-07-29T204726_Alt60_Azi-135_Try1.tiff",
 "tetra3_alt40": "examples/data/2019-07-29T204726_Alt40_Azi-135_Try1.tiff",
}

def sha256(b): return hashlib.sha256(b).hexdigest()

def curl(url, timeout=150, retries=5):
    last = ""
    for a in range(retries):
        r = subprocess.run(["curl","-sS","-L","-A",UA,"--max-time",str(timeout),url],
                           capture_output=True, timeout=timeout+15)
        out = r.stdout
        if r.returncode == 0 and out and any(out.startswith(m) for m in IMG_MAGIC):
            return out
        last = f"rc={r.returncode} starts={out[:12]!r}"
        time.sleep(2 + 2*a)
    raise RuntimeError(f"download failed for {url}: {last}")

def clone_tetra3():
    dst = os.path.join(HERE, ".tetra3_src")
    if not os.path.isdir(dst):
        subprocess.run(["git","clone","--depth","1","https://github.com/esa/tetra3.git",dst],
                       check=True, capture_output=True)
    return dst

def process(raw, w, h, tiff=False):
    from PIL import Image
    img = Image.open(io.BytesIO(raw)); img.load()
    if img.size != (w, h):
        img = img.resize((w, h), Image.LANCZOS)
    clean = Image.new(img.mode, img.size); clean.putdata(list(img.getdata()))
    out = io.BytesIO()
    if tiff:
        clean.save(out, format="TIFF")
    else:
        if clean.mode not in ("RGB","L"): clean = clean.convert("RGB")
        clean.save(out, format="JPEG", quality=92)
    return out.getvalue()

def fetch(rec):
    rid = rec["id"]; path = os.path.join(HERE, rec["file"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tiff = rec["file"].lower().endswith((".tif",".tiff"))
    if rid in TETRA3:
        raw = open(os.path.join(clone_tetra3(), TETRA3[rid]), "rb").read()
    else:
        raw = curl(rec["download_url"])
        got = sha256(raw)
        if rec.get("orig_sha256") and got != rec["orig_sha256"]:
            print(f"  [warn] {rid}: source sha differs (source changed since 2026-07-05)")
    open(path, "wb").write(process(raw, rec["width"], rec["height"], tiff=tiff))
    print(f"  [ok] {rec['file']}  ({rec['license']})")

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--list"]
    if "--list" in sys.argv:
        for m in MAN:
            print(f"{m['id']:36s} {m['license']:12s} {m['file']}")
        sys.exit(0)
    todo = [m for m in MAN if not args or m["id"] in args]
    print(f"fetching {len(todo)} frames...")
    ok = 0
    for m in todo:
        try: fetch(m); ok += 1
        except Exception as e: print(f"  [FAIL] {m['id']}: {e}")
    print(f"done: {ok}/{len(todo)}")
