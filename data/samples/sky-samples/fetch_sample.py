#!/usr/bin/env python3
"""Reconstruct licensed images with strict integrity checks and atomic writes.

A manifest is not a substitute for license review. Legacy records retain their
historical resize/JPEG recipe; new records select oriented_png_v1 (full resolution,
EXIF orientation, no further lossy codec). Originals remain local and ignored.
"""
import argparse
import hashlib
import io
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = "Starglyph/1.0 (https://github.com/starglyph/engine; sky-image research)"
IMG_MAGIC = (b"\xff\xd8\xff", b"\x89PNG", b"II*\x00", b"MM\x00*", b"RIFF", b"GIF8")
TETRA3 = {
    "tetra3_alt60": "examples/data/2019-07-29T204726_Alt60_Azi-135_Try1.tiff",
    "tetra3_alt40": "examples/data/2019-07-29T204726_Alt40_Azi-135_Try1.tiff",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def pixel_sha256(image):
    header = json.dumps([image.mode, image.width, image.height], separators=(",", ":"))
    return sha256(header.encode() + b"\0" + image.tobytes())


def curl(url, timeout=150, retries=3):
    if not url.startswith("https://"):
        raise ValueError("downloads require HTTPS")
    for attempt in range(retries):
        result = subprocess.run(
            ["curl", "-fsSL", "--proto", "=https", "--proto-redir", "=https",
             "-A", UA, "--max-time", str(timeout), url],
            capture_output=True, timeout=timeout + 15)
        if result.returncode == 0 and result.stdout.startswith(IMG_MAGIC):
            return result.stdout
        if attempt + 1 < retries:
            time.sleep(2 + 2 * attempt)
    raise RuntimeError(f"image download failed (curl {result.returncode}): {url}")


def process(raw, w, h, tiff=False):
    """Preserve the historical recipe; do not migrate regression pixels."""
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as source:
        source.load()
        img = source.resize((w, h), Image.Resampling.LANCZOS) if source.size != (w, h) else source
        clean = Image.new(img.mode, img.size)
        clean.putdata(list(img.getdata()))
        out = io.BytesIO()
        if tiff:
            clean.save(out, format="TIFF")
        else:
            if clean.mode not in ("RGB", "L"):
                clean = clean.convert("RGB")
            clean.save(out, format="JPEG", quality=92)
        return out.getvalue()


def normalize(raw):
    """Strip metadata after orientation; retain decoded pixels and sensor range."""
    from PIL import Image, ImageOps
    with Image.open(io.BytesIO(raw)) as source:
        source.load()
        oriented = ImageOps.exif_transpose(source)
        if oriented.mode not in ("RGB", "RGBA", "L", "LA", "I;16", "I", "F"):
            raise ValueError(f"unsupported image mode: {oriented.mode}")
        clean = Image.frombytes(oriented.mode, oriented.size, oriented.tobytes())
        out = io.BytesIO()
        fmt = "TIFF" if clean.mode in ("I;16", "I", "F") else "PNG"
        clean.save(out, format=fmt)
        return out.getvalue(), pixel_sha256(clean), clean.size


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def source_bytes(rec, root):
    original = root / ".originals" / rec["orig_sha256"]
    if original.exists():
        return original.read_bytes()
    if rec["id"] in TETRA3:
        clone = root / ".tetra3_src"
        if not clone.is_dir():
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/esa/tetra3.git", str(clone)],
                           check=True, capture_output=True)
        return (clone / TETRA3[rec["id"]]).read_bytes()
    return curl(rec["download_url"])


def fetch(rec, root=HERE):
    relative = Path(rec["file"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("image path must stay inside the dataset")
    raw = source_bytes(rec, root)
    if sha256(raw) != rec["orig_sha256"]:
        raise ValueError(f"{rec['id']}: source SHA-256 mismatch; existing image preserved")
    mode = rec.get("processing", {}).get("mode", "legacy_v1")
    if mode == "oriented_png_v1":
        data, pixels, size = normalize(raw)
        if pixels != rec["processing"]["pixel_sha256"] or size != (rec["width"], rec["height"]):
            raise ValueError(f"{rec['id']}: normalized pixels or dimensions differ")
    elif mode == "legacy_v1":
        data = process(raw, rec["width"], rec["height"], relative.suffix.lower() in (".tif", ".tiff"))
    else:
        raise ValueError(f"unknown processing mode: {mode}")
    if sha256(data) != rec["clean_sha256"]:
        raise ValueError(f"{rec['id']}: output SHA-256 mismatch; use recorded codec environment")
    atomic_write(root / ".originals" / rec["orig_sha256"], raw)
    atomic_write(root / relative, data)
    print(f"[ok] {rec['id']} ({rec['license']})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="*")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--collection", help="select only this research.collection")
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    args = parser.parse_args()
    records = json.loads(args.manifest.read_text())
    unknown = set(args.ids) - {r["id"] for r in records}
    if unknown:
        parser.error("unknown IDs: " + ", ".join(sorted(unknown)))
    selected = [r for r in records if (not args.ids or r["id"] in args.ids)
                and (not args.collection or r.get("research", {}).get("collection") == args.collection)]
    if not selected:
        parser.error("selection is empty")
    if args.list:
        for rec in selected:
            print(f"{rec['id']:40s} {rec['license']:12s} {rec['file']}")
        return 0
    failed = []
    for rec in selected:
        try:
            fetch(rec, args.manifest.resolve().parent)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            failed.append(rec["id"])
            print(f"[FAIL] {rec['id']}: {error}", flush=True)
    print(f"done: {len(selected) - len(failed)}/{len(selected)}")
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
