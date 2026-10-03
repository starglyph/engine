#!/usr/bin/env python3
"""Diagnostic overview and native correspondence crops; never accepts WCS as GT."""
import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

from collection_run import ROOT, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/samples/sky-samples/manifest.json")
    parser.add_argument("--wcs-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    records = {r["id"]: r for r in json.loads(args.manifest.read_text())}
    plan = json.loads((args.wcs_run / "plan.json").read_text())
    if plan["manifest_sha256"] != digest(args.manifest):
        raise ValueError("manifest differs from WCS run")
    references = []
    for row in json.loads((args.wcs_run / "summary.json").read_text())["frames"]:
        if row["status"] == "solved_candidate":
            references.append((records[row["id"]], json.loads(Path(row["reference"]).read_text())))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=14)
    for start in range(0, len(references), 4):
        sheet = Image.new("RGB", (1200, 1300), "#171717")
        draw = ImageDraw.Draw(sheet)
        for j, (rec, ref) in enumerate(references[start:start + 4]):
            source = args.manifest.parent / rec["file"]
            if digest(source) != ref["source_sha256"] or digest(Path(ref["correspondences_file"])) != ref["correspondences_sha256"]:
                raise ValueError("source or correspondences changed")
            with Image.open(source) as raw:
                im = raw.convert("RGB")
            gain = 1 if np.percentile(np.asarray(im.resize((64, 64))), 90) > 80 else 3
            table = fits.getdata(ref["correspondences_file"])
            if "match_weight" in table.names:
                table = table[table["match_weight"] >= .95]
            x, y = j % 2 * 600, j // 2 * 650
            draw.text((x + 4, y + 3), f"{rec['id']} | pending | {len(table)} high-weight matches", font=font, fill="white")
            draw.text((x + 4, y + 22), f"{rec['author'][:55]} | {rec['license']}", font=font, fill="white")
            preview = ImageOps.contain(im, (580, 330))
            pd = ImageDraw.Draw(preview)
            sx, sy = preview.width / im.width, preview.height / im.height
            for row in table:
                px, py = (row["field_x"] - 1) * sx, (row["field_y"] - 1) * sy
                pd.ellipse((px - 2, py - 2, px + 2, py + 2), outline="cyan")
            sheet.paste(preview, (x, y + 50))
            draw.text((x + 4, y + 384), f"Native crops x{gain} brightness: cyan centroid / red catalog", font=font, fill="white")
            table.sort(order="field_x")
            choices = np.linspace(0, len(table) - 1, min(6, len(table)), dtype=int) if len(table) else []
            for k, index in enumerate(choices):
                row = table[index]
                cx, cy = row["field_x"] - 1, row["field_y"] - 1
                left, top = round(cx) - 90, round(cy) - 55
                crop = ImageEnhance.Brightness(im.crop((left, top, left + 180, top + 110))).enhance(gain)
                cd = ImageDraw.Draw(crop)
                a, b = row["index_x"] - 1 - left, row["index_y"] - 1 - top
                cd.ellipse((a - 6, b - 6, a + 6, b + 6), outline="red")
                a, b = cx - left, cy - top
                cd.line((a - 3, b, a + 3, b), fill="cyan")
                cd.line((a, b - 3, a, b + 3), fill="cyan")
                sheet.paste(crop, (x + k % 3 * 195, y + 415 + k // 3 * 115))
        sheet.save(args.out_dir / f"wcs-{start // 4:02d}.jpg", quality=90)
    (args.out_dir / "credits.json").write_text(json.dumps([
        {k: rec[k] for k in ("id", "page_url", "license_url", "attribution_text")}
        for rec, _ in references], ensure_ascii=False, indent=2) + "\n")
    print(len(references), "candidate previews; no review status changed")


if __name__ == "__main__":
    main()
