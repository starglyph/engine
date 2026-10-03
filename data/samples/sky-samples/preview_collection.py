#!/usr/bin/env python3
"""Make local review sheets. Previews retain each source's license, never commit them."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    records = [r for r in json.loads(args.manifest.read_text())
               if r.get("research", {}).get("collection") == "robustness-2026-10"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=14)
    for start in range(0, len(records), 6):
        sheet = Image.new("RGB", (1200, 1500), "#171717")
        draw = ImageDraw.Draw(sheet)
        for j, rec in enumerate(records[start:start + 6]):
            x, y = j % 2 * 600, j // 2 * 500
            with Image.open(args.manifest.parent / rec["file"]) as raw:
                im = raw.convert("RGB")
            sheet.paste(ImageOps.contain(im, (585, 280)), (x, y + 70))
            draw.text((x + 4, y + 3), f"{rec['id']} | {rec['width']}x{rec['height']}", font=font, fill="white")
            draw.text((x + 4, y + 23), f"{rec['author'][:58]} | {rec['license']}", font=font, fill="white")
            draw.text((x + 4, y + 43), "Native crops, brightness x3 (review only)", font=font, fill="white")
            for k, (u, v) in enumerate(((.2, .2), (.5, .5), (.8, .2))):
                cx, cy = int(im.width * u), int(im.height * v)
                box = (max(0, cx - 90), max(0, cy - 65), min(im.width, cx + 90), min(im.height, cy + 65))
                sheet.paste(ImageEnhance.Brightness(im.crop(box)).enhance(3), (x + k * 195, y + 360))
        sheet.save(args.out_dir / f"sheet-{start // 6:02d}.jpg", quality=90)
    # Keep complete source/credit/license links next to the derived sheets.
    (args.out_dir / "credits.json").write_text(json.dumps([
        {k: r[k] for k in ("id", "page_url", "license_url", "attribution_text")}
        for r in records], ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
