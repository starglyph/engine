#!/usr/bin/env python3
"""Render EXIF-oriented manual annotations for visual review (requires Pillow)."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps


def render(manifest, masks, output):
    entries = {e['id']: e for e in json.loads(manifest.read_text())}
    annotations = json.loads(masks.read_text())
    if annotations['coordinates'] != 'exif_oriented_normalized_image_edges':
        raise ValueError('unsupported coordinates')
    output.mkdir(parents=True, exist_ok=True)
    tiles = []
    for annotation in annotations['masks']:
        frame_id = annotation['id']
        source = manifest.parent / entries[frame_id]['file']
        if hashlib.sha256(source.read_bytes()).hexdigest() != annotation['source_sha256']:
            raise ValueError(f'{frame_id}: SHA-256 mismatch')
        with Image.open(source) as original:
            im = ImageOps.exif_transpose(original).convert('RGB')
        if im.size != (annotation['width'], annotation['height']):
            raise ValueError(f'{frame_id}: oriented dimensions mismatch')
        im.thumbnail((1000, 800))
        polygon = [(x * im.width, y * im.height) for x, y in annotation['sky_polygon']]
        sky = Image.new('L', im.size, 0)
        ImageDraw.Draw(sky).polygon(polygon, fill=255)
        foreground = Image.new('RGB', im.size, '#e04444')
        shaded = Image.blend(im, foreground, 0.40)
        im = Image.composite(im, shaded, sky)
        ImageDraw.Draw(im).line(polygon + polygon[:1], fill='#00ffff', width=2)
        im.save(output / f'{frame_id}.jpg', quality=92)
        im.thumbnail((500, 380))
        tile = Image.new('RGB', (510, 415), '#151515')
        tile.paste(im, ((510-im.width)//2, 0))
        ImageDraw.Draw(tile).text((8, 388), frame_id, fill='white')
        tiles.append(tile)
    sheet = Image.new('RGB', (1530, ((len(tiles)+2)//3)*415), '#151515')
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 3)*510, (i//3)*415))
    sheet.save(output / 'contact.jpg', quality=92)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--masks', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    render(args.manifest, args.masks, args.out_dir)
