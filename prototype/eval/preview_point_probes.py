#!/usr/bin/env python3
"""Render the visual probe sample and optional working-scale deep detections (Pillow)."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from smartphone_gate import read_json, require, sha256
from sky_statistics_experiment import original_detections, point_in_polygon


def render(args):
    probes = read_json(args.probes)['frames']
    entries = {e['id']: e for e in read_json(args.manifest)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cols = 3 if args.run_dir else 1
    sheet = Image.new('RGB', (800*cols, 500*len(probes)), '#111111')
    crop_sheet = Image.new('RGB', (800, 200*((sum(len(f['points']) for f in probes)+3)//4)), '#111111')
    crop_index = 0
    for row, annotation in enumerate(probes):
        source = args.manifest.parent/entries[annotation['id']]['file']
        require(sha256(source) == annotation['source_sha256'], 'probe source mismatch')
        with Image.open(source) as raw:
            original = ImageOps.exif_transpose(raw).convert('RGB')
        require(original.size == (annotation['width'], annotation['height']), 'probe dimensions mismatch')
        for p in annotation['points']:
            x, y = round(p['x']), round(p['y'])
            crop = original.crop((x-32,y-32,x+32,y+32)).resize((160,160))
            crop = ImageEnhance.Brightness(crop).enhance(3)
            cx, cy = (crop_index % 4)*200, (crop_index//4)*200
            crop_sheet.paste(crop, (cx,cy))
            ImageDraw.Draw(crop_sheet).text((cx+4,cy+165), annotation['id'][-6:]+' '+p['id']+
                                           (' ?' if p['label'] != 'visible_compact_source' else ''), fill='white')
            crop_index += 1
        if args.run_dir and annotation.get('cloud_regions'):
            artifact = read_json(args.run_dir/args.modes[0]/'solve-reports'/f"{annotation['id']}.json")
            require(artifact['source_sha256'] == annotation['source_sha256'], 'cloud artifact source mismatch')
            diagnostic = next(d for d in artifact['detection_diagnostics'] if d['width'] == annotation['width'] and d['height'] == annotation['height'] and d['tier'] == 'deep')
            candidates = [d for d in diagnostic['result']['detections'] if d['rank'] < diagnostic['max_detections'] and
                          any(point_in_polygon(d['x'], d['y'], c['polygon']) for c in annotation['cloud_regions'])]
            if candidates:
                cloud_sheet = Image.new('RGB', (600, 200*((len(candidates)+2)//3)), '#111111')
                for i, d in enumerate(candidates):
                    x, y = round(d['x']), round(d['y'])
                    crop = ImageEnhance.Brightness(original.crop((x-32,y-32,x+32,y+32)).resize((160,160))).enhance(1.5)
                    draw = ImageDraw.Draw(crop)
                    draw.line((75,80,85,80),fill='red'); draw.line((80,75,80,85),fill='red')
                    cloud_sheet.paste(crop, ((i%3)*200,(i//3)*200))
                    ImageDraw.Draw(cloud_sheet).text(((i%3)*200+4,(i//3)*200+165),f"rank {d['rank']}",fill='white')
                cloud_sheet.save(args.out_dir/f"{annotation['id']}-cloud-crops.jpg",quality=92)
        for col, mode in enumerate(['manual'] + (args.modes if args.run_dir else [])):
            im = original.copy()
            im.thumbnail((800,451))
            im = ImageEnhance.Brightness(im).enhance(3)
            draw = ImageDraw.Draw(im)
            sx, sy = im.width/original.width, im.height/original.height
            if mode != 'manual':
                artifact = read_json(args.run_dir/mode/'solve-reports'/f"{annotation['id']}.json")
                require(artifact['source_sha256'] == annotation['source_sha256'], 'artifact source mismatch')
                diagnostic = next(d for d in artifact['detection_diagnostics'] if d['tier'] == 'deep' and max(d['width'],d['height']) == 1600)
                for d in original_detections(diagnostic, original.width, original.height):
                    if d['rank'] < diagnostic['max_detections']:
                        x,y = (d['x']+.5)*sx-.5, (d['y']+.5)*sy-.5
                        draw.ellipse((x-3,y-3,x+3,y+3), outline='#ffff00')
            for region in annotation.get('cloud_regions', []):
                polygon = [(x*sx,y*sy) for x,y in region['polygon']]
                draw.line(polygon+polygon[:1],fill='#ff66ff',width=2)
            for p in annotation['points']:
                x,y = (p['x']+.5)*sx-.5, (p['y']+.5)*sy-.5
                color = '#00ffff' if p['label'] == 'visible_compact_source' else '#ff8800'
                draw.ellipse((x-6,y-6,x+6,y+6),outline=color)
                if mode == 'manual':
                    draw.text((x+7,y),p['id'],fill=color)
            sheet.paste(im, (col*800,row*500))
            ImageDraw.Draw(sheet).text((col*800+8,row*500+460), f"{annotation['id']} / {mode} / brightness x3", fill='white')
    sheet.save(args.out_dir/'probes.jpg',quality=92)
    crop_sheet.save(args.out_dir/'probe-crops.jpg',quality=92)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=Path('../data/input/smartphone/manifest.json'))
    parser.add_argument('--probes',type=Path,default=Path('../data/input/smartphone/point-source-probes.json'))
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--modes',nargs=2,default=['centroid','statistics'])
    parser.add_argument('--out-dir',type=Path,required=True)
    render(parser.parse_args())
