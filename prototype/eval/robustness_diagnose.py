#!/usr/bin/env python3
"""Development-only detection review; source pixels and baseline remain untouched.

Produces overviews, native source crops, detector statistics and correspondence
recovery. Requires the original baseline artifacts and a diagnostics CLI run.
All coordinates are zero-based in the normalized original image.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

from collection_run import ROOT, digest, select_records, write_json


def recover(points, detections, width, height, original_width, original_height, radius=12):
    """Greedy one-to-one source recovery, ordered by distance; not star accuracy."""
    if not detections or not len(points):
        return []
    xy = np.array([[(d['x']+.5)*original_width/width-.5,
                    (d['y']+.5)*original_height/height-.5] for d in detections])
    distances = np.linalg.norm(points[:, None, :]-xy[None, :, :], axis=2)
    pairs = sorted((float(distances[i, j]), int(i), int(j))
                   for i, j in zip(*np.where(distances <= radius)))
    seen_points, seen_detections, matches = set(), set(), []
    for distance, i, j in pairs:
        if i not in seen_points and j not in seen_detections:
            seen_points.add(i)
            seen_detections.add(j)
            matches.append({'probe': i, 'rank': j, 'distance_px': distance})
    return matches


def preview(image, rid, detections, out):
    small = image.copy()
    small.thumbnail((900,680))
    small = ImageEnhance.Brightness(small).enhance(2)
    draw = ImageDraw.Draw(small)
    for rank, point in enumerate(detections):
        x, y = point['x']*small.width/image.width, point['y']*small.height/image.height
        draw.ellipse((x-3, y-3, x+3, y+3), outline='red')
        draw.text((x+3, y), str(rank), fill='yellow')
    small.save(out/f'{rid}-overview.jpg')
    sheet = Image.new('RGB', (960,640))
    for k, p in enumerate(detections[:24]):
        cx, cy = round(p['x']), round(p['y'])
        crop = ImageEnhance.Brightness(image.crop((cx-40,cy-40,cx+40,cy+40))).enhance(2)
        crop = crop.resize((150,130))
        draw = ImageDraw.Draw(crop)
        draw.ellipse((72,62,78,68), outline='red')
        draw.text((2,2), f'{k}: {cx},{cy}', fill='yellow')
        sheet.paste(crop, ((k%6)*160,(k//6)*160))
    sheet.save(out/f'{rid}-crops.jpg')


def external_preview(image, table, out):
    """Label original zero-based .corr rows, including rejected vegetation."""
    rows = [(i, r) for i, r in enumerate(table) if r['match_weight'] >= .95]
    small = image.copy()
    small.thumbnail((900,680))
    draw = ImageDraw.Draw(small)
    for i, row in rows:
        x, y = (row['field_x']-1)*small.width/image.width, (row['field_y']-1)*small.height/image.height
        draw.ellipse((x-4,y-4,x+4,y+4), outline='red')
        draw.text((x+4,y),str(i),fill='yellow')
    small.save(out/'external-overview.jpg')
    for start in range(0,len(rows),24):
        sheet = Image.new('RGB',(960,640))
        for k,(i,row) in enumerate(rows[start:start+24]):
            left, top = round(row['field_x']-1)-50, round(row['field_y']-1)-50
            crop = ImageEnhance.Brightness(image.crop((left,top,left+100,top+100))).enhance(1.5)
            crop = crop.resize((150,140))
            draw = ImageDraw.Draw(crop)
            draw.text((2,2),str(i),fill='yellow')
            draw.ellipse((72,67,78,73),outline='cyan')
            x,y = (row['index_x']-1-left)*1.5,(row['index_y']-1-top)*1.4
            draw.ellipse((x-5,y-5,x+5,y+5),outline='red')
            sheet.paste(crop,((k%6)*160,(k//6)*160))
        sheet.save(out/f'external-crops-{start//24}.jpg')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='development detection-diagnostics run')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    manifest = ROOT/'data/samples/sky-samples/manifest.json'
    records = select_records(json.loads(manifest.read_text()), 'development')
    baseline = {r['id']: r for r in json.loads((manifest.parent/'robustness-results.json').read_text())['frames']}
    frames = []
    for rec in records:
        rid = rec['id']
        if rec['track'] != 'solver' or baseline[rid]['starglyph'].get('solve_status') != 'failed':
            continue
        path = args.run/rid/'solve-reports'/f'{rid}.json'
        if not path.exists():
            continue
        artifact = json.loads(path.read_text())
        source = manifest.parent/rec['file']
        if digest(source) != artifact['source_sha256'] or digest(source) != rec['clean_sha256']:
            raise ValueError('source hash mismatch')
        report = artifact['report']
        ds = report['detections']
        with Image.open(source) as raw:
            preview(raw.convert('RGB'), rid, ds, args.out_dir)
        record = {'id': rid, 'source_sha256': digest(source), 'report_sha256': digest(path),
                  'attribution': rec['attribution_text'], 'status': report['status'],
                  'failure': report.get('failure'), 'timing_ms': report['timing_ms'],
                  'detections': len(ds),
                  'span_xy': [(max(d[k] for d in ds)-min(d[k] for d in ds))/rec[dimension]
                              if ds else 0 for k, dimension in [('x','width'),('y','height')]],
                  'detector_tiers': []}
        external = None
        if rid == 'wm_r_161606532':
            from astropy.io import fits
            refpath = ROOT/'prototype/artifacts/robustness-baseline/wcs-run'/rid/'reference.json'
            ref = json.loads(refpath.read_text())
            corrpath = ROOT/'prototype'/ref['correspondences_file']
            if digest(corrpath) != ref['correspondences_sha256'] or ref['source_sha256'] != digest(source):
                raise ValueError('external correspondence provenance mismatch')
            corr = fits.getdata(corrpath)
            wcspath = ROOT/'prototype'/ref['wcs_file']
            if digest(wcspath) != ref['wcs_sha256']:
                raise ValueError('WCS hash mismatch')
            with Image.open(source) as raw:
                external_preview(raw.convert('RGB'),corr,args.out_dir)
            # Reviewed all high-weight native crops. Exclude vegetation and the
            # four uncertain boundary points; do not call remaining IDs certified.
            rows = [i for i,c in enumerate(corr) if c['match_weight'] >= .95
                    and i not in (1,10,22,31,61,64,66)]
            external = np.array([[corr[i]['field_x']-1,corr[i]['field_y']-1] for i in rows])
            record['external_probe_rows'] = rows
            record['external_review'] = {'status': 'pending', 'correspondences_sha256': digest(corrpath),
                'wcs_sha256':digest(wcspath),
                'vegetation_rows': [61,64,66], 'uncertain_boundary_rows': [1,10,22,31],
                'probe_count': len(rows), 'scope': 'visible source recovery only; identities and full-field WCS not certified',
                'span_xy': (np.ptp(external,axis=0)/[rec['width'],rec['height']]).tolist()}
        for tier in artifact['detection_diagnostics']:
            item = {k: tier[k] for k in ('width','height','tier','max_detections')}
            item['stats'] = tier['result']['stats']
            dets = tier['result']['detections']
            item['pre_top_k_count'] = len(dets)
            if external is not None:
                matches = recover(external,dets,tier['width'],tier['height'],rec['width'],rec['height'])
                item['external_source_recovery'] = {'radius_original_px': 12,
                    'before_top_k': len(matches), 'after_top_k': sum(m['rank']<tier['max_detections'] for m in matches),
                    'matches': [{**m,'corr_row':record['external_probe_rows'][m['probe']]} for m in matches]}
            record['detector_tiers'].append(item)
        frames.append(record)
    write_json(args.out_dir/'diagnosis.json', {'split':'development', 'manifest_sha256':digest(manifest),
        'script_sha256':digest(Path(__file__)), 'frames':frames})


if __name__ == '__main__':
    main()
