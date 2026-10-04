#!/usr/bin/env python3
"""Review every frozen wide-rematch pair before choosing any new intervention."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image, ImageDraw, ImageOps

from collection_run import ROOT, digest, write_json
from robustness_geometry import native_centroid, selected, validate_review
from robustness_pairs import RID

TRACE = ROOT/'prototype/artifacts/robustness-iteration-8/verified/trace.json'
GEOMETRY = ROOT/'docs/experiments/robustness-iteration-8-geometry.json'
REVIEW = ROOT/'docs/experiments/robustness-iteration-7-review.json'
CATALOG = ROOT/'data/catalogs/hyg_v42.csv.gz'
CASE = 'drop_detection_all_stages'
STAGE = 'rematch_10_before'


def load_stage():
    selected([RID])
    geometry = json.loads(GEOMETRY.read_text())
    if digest(TRACE) != geometry['trace_sha256']:
        raise ValueError('frozen trace changed')
    trace = json.loads(TRACE.read_text())
    if trace['split'] != 'development' or trace['id'] != RID:
        raise ValueError('development trace required')
    case = next(c for c in trace['cases'] if c['name'] == CASE)
    stage = next(s for s in case['stages'] if s['stage'] == STAGE)
    if len(stage['pairs']) != 34:
        raise ValueError('wide-pair selection changed')
    return stage, geometry['sources']


def prepare(out):
    stage, probes = load_stage()
    rec = selected([RID])[0]
    previous = json.loads(REVIEW.read_text())['frames'][0]
    source = ROOT/previous['image']
    if digest(source) != rec['clean_sha256'] or previous['source_sha256'] != rec['clean_sha256']:
        raise ValueError('image changed')
    if out.exists():
        raise ValueError('fresh output directory required')
    with gzip.open(CATALOG, 'rt') as stream:
        catalog = {int(s['id']):s for s in csv.DictReader(stream)}
    nearby = [s for s in catalog.values() if s['mag'] and float(s['mag']) <= 7]
    ra, dec = np.deg2rad([[float(s['ra'])*15, float(s['dec'])] for s in nearby]).T
    units = np.column_stack((np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)))
    extraction = ROOT/'prototype/artifacts/robustness-baseline/wcs-run'/RID/'downsample-2/field.axy'
    axy = fits.getdata(extraction)
    xy = np.column_stack((axy['X'], axy['Y'])).astype(float)-1
    with Image.open(source) as original:
        image = original.convert('RGB')
    gray = np.asarray(image, dtype=np.float32) @ np.array([.2126, .7152, .0722], dtype=np.float32)
    points = []
    for i, pair in enumerate(stage['pairs']):
        if len(pair['hyg_ids']) != 1:
            raise ValueError('ambiguous saved identity')
        star = catalog[pair['hyg_ids'][0]]
        center = (np.array(pair['xy'])+.5)*2.5-.5
        nearest = np.argsort(np.linalg.norm(xy-center, axis=1))[:3]
        angular = np.rad2deg(np.arccos(np.clip(units@pair['world'], -1, 1)))
        neighbours = [dict(hyg_id=int(nearby[n]['id']), mag=float(nearby[n]['mag']), separation_deg=float(angular[n]))
                      for n in np.flatnonzero((angular < .2) & (angular > 1e-5))]
        prior = [dict(hyg_id=p['hyg_id'], label=p['label'], distance_px=float(np.linalg.norm(center-p['detector_xy_original'])))
                 for p in previous['points'] if np.linalg.norm(center-p['detector_xy_original']) < 12]
        old_probes = [dict(hyg_id=p['hyg_id'], name=p['name'], distance_px=float(np.linalg.norm(center-p['xy'])))
                      for p in probes if np.linalg.norm(center-p['xy']) < 12]
        points.append(dict(id=i, hyg_id=int(star['id']), name=star['proper'] or star['bf'], mag=float(star['mag']),
            world=pair['world'], detector_xy_original=center.tolist(), detector_xy_working=pair['xy'],
            native_r8_xy=native_centroid(gray, center, 8), native_r12_xy=native_centroid(gray, center, 12),
            choices=[dict(row=int(n), x=float(xy[n,0]), y=float(xy[n,1]), distance_px=float(np.linalg.norm(xy[n]-center))) for n in nearest],
            prior_pairs=prior, prior_probes=old_probes, neighbours=neighbours, label='pending', selected_row=None))
    out.mkdir(parents=True)
    template = dict(split='development', frames=[dict(id=RID, image=str(source.relative_to(ROOT)),
        source_sha256=rec['clean_sha256'], width=image.width, height=image.height, points=points)])
    write_json(out/'candidates.json', template)
    write_json(out/'review.json', template)
    for start in range(0, len(points), 12):
        sheet = Image.new('RGB', (1200, 1140))
        draw = ImageDraw.Draw(sheet)
        for point in points[start:start+12]:
            slot = point['id']-start
            left, top = (slot%4)*300, (slot//4)*380
            x, y = point['detector_xy_original']
            tile = image.crop((round(x)-75, round(y)-75, round(x)+75, round(y)+75)).resize((300,300))
            sheet.paste(ImageOps.autocontrast(tile, cutoff=.5), (left, top))
            draw.line((left+130, top+150, left+140, top+150), fill='cyan')
            draw.line((left+160, top+150, left+170, top+150), fill='cyan')
            draw.text((left+6, top+307), f"{point['id']}: HYG {point['hyg_id']} {point['name']}", fill='white')
            draw.text((left+6, top+327), f"mag {point['mag']}; axy {point['choices'][0]['distance_px']:.2f}px", fill='white')
            draw.text((left+6, top+347), f"prior pair/probe {len(point['prior_pairs'])}/{len(point['prior_probes'])}", fill='white')
        sheet.save(out/f'pairs-{start//12}.jpg')
    overview = image.copy()
    overview.thumbnail((900,1200))
    draw = ImageDraw.Draw(overview)
    for point in points:
        x, y = (np.array(point['detector_xy_original'])+.5)*overview.width/image.width-.5
        draw.ellipse((x-5, y-5, x+5, y+5), outline='cyan')
        draw.text((x+6,y), str(point['id']), fill='yellow')
    overview.save(out/'overview.jpg')
    files = [source, CATALOG, extraction, TRACE, GEOMETRY, REVIEW, Path(__file__).resolve(),
             ROOT/'prototype/eval/robustness_geometry.py', ROOT/'prototype/eval/robustness_pairs.py',
             ROOT/'data/samples/sky-samples/manifest.json']
    write_json(out/'protocol.json', dict(iteration=10, id=RID, split='development', case=CASE, stage=STAGE,
        pair_count=34, hashes={str(p.relative_to(ROOT)):digest(p) for p in files},
        candidates_sha256=digest(out/'candidates.json'), selection='all 34 wide-rematch pairs; no residual-based selection',
        followup='new fit only for visually confirmed false pairs; ambiguous/blended identities remain uncertain',
        limitations='physical source review and relative field agreement do not constitute absolute astrometric ground truth'))


def reviewed(out):
    protocol = json.loads((out/'protocol.json').read_text())
    if protocol['split'] != 'development' or protocol['id'] != RID:
        raise ValueError('development required')
    selected([RID])
    for path, sha in protocol['hashes'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('frozen input changed')
    if digest(out/'candidates.json') != protocol['candidates_sha256']:
        raise ValueError('candidates changed')
    original = json.loads((out/'candidates.json').read_text())
    review = json.loads((out/'review.json').read_text())
    validate_review(original, review)
    points = review['frames'][0]['points']
    used = set()
    for point in points:
        if point['label'] not in ('visible_source', 'ambiguous', 'blend', 'not_visible') or not point.get('note'):
            raise ValueError('complete visual review with reasons')
        if point['label'] == 'visible_source':
            row = point['selected_row']
            if row not in [c['row'] for c in point['choices']] or row in used:
                raise ValueError('invalid or reused external centroid')
            used.add(row)
    return points


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'check'])
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.out.resolve())
    else:
        reviewed(args.out.resolve())
