#!/usr/bin/env python3
"""Prepare development source traces and controlled replay for iteration 2.

The source probes are inherited from iteration 1, not new ground truth.
Run with the Python environment containing astropy for the saved .corr file.
"""
import argparse
import json
from pathlib import Path

from centroid_replay import prepare_cases
from collection_run import ROOT, digest, select_records, write_json

IDS = ('wm_r_112929230', 'wm_r_161606532')
MANIFEST = ROOT / 'data/samples/sky-samples/manifest.json'


def prepare(out):
    records = {r['id']: r for r in select_records(json.loads(MANIFEST.read_text()), 'development')}
    for rid in IDS:
        rec = records[rid]  # Split selection precedes all image/reference access.
        source = MANIFEST.parent / rec['file']
        if digest(source) != rec['clean_sha256']:
            raise ValueError('source hash mismatch')
        if rid == 'wm_r_112929230':
            path = ROOT / 'docs/experiments/robustness-iteration-1-regression.json'
            points = json.loads(path.read_text())['control_source_points']
            provenance = dict(kind='control inliers; not independent truth', source_sha256=digest(path))
        else:
            from astropy.io import fits
            path = ROOT / f'prototype/artifacts/robustness-baseline/wcs-run/{rid}/reference.json'
            ref = json.loads(path.read_text())
            corrpath = ROOT / 'prototype' / ref['correspondences_file']
            if digest(corrpath) != ref['correspondences_sha256'] or ref['source_sha256'] != digest(source):
                raise ValueError('external provenance mismatch')
            table = fits.getdata(corrpath)
            rows = [i for i, c in enumerate(table) if c['match_weight'] >= .95
                    and i not in (1, 10, 22, 31, 61, 64, 66)]
            points = [[float(table[i]['field_x']-1), float(table[i]['field_y']-1)] for i in rows]
            provenance = dict(kind='reviewed visible sources; identities and WCS remain pending',
                              corr_rows=rows, source_sha256=digest(path), corr_sha256=digest(corrpath))
        write_json(out / f'{rid}-trace-input.json', dict(image=str(source), probes=points,
            radius_original_px=12, provenance=provenance, split='development',
            image_sha256=digest(source), manifest_sha256=digest(MANIFEST)))


def replay(out, rid, width, tier):
    if rid not in IDS:
        raise ValueError('only the two selected development IDs are allowed')
    path = out / f'{rid}-traces.json'
    inputs = json.loads((out / f'{rid}-trace-input.json').read_text())
    rows = json.loads(path.read_text())['tiers']
    ds = [next(t for t in rows if t['width'] == width and t['tier'] == tier
               and t['quantile'] == mode and not t.get('blob_concentration')) for mode in (False, True)]
    # Same 12 original px association radius as source recovery, conservatively
    # mapped to the longest working dimension. Correspondence is pose-independent.
    from PIL import Image
    with Image.open(inputs['image']) as im:
        radius = 12 * max(width/im.width, ds[0]['height']/im.height)
    cases, pairs = prepare_cases([d['result']['detections'] for d in ds], radius)
    write_json(out / f'{rid}-{width}-{tier}-replay-input.json', dict(
        image=inputs['image'], catalog=str(ROOT / 'data/catalogs/hyg_v42.csv.gz'),
        cache=str(ROOT / 'prototype/artifacts/cache'), width=width, height=ds[0]['height'],
        tier=tier, cases=cases, provenance=dict(trace_sha256=digest(path),
            pairs=pairs, radius_working_px=radius, split='development',
            scope='matching only before refinement; controlled intervention, not automatic improvement')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--replay-id', choices=IDS)
    parser.add_argument('--width', type=int, default=1600)
    parser.add_argument('--tier', choices=('default', 'deep'), default='default')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.replay_id:
        replay(args.out, args.replay_id, args.width, args.tier)
    else:
        prepare(args.out)


if __name__ == '__main__':
    main()
