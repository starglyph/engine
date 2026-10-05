#!/usr/bin/env python3
"""Frozen development WCS/camera audit on visually reviewed physical sources."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import time

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image

from collection_run import ROOT, digest, select_records, write_json
from compare_local_wcs import project, stats
from robustness_geometry import preview, validate_review, native_centroid

RID = 'wm_r_154807046'
MANIFEST = ROOT/'data/samples/sky-samples/manifest.json'
CATALOG = ROOT/'data/catalogs/hyg_v42.csv.gz'
RAW = ROOT/f'prototype/artifacts/robustness-baseline/wcs-run/{RID}'
PRIOR = ROOT/'prototype/artifacts/robustness-iteration-2/development-final'
ARMS = ('control', 'blob')


def read(path):
    return json.loads(path.read_text())


def selected():
    records = select_records(read(MANIFEST), 'development', [RID])
    if len(records) != 1 or records[0]['id'] != RID or records[0]['track'] != 'solver':
        raise ValueError('one development solver required')
    return records[0]


def reports():
    return {a: PRIOR/a/RID/'solve-reports'/f'{RID}.json' for a in ARMS}


def distributed(points, width, height):
    """Three brightest per 4x4 cell, selected without measured residuals."""
    cells, chosen = {}, []
    for p in sorted(points, key=lambda p: (p['mag'], p['hyg_id'])):
        x, y = p['review_center']
        cell = (int(4*x/width), int(4*y/height))
        if cells.get(cell, 0) < 3:
            chosen.append(p | dict(id=len(chosen)))
            cells[cell] = cells.get(cell, 0) + 1
    return chosen


def prepare(out):
    rec = selected()
    if out.exists():
        raise ValueError('fresh output required')
    source = MANIFEST.parent/rec['file']
    ref = read(RAW/'reference.json')
    wcs_path, corr_path = (ROOT/'prototype'/ref[k] for k in ('wcs_file', 'correspondences_file'))
    if digest(source) != rec['clean_sha256'] or ref['source_sha256'] != rec['clean_sha256']:
        raise ValueError('photo changed')
    if digest(wcs_path) != ref['wcs_sha256'] or digest(corr_path) != ref['correspondences_sha256']:
        raise ValueError('external artifacts changed')
    for path in reports().values():
        r = read(path)
        if r['source_sha256'] != rec['clean_sha256'] or r['report']['status'] != 'solved':
            raise ValueError('saved report mismatch')
    with gzip.open(CATALOG, 'rt') as f:
        stars = [s for s in csv.DictReader(f) if s['mag'] and float(s['mag']) <= 5.5 and int(s['id']) > 0]
    radec = np.array([[float(s['ra'])*15, float(s['dec'])] for s in stars])
    ra, dec = np.deg2rad(radec).T
    units = np.column_stack((np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)))
    cra, cdec = np.deg2rad([ref['center_ra_deg'], ref['center_dec_deg']])
    center = [np.cos(cdec)*np.cos(cra), np.cos(cdec)*np.sin(cra), np.sin(cdec)]
    indices = np.flatnonzero(units@center > .5)
    projected = WCS(fits.getheader(wcs_path)).all_world2pix(radec[indices], 0, quiet=True)
    w, h = rec['width'], rec['height']
    inside = np.isfinite(projected).all(axis=1) & (projected[:, 0] >= 12) & (projected[:, 0] < w-12) & (projected[:, 1] >= 12) & (projected[:, 1] < h-12)
    extraction = RAW/'downsample-4/field.axy'
    data = fits.getdata(extraction)
    header = fits.getheader(extraction)
    if (header['IMAGEW'], header['IMAGEH']) != (w, h):
        raise ValueError('extraction dimensions mismatch')
    xy = np.column_stack((data['X'], data['Y'])).astype(float)-1
    points = []
    for local in np.flatnonzero(inside):
        star = stars[indices[local]]
        location = projected[local]
        distances = np.linalg.norm(xy-location, axis=1)
        nearby = np.flatnonzero(distances <= 45)
        choices = list(nearby[np.argsort(distances[nearby])[:3]])
        if len(nearby):
            brightest = int(nearby[np.argmax(data['FLUX'][nearby])])
            if brightest not in choices:
                choices.append(brightest)
        points.append(dict(hyg_id=int(star['id']), name=star['proper'] or star['bf'] or star['con'],
            mag=float(star['mag']), radec=radec[indices[local]].tolist(), review_center=location.tolist(),
            choices=[dict(row=int(i), x=float(xy[i, 0]), y=float(xy[i, 1])) for i in choices],
            label='pending', selected_row=None))
    points = distributed(points, w, h)
    frame = dict(id=RID, image=str(source.relative_to(ROOT)), source_sha256=rec['clean_sha256'],
        width=w, height=h, points=points)
    out.mkdir(parents=True)
    for name in ('candidates', 'review'):
        write_json(out/f'{name}.json', dict(split='development', frames=[frame]))
    paths = [Path(__file__).resolve(), MANIFEST, CATALOG, source, RAW/'reference.json', wcs_path,
        corr_path, extraction, out/'candidates.json', *reports().values(),
        *(Path(__file__).with_name(n) for n in ['collection_run.py', 'compare_local_wcs.py', 'robustness_geometry.py']),
        *(MANIFEST.parent/n for n in ['robustness-results.json', 'collection-wcs-review.json'])]
    write_json(out/'protocol.json', dict(iteration=40, id=RID, split='development', holdout=False,
        hypothesis='large Starglyph/external disagreement may reflect field geometry or invalid external correspondences; discriminate on visible sources',
        selection='WCS-proposed HYG mag<=5.5, three brightest per 4x4 cell; no selection by residual or detection availability',
        criteria=['review every candidate; never alter coordinates or delete inconvenient probes',
            'compare all saved models on the same reviewed physical sources and native r8/r12 centers',
            'report all edges and groups outside each input, including all 108 external corr rows',
            'identities are conditional on visual pattern review, WCS stays pending, no absolute GT',
            'no solver, fit, threshold change or holdout; freeze review before geometry measurements'],
        budget=dict(new_solver_calls=0, new_wcs_calls=0, new_fit_calls=0),
        hashes={str(p.relative_to(ROOT)): digest(p) for p in paths}))
    preview(frame, out)


def validate(out):
    p = read(out/'protocol.json')
    if (p['id'], p['split'], p['holdout']) != (RID, 'development', False):
        raise ValueError('development only')
    selected()
    for name, sha in p['hashes'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('frozen input changed: '+name)
    return p


def group_masks(xy, detections, corr, width, height):
    def outside(points):
        return np.min(np.linalg.norm(xy[:, None, :]-points[None, :, :], axis=2), axis=1) >= 12
    sg = outside(detections)
    ext = outside(corr)
    return dict(all_reviewed=np.ones(len(xy), dtype=bool), outside_both_starglyph_inputs=sg,
        outside_external_corr=ext, outside_all_inputs=sg & ext,
        left_10pct=xy[:, 0] < width*.1, right_10pct=xy[:, 0] > width*.9,
        top_10pct=xy[:, 1] < height*.1, bottom_10pct=xy[:, 1] > height*.9)


def measure(out):
    started = time.monotonic()
    validate(out)
    review = read(out/'review.json')
    validate_review(read(out/'candidates.json'), review)
    if (out/'review-protocol.json').exists():
        raise ValueError('fresh measurement required')
    frame = review['frames'][0]
    accepted, seen = [], set()
    for p in frame['points']:
        if p['label'] not in ('visible_source', 'ambiguous', 'blend', 'not_visible', 'occluded'):
            raise ValueError('finish every review')
        if p['label'] != 'visible_source':
            continue
        source = next((c for c in p['choices'] if c['row'] == p['selected_row']), None)
        if source is None or source['row'] in seen:
            raise ValueError('invalid or duplicate physical source')
        seen.add(source['row'])
        accepted.append(p | dict(xy=[source['x'], source['y']]))
    if not accepted:
        raise ValueError('no visible sources')
    write_json(out/'review-protocol.json', dict(review_sha256=digest(out/'review.json'), protocol_sha256=digest(out/'protocol.json')))
    saved = {a: read(p) for a, p in reports().items()}
    xy = np.array([p['xy'] for p in accepted])
    world = np.array([p['radec'] for p in accepted])
    ref = read(RAW/'reference.json')
    wcs = WCS(fits.getheader(ROOT/'prototype'/ref['wcs_file']))
    corr = fits.getdata(ROOT/'prototype'/ref['correspondences_file'])
    corr_xy = np.column_stack((corr['field_x'], corr['field_y'])).astype(float)-1
    dets = np.array([[d['x'], d['y']] for r in saved.values() for d in r['report']['detections']])
    masks = group_masks(xy, dets, corr_xy, frame['width'], frame['height'])
    predicted = {a: project(r['camera'], world) for a, r in saved.items()}
    predicted['external'] = wcs.all_world2pix(world, 0)
    if not all(np.isfinite(p).all() for p in predicted.values()):
        raise ValueError('nonfinite geometry')
    with Image.open(ROOT/frame['image']) as im:
        gray = np.asarray(im.convert('RGB'), dtype=np.float32)@np.array([.2126, .7152, .0722], dtype=np.float32)
    coords = dict(external_centroid=xy, **{f'native_r{r}': np.array([native_centroid(gray, p, r) for p in xy]) for r in [8, 12]})
    measurements = {}
    for method, points in coords.items():
        errors = {a: np.linalg.norm(p-points, axis=1) for a, p in predicted.items()}
        measurements[method] = dict(centroid_shift=stats(np.linalg.norm(points-xy, axis=1)),
            groups={g: {a: stats(e[m]) if m.any() else dict(count=0) for a, e in errors.items()} for g, m in masks.items()},
            errors_px={a: e.tolist() for a, e in errors.items()})
    rows = [dict(id=p['id'], hyg_id=p['hyg_id'], name=p['name'], axy_row=p['selected_row'], radec=p['radec'],
        coordinates={a: points[i].tolist() for a, points in coords.items()},
        groups=[g for g, m in masks.items() if m[i]],
        predicted_xy={a: points[i].tolist() for a, points in predicted.items()}) for i, p in enumerate(accepted)]
    write_json(out/'results.json', dict(iteration=40, id=RID, split='development', holdout=False,
        protocol_sha256=digest(out/'protocol.json'), review_protocol_sha256=digest(out/'review-protocol.json'),
        candidates=len(frame['points']), sources=rows,
        excluded=[dict(id=p['id'], name=p['name'], label=p['label'], note=p.get('note')) for p in frame['points'] if p['label'] != 'visible_source'],
        measurements=measurements, saved_quality={a: r['report']['quality'] for a, r in saved.items()},
        external_corr_rows=len(corr), external_corr_high_weight=int(sum(corr['match_weight'] >= .95)),
        elapsed_s=time.monotonic()-started, independent_ground_truth=False, production_changed=False, automatic_improvement_confirmed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'validate', 'measure'])
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.stage](args.out_dir.resolve())
