#!/usr/bin/env python3
"""Frozen development geometry review; reuse external centroids, never fit cameras."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image, ImageDraw, ImageEnhance

from collection_run import ROOT, digest, select_records, write_json
from compare_local_wcs import project, stats
from review_external_stars import compare_frame

IDS = ('wm_r_143159342', 'wm_r_132162731', 'wm_r_149276071')
MANIFEST = ROOT / 'data/samples/sky-samples/manifest.json'
CATALOG = ROOT / 'data/catalogs/hyg_v42.csv.gz'
PRIOR = ROOT / 'prototype/artifacts/robustness-iteration-4/final/development'


def selected(ids):
    return select_records(json.loads(MANIFEST.read_text()), 'development', ids)


def report_path(rid, arm):
    return PRIOR / arm / rid / 'solve-reports' / (rid + '.json')


def prepare(out, ids):
    records = selected(ids)  # Split check precedes any image access.
    if out.exists():
        raise ValueError('choose a fresh output directory')
    out.mkdir(parents=True)
    with gzip.open(CATALOG, 'rt') as stream:
        stars = [s for s in csv.DictReader(stream) if s['mag'] and float(s['mag']) <= 5.5 and int(s['id']) > 0]
    radec = np.array([[float(s['ra']) * 15, float(s['dec'])] for s in stars])
    ra, dec = np.deg2rad(radec).T
    units = np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)))
    dependencies = [Path(__file__).resolve(), *(ROOT / 'prototype/eval' / name for name in
        ('collection_run.py', 'compare_local_wcs.py', 'review_external_stars.py'))]
    provenance = {str(p.relative_to(ROOT)): digest(p) for p in (MANIFEST, CATALOG, *dependencies)}
    frames = []
    for rec in records:
        rid = rec['id']
        image_path = MANIFEST.parent / rec['file']
        if digest(image_path) != rec['clean_sha256']:
            raise ValueError('image changed')
        reports = {a: json.loads(report_path(rid, a).read_text()) for a in ('control', 'footprint')}
        for a, report in reports.items():
            if report['source_sha256'] != rec['clean_sha256'] or report['report']['status'] != 'solved':
                raise ValueError('wrong source or unsolved report')
            provenance[str(report_path(rid, a).relative_to(ROOT))] = digest(report_path(rid, a))
        width, height = reports['control']['width'], reports['control']['height']
        cams = [reports[a]['camera'] for a in ('control', 'footprint')]
        front = np.logical_and.reduce([units @ np.asarray(c['world_to_camera'])[2] > .1 for c in cams])
        indices = np.flatnonzero(front)
        projections = np.array([project(c, radec[front]) for c in cams])
        midpoint = projections.mean(axis=0)
        # Observable sky limits chosen from image overviews, before residuals.
        ymin = 160 if rid == 'wm_r_149276071' else 12
        ymax = 3500 if rid == 'wm_r_132162731' else height - 12
        inside = ((midpoint[:, 0] >= 12) & (midpoint[:, 0] < width - 12)
                  & (midpoint[:, 1] >= ymin) & (midpoint[:, 1] < ymax))
        scale = 4 if rid == 'wm_r_149276071' else 2
        axy_path = ROOT / f'prototype/artifacts/robustness-baseline/wcs-run/{rid}/downsample-{scale}/field.axy'
        axy = fits.getdata(axy_path)
        header = fits.getheader(axy_path)
        if (header['IMAGEW'], header['IMAGEH']) != (width, height):
            raise ValueError('external extraction dimensions differ')
        xy = np.column_stack((axy['X'], axy['Y'])).astype(float) - 1
        provenance[str(axy_path.relative_to(ROOT))] = digest(axy_path)
        provenance[str(image_path.relative_to(ROOT))] = digest(image_path)
        points = []
        for local in np.flatnonzero(inside):
            star = stars[indices[local]]
            center = midpoint[local]
            distances = np.linalg.norm(xy - center, axis=1)
            nearby = np.flatnonzero(distances <= 45)
            choices = list(nearby[np.argsort(distances[nearby])[:3]])
            if len(nearby):
                brightest = int(nearby[np.argmax(axy['FLUX'][nearby])])
                if brightest not in choices:
                    choices.append(brightest)
            points.append(dict(hyg_id=int(star['id']), name=star['proper'] or star['bf'] or star['con'],
                mag=float(star['mag']), ra_deg=float(star['ra']) * 15, dec_deg=float(star['dec']),
                review_center=center.tolist(), choices=[dict(row=int(i), x=float(xy[i, 0]), y=float(xy[i, 1]),
                    flux=float(axy['FLUX'][i])) for i in choices], label='pending', selected_row=None))
        points.sort(key=lambda p: (p['mag'], p['hyg_id']))
        # Freeze a bounded, spatially distributed visual review before errors
        # are computed; do not select by presence or distance of an extraction.
        cells, distributed = {}, []
        for point in points:
            x, y = point['review_center']
            cell = (min(int(4*x/width), 3), min(int(4*y/height), 3))
            if cells.get(cell, 0) < 3:
                distributed.append(point)
                cells[cell] = cells.get(cell, 0) + 1
        points = distributed
        for i, point in enumerate(points):
            point['id'] = i
        frames.append(dict(id=rid, source_sha256=rec['clean_sha256'], image=str(image_path.relative_to(ROOT)),
            width=width, height=height, extraction=str(axy_path.relative_to(ROOT)),
            sky_bounds=[12, ymin, width - 12, ymax], points=points))
        preview(frames[-1], out)
    write_json(out / 'review.json', dict(schema_version=1, split='development', frames=frames))
    write_json(out / 'candidates.json', dict(schema_version=1, split='development', frames=frames))
    write_json(out / 'protocol.json', dict(split='development', ids=[r['id'] for r in records], hashes=provenance,
        candidates_sha256=digest(out / 'candidates.json'),
        max_magnitude=5.5, review_selection='three brightest candidates per 4x4 image cell',
        association_radius_px=45, exclusion_from_both_solver_inputs_px=12,
        candidates='union-symmetric camera midpoint; identities require visual review',
        measurements='pre-existing Astrometry.net X/Y minus one; never camera-fitted',
        geometry='conditional on reviewed identities; no ground-truth promotion',
        primary='same visible sources absent from both final solver input lists',
        sensitivity='image-only positive-flux centroids, radii 8 and 12 px, background annulus 16-22 px',
        acceptance='report all reviewed points and edge groups, including regressions; no tuning'))


def preview(frame, out):
    raw = Image.open(ROOT / frame['image']).convert('RGB')
    bright = ImageEnhance.Brightness(raw).enhance(2)
    overview = bright.copy()
    overview.thumbnail((1100, 1100))
    draw = ImageDraw.Draw(overview)
    sx, sy = overview.width / raw.width, overview.height / raw.height
    for point in frame['points']:
        x, y = point['review_center']
        x, y = x * sx, y * sy
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline='cyan')
        draw.text((x + 5, y), str(point['id']), fill='yellow')
    overview.save(out / (frame['id'] + '-identities.jpg'))
    for start in range(0, len(frame['points']), 20):
        sheet = Image.new('RGB', (1024, 1400))
        draw = ImageDraw.Draw(sheet)
        for j, p in enumerate(frame['points'][start:start + 20]):
            left, top = j % 4 * 256, j // 4 * 280
            x0, y0 = np.rint(p['review_center']).astype(int) - 64
            tile = bright.crop((x0, y0, x0 + 128, y0 + 128)).resize((256, 256))
            sheet.paste(tile, (left, top + 24))
            draw.text((left + 2, top + 2), f"{p['id']} H{p['hyg_id']} {p['name']} {p['mag']:.1f}", fill='white')
            for n, option in enumerate(p['choices']):
                x, y = left + 2 * (option['x'] - x0), top + 24 + 2 * (option['y'] - y0)
                draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline='cyan')
                draw.text((x + 8, y - 8), str(n), fill='yellow')
        sheet.save(out / f"{frame['id']}-sources-{start // 20}.jpg")


def group_metrics(rows, mask):
    selected_rows = [r for r, take in zip(rows, mask) if take]
    if not selected_rows:
        return dict(count=0)
    result = {a: stats(np.array([r[a]['error_px'] for r in selected_rows])) for a in ('control', 'footprint')}
    delta = np.array([r['footprint']['error_px'] - r['control']['error_px'] for r in selected_rows])
    return dict(count=len(selected_rows), **result, median_paired_error_change_px=float(np.median(delta)),
        improved=sum(delta < 0).item(), worsened=sum(delta > 0).item())


def validate_review(template, review):
    """Review may label fixed candidates, never change a measured coordinate."""
    if template['split'] != review['split'] or len(template['frames']) != len(review['frames']):
        raise ValueError('review selection differs')
    for original, annotated in zip(template['frames'], review['frames']):
        for key, value in original.items():
            if key != 'points' and annotated.get(key) != value:
                raise ValueError('frozen frame changed')
        if len(original['points']) != len(annotated['points']):
            raise ValueError('frozen candidate count changed')
        for a, b in zip(original['points'], annotated['points']):
            if {k: v for k, v in a.items() if k not in ('label', 'selected_row', 'note')} != {
                    k: v for k, v in b.items() if k not in ('label', 'selected_row', 'note')}:
                raise ValueError('frozen candidate changed')


def native_centroid(gray, xy, radius):
    """Image-only sensitivity check around the fixed external extraction."""
    x, y = xy
    x0, x1 = max(0, int(x)-23), min(gray.shape[1], int(x)+24)
    y0, y1 = max(0, int(y)-23), min(gray.shape[0], int(y)+24)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    distance = np.hypot(xx-x, yy-y)
    patch = gray[y0:y1, x0:x1]
    ring = (distance >= 16) & (distance <= 22)
    if ring.sum() < 10:
        raise ValueError('insufficient background annulus')
    weights = np.maximum(patch - np.median(patch[ring]), 0) * (distance <= radius)
    mass = weights.sum()
    if mass <= 0:
        raise ValueError('no positive source flux')
    return [float((xx*weights).sum()/mass), float((yy*weights).sum()/mass)]


def measure(out, output):
    protocol = json.loads((out / 'protocol.json').read_text())
    if protocol['split'] != 'development':
        raise ValueError('development required')
    selected(protocol['ids'])
    for path, expected in protocol['hashes'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('frozen input changed: ' + path)
    review = json.loads((out / 'review.json').read_text())
    if digest(out / 'candidates.json') != protocol['candidates_sha256']:
        raise ValueError('candidate file changed')
    validate_review(json.loads((out / 'candidates.json').read_text()), review)
    if review['split'] != 'development' or [r['id'] for r in review['frames']] != protocol['ids']:
        raise ValueError('review selection differs')
    results = []
    for frame in review['frames']:
        if not any(p['label'] == 'visible_source' for p in frame['points']):
            raise ValueError('no reviewed visible sources')
        reports = {a: json.loads(report_path(frame['id'], a).read_text()) for a in ('control', 'footprint')}
        sources, stars, matches, used = [], {}, [], set()
        for p in frame['points']:
            if p['label'] not in ('visible_source', 'ambiguous', 'blend', 'not_visible', 'occluded'):
                raise ValueError('finish visual review first')
            if p['label'] != 'visible_source':
                continue
            source = next((c for c in p['choices'] if c['row'] == p['selected_row']), None)
            if source is None:
                raise ValueError('selected row is not a frozen candidate')
            if source['row'] in used:
                raise ValueError('external centroid reused')
            used.add(source['row'])
            sources.append(dict(id=p['id'], label=p['label'], x=source['x'], y=source['y']))
            stars[p['hyg_id']] = p
            matches.append(dict(source_id=p['id'], star_id=p['hyg_id'], method='visual_pattern_transfer'))
        common = dict(frame, matches=matches)
        measured = {a: compare_frame(common, dict(frame, points=sources), stars, r) for a, r in reports.items()}
        rows = []
        for i, s in enumerate(sources):
            point = next(p for p in frame['points'] if p['id'] == s['id'])
            distances = {}
            for arm, report in reports.items():
                d = np.array([[p['x'], p['y']] for p in report['report']['detections']])
                distances[arm] = float(np.min(np.linalg.norm(d - [s['x'], s['y']], axis=1)))
            rows.append(dict(source_id=s['id'], hyg_id=point['hyg_id'], name=point['name'],
                axy_row=point['selected_row'], xy=[s['x'], s['y']], nearest_solver_detection_px=distances,
                outside_both_inputs=all(v >= 12 for v in distances.values()),
                **{a: dict(error_px=measured[a]['points'][i]['error_px'],
                    predicted_xy=[measured[a]['points'][i]['predicted_x'], measured[a]['points'][i]['predicted_y']]) for a in reports}))
        w, h = frame['width'], frame['height']
        groups = {'all_reviewed': [True] * len(rows), 'outside_both_inputs': [r['outside_both_inputs'] for r in rows],
            'left_10pct': [r['xy'][0] < .1*w for r in rows], 'right_10pct': [r['xy'][0] > .9*w for r in rows],
            'top_10pct': [r['xy'][1] < .1*h for r in rows], 'bottom_10pct': [r['xy'][1] > .9*h for r in rows]}
        with Image.open(ROOT / frame['image']) as im:
            rgb = np.asarray(im.convert('RGB'), dtype=np.float32)
            gray = rgb @ np.array([.2126, .7152, .0722], dtype=np.float32)
        sensitivity = {}
        for radius in (8, 12):
            alternate, shifts = [], []
            for row in rows:
                xy = native_centroid(gray, row['xy'], radius)
                shifts.append(float(np.linalg.norm(np.array(xy)-row['xy'])))
                row[f'native_r{radius}_xy'] = xy
                alternate.append({a: dict(error_px=float(np.linalg.norm(np.array(xy)-row[a]['predicted_xy'])))
                                  for a in reports})
            sensitivity[f'native_r{radius}'] = dict(centroid_shift=stats(np.array(shifts)),
                groups={k: group_metrics(alternate, mask) for k, mask in groups.items()})
        results.append(dict(id=frame['id'], independent_ground_truth=False,
            coverage=measured['control'].get('coverage'), groups={k: group_metrics(rows, v) for k, v in groups.items()},
            centroid_sensitivity=sensitivity,
            excluded=[dict(id=p['id'], hyg_id=p['hyg_id'], label=p['label']) for p in frame['points'] if p['label'] != 'visible_source'],
            sources=rows, quality={a: r['report']['quality'] for a, r in reports.items()}))
    write_json(output, dict(iteration=5, split='development', protocol_sha256=digest(out / 'protocol.json'),
        review_sha256=digest(out / 'review.json'), frames=results,
        limitations=['identities suggested by tested cameras then visually reviewed; conditional, not absolute ground truth',
                    'catalogue-limited and brightness-selected probes do not certify unobserved field areas']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'measure'))
    p.add_argument('out', type=Path)
    p.add_argument('--ids', default=','.join(IDS))
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.command == 'prepare':
        prepare(a.out, a.ids.split(','))
    else:
        if a.output is None:
            p.error('--output required')
        measure(a.out, a.output)


if __name__ == '__main__':
    main()
