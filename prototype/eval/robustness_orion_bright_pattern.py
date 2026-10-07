#!/usr/bin/env python3
"""Frozen wider bright-star diagnostic; conditional identities, development only."""
import argparse
import math
from pathlib import Path
import time

import numpy as np

from collection_run import ROOT, digest, write_json
from robustness_orion_fit import RID, METHODS, previous, read
from robustness_orion_neighbour_audit import IDS, tangent
from robustness_reference_audit import selected, MANIFEST
from review_external_stars import registration_check
from sky_statistics_experiment import point_in_polygon

RADIUS_DEG = 15.
LIMIT_PX = 12.


def choose_pattern(target, catalogue):
    """Select solely by catalogue geometry; neither target may support either fit."""
    eligible = []
    for p in catalogue:
        if p['id'] in IDS or p['mag'] > 5.5:
            continue
        xy = tangent(target['radec'], [p['radec']])[0]
        distance = math.degrees(math.atan(math.radians(math.hypot(*xy))))
        if not np.isfinite(xy).all() or distance > RADIUS_DEG:
            continue
        eligible.append(dict(id=p['id'], hyg_id=p['hyg_id'], name=p['name'], mag=p['mag'],
                             radec=p['radec'], tangent_xy=xy, distance_deg=distance,
                             quadrant=int(xy[0] >= 0) + 2*int(xy[1] >= 0)))
    eligible.sort(key=lambda p: (p['distance_deg'], p['hyg_id']))
    anchors = [next((p for p in eligible if p['quadrant'] == q), None) for q in range(4)]
    missing = [q for q, p in enumerate(anchors) if p is None]
    anchors = [p for p in anchors if p is not None]
    used = {p['id'] for p in anchors}
    controls = [p for p in eligible if p['id'] not in used][:2]
    return dict(status='ready' if not missing and len(controls) == 2 else 'insufficient_coverage',
                missing_quadrants=missing, eligible=eligible,
                points=[dict(p, role='anchor') for p in anchors] +
                       [dict(p, role='held_out') for p in controls])


def hull(points):
    """Monotone-chain convex hull; duplicate/collinear interior points disappear."""
    points = sorted(set(tuple(p) for p in points))
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    halves = []
    for order in (points, points[::-1]):
        half = []
        for p in order:
            while len(half) >= 2 and cross(half[-2], half[-1], p) <= 0:
                half.pop()
            half.append(p)
        halves.append(half[:-1])
    return halves[0] + halves[1]


def inside(point, polygon):
    if len(polygon) < 3:
        return False
    # Boundary is not interior: do not claim interpolation at an anchor edge.
    for a, b in zip(polygon, polygon[1:]+polygon[:1]):
        d = np.array(b)-a; v = np.array(point)-a
        if abs(d[0]*v[1]-d[1]*v[0]) <= 1e-10 and 0 <= v@d <= d@d:
            return False
    return bool(point_in_polygon(*point, polygon))


def evaluate(target, method):
    if target['status'] != 'ready':
        return dict(status='insufficient_coverage', method=method, fit_attempts=0)
    points = target['points'] + [dict(id=target['id'], tangent_xy=[0., 0.],
                                    role='held_out', coordinates=target['coordinates'])]
    registration = dict(points=[dict(source_id=p['id'], primary_xy=p['tangent_xy'], role=p['role']) for p in points])
    reviewed = dict(points=[dict(id=p['id'], x=p['coordinates'][method][0],
                                y=p['coordinates'][method][1], label='visible_source') for p in points])
    try:
        result = registration_check(registration, reviewed)
    except ValueError as e:
        return dict(status='degenerate_registration', method=method, fit_attempts=1, reason=str(e))
    anchors = [p for p in points if p['role'] == 'anchor']
    cat_hull = hull([p['tangent_xy'] for p in anchors])
    image_hull = hull([p['coordinates'][method] for p in anchors])
    for row, p in zip(result['points'], points):
        row.update(measured_xy=p['coordinates'][method],
                   inside_catalogue_hull=inside(p['tangent_xy'], cat_hull),
                   inside_image_hull=inside(p['coordinates'][method], image_hull))
    goal = result['points'][-1]
    passed = (goal['inside_catalogue_hull'] and goal['inside_image_hull'] and
              result['held_out_error']['max_px'] <= LIMIT_PX)
    return dict(status='conditional_local_registration', method=method, fit_attempts=1,
                criterion_passed=passed, catalogue_hull=cat_hull, image_hull=image_hull, **result)


def prepare(out):
    rec = selected()
    if out.exists():
        raise ValueError('fresh output required')
    paths = []
    for name in ['40-results', '40-review', '40-candidates']:
        p = previous(name)
        if digest(p) != read(previous('40-checks'))['artifact_hashes'][str(p.relative_to(ROOT))]:
            raise ValueError('published evidence changed: '+name)
        paths.append(p)
    paths += [previous('40-checks'), previous('52-checks')]
    protected = read(previous('52-checks'))['protected_hashes']
    for name, sha in protected.items():
        if digest(ROOT/name) != sha:
            raise ValueError('protected data changed: '+name)
        paths.append(ROOT/name)
    sources = read(previous('40-results'))['sources']
    review = {p['id']: p for p in read(previous('40-review'))['frames'][0]['points']}
    catalogue = [{k: review[s['id']][k] for k in ['id', 'hyg_id', 'name', 'mag', 'radec']} for s in sources]
    for s in sources:
        p = review[s['id']]
        if p['label'] != 'visible_source' or (p['hyg_id'], p['radec'], p['selected_row']) != (s['hyg_id'], s['radec'], s['axy_row']):
            raise ValueError('review identity changed')
    targets = []
    by_id = {s['id']: s for s in sources}
    for sid in IDS:
        s = by_id[sid]; choice = choose_pattern(s, catalogue)
        for p in choice['points']:
            p['coordinates'] = by_id[p['id']]['coordinates']
        targets.append(dict(id=sid, name=s['name'], hyg_id=s['hyg_id'], radec=s['radec'],
                            coordinates=s['coordinates'], **choice))
    out.mkdir(parents=True)
    write_json(out/'input.json', dict(id=RID, split='development', holdout=False,
               image=str((MANIFEST.parent/rec['file']).relative_to(ROOT)), catalogue=catalogue, targets=targets))
    paths += [out/'input.json', Path(__file__).resolve()]
    paths += [Path(__file__).with_name(n) for n in ['collection_run.py', 'robustness_orion_fit.py',
        'robustness_orion_neighbour_audit.py', 'robustness_reference_audit.py', 'review_external_stars.py',
        'compare_local_wcs.py', 'robustness_geometry.py', 'sky_statistics_experiment.py',
        'test_robustness_orion_bright_pattern.py']]
    write_json(out/'protocol.json', dict(iteration=53, id=RID, split='development', holdout=False,
        source_ids=list(IDS), methods=list(METHODS), radius_deg=RADIUS_DEG, limit_px=LIMIT_PX,
        selection='47 conditionally reviewed stars from40, mag<=5.5; exclude both targets; nearest per TAN quadrant within15deg, then nearest2 unused controls; distance/HYG order',
        universe_limitation='40 candidates were WCS-proposed brightest3 per4x4 cell, not a new blind all-sky identity sample',
        criteria=['same four anchors across all3 centroid methods; target plus2 controls never used in fit',
                  'target strictly inside both catalogue and measured anchor convex hulls; all3 held-out errors<=12 originalpx in all3 methods',
                  'report control extrapolation; do not expand radius, substitute identities or remove failures',
                  'reuse40 conditional visual labels; no GT promotion, global camera fit, solver/WCS or holdout',
                  'local homography tests consistency under an approximate model, cannot distinguish identity error from distortion'],
        protected_hashes=protected, hashes={str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))}))


def validate(out):
    p = read(out/'protocol.json')
    if (p['id'], p['split'], p['holdout'], p['source_ids']) != (RID, 'development', False, list(IDS)):
        raise ValueError('development only')
    selected()
    for name, sha in p['hashes'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('frozen input changed: '+name)
    return p


def render(out):
    validate(out)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    inp = read(out/'input.json'); rgb = np.asarray(Image.open(ROOT/inp['image']).convert('RGB'))
    for t in inp['targets']:
        pts = t['points'] + [dict(t, role='target', tangent_xy=[0., 0.])]
        xy = np.array([p['coordinates']['external_centroid'] for p in pts])
        lo = np.maximum(np.floor(xy.min(axis=0)-90).astype(int), 0)
        hi = np.minimum(np.ceil(xy.max(axis=0)+90).astype(int), [rgb.shape[1], rgb.shape[0]])
        fig, axes = plt.subplots(1, 2, figsize=(13, 7), layout='constrained')
        patch = np.minimum(rgb[lo[1]:hi[1], lo[0]:hi[0]].astype(float)*2, 255).astype('uint8')
        axes[0].imshow(patch, extent=[lo[0]-.5, hi[0]-.5, hi[1]-.5, lo[1]-.5])
        for p in pts:
            color = dict(anchor='#39d', held_out='#d90', target='#e46')[p['role']]
            for ax, coord in zip(axes, [p['coordinates']['external_centroid'], p['tangent_xy']]):
                ax.scatter(*coord, s=60, marker='+', color=color)
                ax.annotate(f"{p['id']}: {p['name']}", coord, xytext=(4, 5), textcoords='offset points', color=color, fontsize=8)
        axes[0].set(title='Photo x2 display; original measured centres', xlabel='Original x, px', ylabel='Original y, px (down)')
        axes[1].set(title='Catalogue only; no fitted alignment', xlabel='TAN degrees, east left', ylabel='TAN degrees, north up', aspect='equal')
        axes[1].grid(alpha=.2)
        fig.suptitle(f"{RID} / {t['name']}: blue anchors, orange controls, red target\nConditional identities; Davydushta CC BY4.0; HYG/Astronexus CC BY-SA4.0")
        fig.savefig(out/f"source-{t['id']}.png", dpi=150); plt.close(fig)


def measure(out):
    validate(out); started = time.monotonic()
    if (out/'results.json').exists():
        raise ValueError('fresh analysis required')
    targets = []
    for t in read(out/'input.json')['targets']:
        cases = [evaluate(t, method) for method in METHODS]
        targets.append(dict(source_id=t['id'], name=t['name'], selection_status=t['status'], cases=cases,
                            criterion_passed=all(c.get('criterion_passed', False) for c in cases)))
    write_json(out/'results.json', dict(iteration=53, id=RID, split='development', holdout=False,
        protocol_sha256=digest(out/'protocol.json'), targets=targets, elapsed_s=time.monotonic()-started,
        new_local_fit_attempts=sum(c['fit_attempts'] for t in targets for c in t['cases']),
        new_solver_calls=0, new_wcs_calls=0, new_camera_fits=0, production_changed=False,
        ground_truth_promoted=False, automatic_improvement_confirmed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'render', 'measure', 'validate'])
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args(); globals()[args.stage](args.out_dir.resolve())
