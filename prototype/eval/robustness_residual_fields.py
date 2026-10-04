#!/usr/bin/env python3
"""Describe frozen development residual fields without fitting any camera."""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np

from collection_run import ROOT, digest, write_json
from compare_local_wcs import project
from robustness_geometry import report_path, selected

GEOMETRY = ROOT/'docs/experiments/robustness-iteration-5-geometry.json'
REVIEW = ROOT/'docs/experiments/robustness-iteration-5-review.json'
PRIOR = ROOT/'docs/experiments/robustness-iteration-5-protocol.json'
ARMS = ('control', 'footprint')
RADIAL_BINS = (0., .35, .70, 1.000001)


def prepare(out):
    geometry = json.loads(GEOMETRY.read_text())
    records = selected([f['id'] for f in geometry['frames']])
    if geometry['split'] != 'development':
        raise ValueError('development required')
    if digest(REVIEW) != geometry['review_sha256'] or digest(PRIOR) != geometry['protocol_sha256']:
        raise ValueError('prior annotations/protocol changed')
    prior = json.loads(PRIOR.read_text())
    files = [GEOMETRY, REVIEW, PRIOR, Path(__file__).resolve(),
        ROOT/'prototype/eval/robustness_residual_plot.py', ROOT/'prototype/eval/compare_local_wcs.py',
        ROOT/'prototype/eval/robustness_geometry.py', ROOT/'prototype/eval/collection_run.py',
        ROOT/'data/samples/sky-samples/manifest.json']
    for record in records:
        for arm in ARMS:
            path = report_path(record['id'], arm)
            if digest(path) != prior['hashes'][str(path.relative_to(ROOT))]:
                raise ValueError('saved camera changed')
            files.append(path)
    if out.exists():
        raise ValueError('fresh output directory required')
    out.mkdir(parents=True)
    write_json(out/'protocol.json', dict(iteration=12, split='development', ids=[r['id'] for r in records],
        hashes={str(p.relative_to(ROOT)):digest(p) for p in files}, arms=list(ARMS), radial_bins=list(RADIAL_BINS),
        edge_fraction=.1, grid=[3,3], residual_sign='predicted minus measured; x right, y down',
        radial_origin='unchanged camera principal point (width/2,height/2)',
        radial_normalization='distance / half image diagonal',
        tangential_sign='clockwise from outward radial direction in image coordinates',
        angular_comparison_min_residual_px=1., plot_arrow_magnification=15.,
        criteria=['reproduce every saved iteration-5 projected position and error',
                  'retain all 117 probes, outside-input and edge groups, including sparse or empty cells',
                  'repeat summaries with both previously measured aperture centroids',
                  'do not infer a universal lens model from three heterogeneous conditional fields'],
        environment=dict(python=platform.python_version(), numpy=np.__version__),
        interpretation='descriptive decomposition only; no pose, translation or distortion fit', holdout=False))


def validate(out):
    protocol = json.loads((out/'protocol.json').read_text())
    if protocol['split'] != 'development' or protocol['holdout']:
        raise ValueError('development required')
    records = selected(protocol['ids'])
    if [r['id'] for r in records] != protocol['ids']:
        raise ValueError('selection changed')
    for path, sha in protocol['hashes'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('frozen input changed: '+path)
    return protocol, records


def decompose(xy, residuals, width, height):
    xy, residuals = np.asarray(xy, dtype=float), np.asarray(residuals, dtype=float)
    offset = xy-[width/2, height/2]
    radius = np.linalg.norm(offset, axis=1)
    valid = radius > 1e-12
    radial_unit = np.divide(offset, radius[:,None], out=np.zeros_like(offset), where=valid[:,None])
    tangential_unit = np.column_stack((-radial_unit[:,1], radial_unit[:,0]))
    radial = np.sum(residuals*radial_unit, axis=1)
    tangential = np.sum(residuals*tangential_unit, axis=1)
    return radius/np.hypot(width/2,height/2), radial, tangential, valid


def vector_metrics(residuals, radial, tangential, valid, half_diagonal):
    if not len(residuals):
        return dict(count=0)
    energy = np.sum(residuals**2, axis=1)
    result = dict(count=len(residuals), rms_px=float(np.sqrt(np.mean(energy))),
        rms_per_mille_half_diagonal=float(1000*np.sqrt(np.mean(energy))/half_diagonal),
        mean_xy_px=np.mean(residuals, axis=0).tolist(), radial_valid_count=int(sum(valid)))
    if valid.any():
        rr, tt = radial[valid], tangential[valid]
        total = np.sum(rr**2+tt**2)
        result.update(radial_mean_px=float(np.mean(rr)), radial_median_px=float(np.median(rr)),
            tangential_mean_px=float(np.mean(tt)), radial_rms_px=float(np.sqrt(np.mean(rr**2))),
            tangential_rms_px=float(np.sqrt(np.mean(tt**2))),
            radial_energy_fraction=float(np.sum(rr**2)/total) if total else None)
    return result


def groups(xy, outside, width, height, rho):
    x,y = np.asarray(xy).T
    result = dict(all_reviewed=np.ones(len(x),dtype=bool), outside_both_inputs=np.asarray(outside,dtype=bool),
        left_10pct=x<width*.1, right_10pct=x>width*.9, top_10pct=y<height*.1, bottom_10pct=y>height*.9)
    for i,(lo,hi) in enumerate(zip(RADIAL_BINS, RADIAL_BINS[1:])):
        result[f'radius_{i}'] = (rho>=lo)&(rho<hi)
    for row in range(3):
        for col in range(3):
            result[f'cell_{row}_{col}'] = (x>=col*width/3)&(x<(col+1)*width/3)&(y>=row*height/3)&(y<(row+1)*height/3)
    return result


def agreement(a, b, mask):
    a,b = a[mask],b[mask]
    if not len(a):
        return dict(count=0)
    norms_a,norms_b = np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1)
    nonzero = (norms_a>=1.)&(norms_b>=1.)
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return dict(count=len(a), field_cosine=float(np.sum(a*b)/denominator) if denominator else None,
        angular_sample_count=int(sum(nonzero)),
        median_cosine=float(np.median(np.sum(a[nonzero]*b[nonzero],axis=1)/(norms_a[nonzero]*norms_b[nonzero]))) if nonzero.any() else None,
        camera_disagreement_rms_px=float(np.sqrt(np.mean(np.sum((a-b)**2,axis=1)))))


def measure(out):
    protocol, records = validate(out)
    started = time.monotonic()
    geometry = json.loads(GEOMETRY.read_text())
    reviews = {f['id']:f for f in json.loads(REVIEW.read_text())['frames']}
    frames = []
    for record in records:
        previous = next(f for f in geometry['frames'] if f['id']==record['id'])
        sources = previous['sources']
        annotated = {p['id']:p for p in reviews[record['id']]['points']}
        radec = [[annotated[s['source_id']]['ra_deg'],annotated[s['source_id']]['dec_deg']] for s in sources]
        xy = np.array([s['xy'] for s in sources])
        w,h = record['width'],record['height']
        rho,_,_,_ = decompose(xy,np.zeros_like(xy),w,h)
        selections = groups(xy,[s['outside_both_inputs'] for s in sources],w,h,rho)
        arms, residual_fields = {},{}
        for arm in ARMS:
            report = json.loads(report_path(record['id'],arm).read_text())
            if report['source_sha256'] != record['clean_sha256'] or [report['width'],report['height']] != [w,h]:
                raise ValueError('saved camera image mismatch')
            prediction = project(report['camera'],radec)
            saved = np.array([s[arm]['predicted_xy'] for s in sources])
            if not np.allclose(prediction,saved,rtol=0,atol=1e-8):
                raise ValueError('saved projections differ')
            residual = prediction-xy
            if not np.allclose(np.linalg.norm(residual,axis=1),[s[arm]['error_px'] for s in sources],rtol=0,atol=1e-8):
                raise ValueError('saved errors differ')
            _,radial,tangential,valid = decompose(xy,residual,w,h)
            def metrics(vectors, rr, tt, vv):
                return {name:vector_metrics(vectors[m],rr[m],tt[m],vv[m],np.hypot(w/2,h/2)) for name,m in selections.items()}
            alternate = {}
            for radius in (8,12):
                native_xy = np.array([s[f'native_r{radius}_xy'] for s in sources])
                vectors = prediction-native_xy
                _,rr,tt,vv = decompose(native_xy,vectors,w,h)
                alternate[f'native_r{radius}'] = metrics(vectors,rr,tt,vv)
            arms[arm] = dict(camera=report['camera'], projection_max_abs_difference=float(np.max(np.abs(prediction-saved))),
                residual_xy_px=residual.tolist(),radial_px=[float(v) if ok else None for v,ok in zip(radial,valid)],
                tangential_px=[float(v) if ok else None for v,ok in zip(tangential,valid)],
                groups=metrics(residual,radial,tangential,valid),centroid_sensitivity=alternate)
            residual_fields[arm] = residual
        frames.append(dict(id=record['id'],width=w,height=h,camera_metadata=record['research']['camera'],
            sources=[dict(source_id=s['source_id'],hyg_id=s['hyg_id'],name=s['name'],xy=s['xy'],
                outside_both_inputs=s['outside_both_inputs'],radius_normalized=float(r)) for s,r in zip(sources,rho)],
            arms=arms, agreement={name:agreement(residual_fields['control'],residual_fields['footprint'],selections[name])
                                  for name in ('all_reviewed','outside_both_inputs')},
            coverage=previous['coverage']))
    write_json(out/'results.json', dict(iteration=12,split='development',protocol_sha256=digest(out/'protocol.json'),
        frames=frames,elapsed_s=time.monotonic()-started,
        interpretation='predicted-minus-measured residual fields on conditional identities; no new camera or ground truth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','measure'])
    parser.add_argument('out',type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':prepare(args.out.resolve())
    else:measure(args.out.resolve())
