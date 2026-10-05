#!/usr/bin/env python3
"""Describe saved inlier coverage after the frozen iteration-40 source audit."""
import argparse
from pathlib import Path

import numpy as np

from collection_run import ROOT, digest, write_json
from robustness_reference_audit import read, reports, validate, RID


def extent(detections, width, height):
    if not detections:
        return dict(count=0, bounds_xyxy=None, width_fraction=None, height_fraction=None)
    xy = np.array([[d['x'], d['y']] for d in detections])
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    return dict(count=len(xy), bounds_xyxy=[*lo.tolist(), *hi.tolist()],
        width_fraction=float((hi[0]-lo[0])/width), height_fraction=float((hi[1]-lo[1])/height))


def prepare(out):
    validate(out)
    if (out/'coverage-protocol.json').exists():
        raise ValueError('fresh coverage protocol required')
    paths = [Path(__file__).resolve(), out/'protocol.json', out/'review.json', out/'review-protocol.json', out/'results.json', *reports().values()]
    write_json(out/'coverage-protocol.json', dict(id=RID, split='development', holdout=False,
        scope='post-measurement descriptive inlier coverage, no new fit, no hypothesis tuning',
        hashes={str(p.relative_to(ROOT)): digest(p) for p in paths}))


def run(out):
    validate(out)
    p = read(out/'coverage-protocol.json')
    if (p['id'], p['split'], p['holdout']) != (RID, 'development', False):
        raise ValueError('development only')
    for name, sha in p['hashes'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('coverage input changed: '+name)
    if (out/'coverage.json').exists():
        raise ValueError('fresh coverage run required')
    audit = read(out/'results.json')
    rows = {}
    for arm, path in reports().items():
        report = read(path)
        dets = report['report']['detections']
        inliers = [d for d in dets if d['inlier']]
        if len(inliers) != report['report']['quality']['n_inliers']:
            raise ValueError('inlier count mismatch')
        xy = np.array([[d['x'], d['y']] for d in inliers])
        sources = []
        for s in audit['sources']:
            measured = np.array(s['coordinates']['external_centroid'])
            sources.append(dict(source_id=s['id'], nearest_final_inlier_px=float(np.linalg.norm(xy-measured, axis=1).min())))
        rows[arm] = dict(detections=extent(dets, report['width'], report['height']),
            inliers=extent(inliers, report['width'], report['height']), sources=sources,
            final_inlier_indices=[i for i, d in enumerate(dets) if d['inlier']],
            note='proximity to an inlier is not a confirmed catalog identity')
    write_json(out/'coverage.json', dict(iteration=40, id=RID, split='development', holdout=False,
        protocol_sha256=digest(out/'coverage-protocol.json'), arms=rows))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'run'])
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    globals()[a.stage](a.out_dir.resolve())
