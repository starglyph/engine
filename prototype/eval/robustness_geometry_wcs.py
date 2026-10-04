#!/usr/bin/env python3
"""Check existing external WCS candidates on the frozen development review."""
import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from collection_run import ROOT, digest, write_json
from compare_local_wcs import stats
from robustness_geometry import selected, validate_review


def check(out, output):
    protocol = json.loads((out / 'protocol.json').read_text())
    review = json.loads((out / 'review.json').read_text())
    if protocol['split'] != 'development' or review['split'] != 'development':
        raise ValueError('development required')
    if [f['id'] for f in review['frames']] != protocol['ids']:
        raise ValueError('review selection differs')
    selected(protocol['ids'])
    if digest(out / 'candidates.json') != protocol['candidates_sha256']:
        raise ValueError('candidate file changed')
    validate_review(json.loads((out / 'candidates.json').read_text()), review)
    results = []
    for frame in review['frames']:
        refpath = ROOT / f"prototype/artifacts/robustness-baseline/wcs-run/{frame['id']}/reference.json"
        reference = json.loads(refpath.read_text())
        if reference['source_sha256'] != frame['source_sha256']:
            raise ValueError('reference image mismatch')
        row = dict(id=frame['id'], reference_sha256=digest(refpath), status=reference['status'],
                   review_status=reference.get('review_status', 'unresolved'))
        if reference['status'] == 'solved_candidate':
            paths = {k: ROOT / 'prototype' / reference[k + '_file'] for k in ('wcs', 'correspondences')}
            for k, path in paths.items():
                if digest(path) != reference[k + '_sha256']:
                    raise ValueError('external artifact changed')
            corr = fits.getdata(paths['correspondences'])
            corr_xy = np.column_stack((corr['field_x'], corr['field_y'])) - 1
            points = [p for p in frame['points'] if p['label'] == 'visible_source']
            xy = np.array([[c['x'], c['y']] for p in points for c in p['choices'] if c['row'] == p['selected_row']])
            world = np.array([[p['ra_deg'], p['dec_deg']] for p in points])
            predicted = WCS(fits.getheader(paths['wcs'])).all_world2pix(world, 0)
            if not np.isfinite(predicted).all():
                raise ValueError('non-finite external projection')
            errors = np.linalg.norm(predicted - xy, axis=1)
            nearest = np.min(np.linalg.norm(xy[:, None] - corr_xy[None, :], axis=2), axis=1)
            withheld = nearest >= 12
            row.update(wcs_sha256=digest(paths['wcs']), correspondences_sha256=digest(paths['correspondences']),
                correspondences_count=len(corr), high_weight_count=int((corr['match_weight'] >= .95).sum()),
                all_reviewed=stats(errors),
                outside_all_corr_sources=stats(errors[withheld]) if withheld.any() else dict(count=0),
                sources=[dict(hyg_id=p['hyg_id'], xy=m.tolist(), predicted_xy=q.tolist(), error_px=float(e),
                    outside_all_corr_sources=bool(w), nearest_corr_source_px=float(d))
                    for p, m, q, e, w, d in zip(points, xy, predicted, errors, withheld, nearest)])
        results.append(row)
    write_json(output, dict(iteration=5, split='development', review_sha256=digest(out / 'review.json'),
        frames=results, ground_truth_promotions=0,
        limitations=['all corr rows excluded from withheld group regardless of match weight',
                    'identities remain conditional on visual pattern review; no WCS refit']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    check(a.out, a.output)


if __name__ == '__main__':
    main()
