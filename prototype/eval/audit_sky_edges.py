#!/usr/bin/env python3
"""Audit external WCS correspondences in fixed outer 10% image strips.

Reports sparse measurements, not full-field accuracy. Catalog independence must
be stated separately for each reference run (HYG is shared with the engine).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from compare_local_wcs import compare, project, sha256, stats


def edge_margin(x, y, width, height):
    if width < 2 or height < 2 or not (0 <= x <= width-1 and 0 <= y <= height-1):
        raise ValueError('invalid edge pixel or dimensions')
    return float(min(x/(width-1), 1-x/(width-1), y/(height-1), 1-y/(height-1)))


def audit_points(reference, artifact, reviewed):
    result = compare(reference, artifact)  # verifies source, geometry, WCS and corr hashes
    if reference['status'] != 'solved_candidate' or artifact['report']['status'] != 'solved':
        return result
    for key in ('source_sha256', 'width', 'height'):
        if reviewed[key] != reference[key]:
            raise ValueError(f'review {key} mismatch')
    corr = fits.getdata(reference['correspondences_file'])
    sources = reviewed['points']
    measured = np.column_stack((corr['field_x'], corr['field_y']))-1
    world = np.column_stack((corr['index_ra'], corr['index_dec']))
    predicted = project(artifact['camera'], world)
    rows = []
    seen = set()
    for external, xy, sky, projection in zip(corr, measured, world, predicted):
        distances = [np.linalg.norm(xy-[p['x'], p['y']]) for p in sources]
        point = sources[int(np.argmin(distances))]
        if min(distances) > 1e-4 or point['label'] != 'visible_source' or point['id'] in seen:
            raise ValueError('external correspondence not a unique frozen visible source')
        seen.add(point['id'])
        weight = float(external['match_weight'])
        margin = edge_margin(*xy, reference['width'], reference['height'])
        rows.append({'source_id': point['id'], 'x': float(xy[0]), 'y': float(xy[1]),
                     'ra_deg': float(sky[0]), 'dec_deg': float(sky[1]),
                     'match_weight': weight, 'used_in_metrics': weight >= .95,
                     'edge_margin_fraction': margin, 'edge': margin <= .1,
                     'error_px': float(np.linalg.norm(projection-xy))})
    result['points'] = rows
    accepted = [p for p in rows if p['used_in_metrics']]
    for label, selected in [('edge_error', [p for p in accepted if p['edge']]),
                            ('interior_error', [p for p in accepted if not p['edge']])]:
        result[label] = stats(np.array([p['error_px'] for p in selected])) if selected else None
    result['edge_source_ids'] = [p['source_id'] for p in accepted if p['edge']]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=Path, nargs='+', required=True)
    parser.add_argument('--source-review', type=Path, required=True)
    parser.add_argument('--reports-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    review = json.loads(args.source_review.read_text())
    if review['schema_version'] != 1 or review['coordinates'] != 'exif_oriented_top_left_zero_based':
        raise ValueError('unsupported source review format')
    reviewed = {f['id']: f for f in review['frames']}
    runs = []
    for path in args.references:
        summary = json.loads(path.read_text())
        if summary['source_review_sha256'] != sha256(args.source_review):
            raise ValueError('source review SHA-256 mismatch')
        results = []
        for reference in summary['frames']:
            frame_id = Path(reference['source']).stem
            report_path = args.reports_dir / f'{frame_id}.json'
            result = audit_points(reference, json.loads(report_path.read_text()), reviewed[frame_id])
            results.append({'id': frame_id, 'report_sha256': sha256(report_path), **result})
        runs.append({'reference_file': str(path), 'reference_sha256': sha256(path), 'frames': results})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'schema_version': 1, 'status': 'diagnostic_only',
        'edge_definition': 'minimum distance to an image boundary <= 10% of that axis pixel span',
        'source_review_sha256': sha256(args.source_review), 'runs': runs}, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
