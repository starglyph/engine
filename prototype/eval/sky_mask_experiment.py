#!/usr/bin/env python3
"""Paired 17-frame manual-mask evaluation, with per-ID regression checks."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from smartphone_gate import check_inputs, check_reports, read_json, require, sha256


def in_sky(detection, annotation):
    x = (detection['x'] + 0.5) / annotation['width']
    y = (detection['y'] + 0.5) / annotation['height']
    polygon = annotation['sky_polygon']
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        cross = (x-a[0])*(b[1]-a[1]) - (y-a[1])*(b[0]-a[0])
        if abs(cross) <= 1e-12 and min(a[0],b[0]) <= x <= max(a[0],b[0]) and min(a[1],b[1]) <= y <= max(a[1],b[1]):
            return False
        if (a[1] > y) != (b[1] > y) and x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:
            inside = not inside
    return inside


def compare(manifest, baseline, masks_path, output):
    masks = {m['id']: m for m in read_json(masks_path)['masks']}
    entries = check_inputs(manifest, baseline)
    check_reports(manifest, baseline, output / 'unmasked')
    a, b = [dict(read_json(output / mode / 'summary.json')['config']) for mode in ('unmasked', 'masked')]
    require(b.pop('sky_masks', None) == str(masks_path), 'masked configuration missing')
    require(a == b, 'A/B configuration differs beyond sky mask')
    rows = []
    for entry in entries:
        frame_id = entry['id']
        a, b = [read_json(output / mode / 'solve-reports' / f'{frame_id}.json') for mode in ('unmasked', 'masked')]
        require(a['source_sha256'] == b['source_sha256'] == entry['sha256'], f'{frame_id}: input mismatch')
        require((a['width'], a['height']) == (b['width'], b['height']), f'{frame_id}: geometry changed')
        require(b.get('sky_mask') == masks.get(frame_id), f'{frame_id}: mask provenance mismatch')
        if frame_id not in masks:
            clean_a, clean_b = dict(a['report']), dict(b['report'])
            clean_a.pop('timing_ms', None)
            clean_b.pop('timing_ms', None)
            # Connected-component collection may reorder equal-flux points.
            # Compare their contents, preserving multiplicity and all fields.
            for report in (clean_a, clean_b):
                if 'detections' in report:
                    report['detections'] = sorted(report['detections'], key=lambda d: (-d['flux'], d['x'], d['y']))
            require(clean_a == clean_b and a['camera'] == b['camera'], f'{frame_id}: unmasked result changed')
        if frame_id in baseline['required_solved']:
            require(b['report']['status'] == 'solved', f'{frame_id}: previously solved frame lost')
        row = {'id': frame_id, 'mask_applied': frame_id in masks}
        for mode, artifact in [('unmasked', a), ('masked', b)]:
            report = artifact['report']
            quality = report.get('quality') or {}
            row[mode] = {'status': report['status'], 'detections': len(report.get('detections', [])),
                         'inliers': quality.get('n_inliers', 0), 'rms_px': quality.get('rms_px'),
                         'log_odds': quality.get('log_odds'), 'failure': (report.get('failure') or {}).get('code'),
                         'timing_ms': report['timing_ms']}
            if frame_id in masks:
                row[mode]['detections_in_sky'] = sum(in_sky(d, masks[frame_id]) for d in report.get('detections', []))

        rows.append(row)
    return {'schema_version': 1, 'mask_sha256': sha256(masks_path),
            'required_successes_preserved': True, 'unannotated_reports_identical_except_timing_and_equal_flux_order': True,
            'solved': {mode: sum(r[mode]['status'] == 'solved' for r in rows) for mode in ('unmasked', 'masked')},
            'frames': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('../data/input/smartphone/manifest.json'))
    parser.add_argument('--masks', type=Path, default=Path('../data/input/smartphone/sky-masks.json'))
    parser.add_argument('--baseline', type=Path, default=Path('eval/baseline-smartphone.json'))
    parser.add_argument('--binary', type=Path, default=Path('target/release/starglyph'))
    parser.add_argument('--catalog', type=Path, default=Path('../data/catalogs/hyg_v42.csv.gz'))
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--compare-only', action='store_true')
    args = parser.parse_args()
    baseline = read_json(args.baseline)
    check_inputs(args.manifest, baseline)
    if not args.compare_only:
        if any((args.out_dir / mode).exists() for mode in ('unmasked', 'masked')):
            parser.error('use a fresh output directory or --compare-only')
        args.out_dir.mkdir(parents=True, exist_ok=True)
        provenance = {'binary_sha256': sha256(args.binary), 'mask_sha256': sha256(args.masks),
                      'catalog_sha256': sha256(args.catalog), 'platform': platform.platform(),
                      'cpu_count': os.cpu_count(), 'available_cpus': len(os.sched_getaffinity(0)),
                      'rayon_num_threads': os.environ.get('RAYON_NUM_THREADS'),
                      'cache_before': sorted(str(p) for p in Path('artifacts/cache').glob('*.bin')),
                      'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                      'runs': {}}
        for mode in ('unmasked', 'masked'):
            command = [str(args.binary), 'eval', '--manifest', str(args.manifest), '--catalog', str(args.catalog), '--out-dir', str(args.out_dir / mode)]
            if mode == 'masked':
                command += ['--sky-masks', str(args.masks)]
            print(f'Running {mode}', flush=True)
            started = time.monotonic()
            with (args.out_dir / f'{mode}.log').open('w') as log:
                outcome = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'STARGLYPH_SOLVE_DEBUG': '1'})
            provenance['runs'][mode] = {'command': command, 'wall_seconds': time.monotonic()-started, 'exit_code': outcome.returncode}
            (args.out_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
            outcome.check_returncode()
    result = compare(args.manifest, baseline, args.masks, args.out_dir)
    (args.out_dir / 'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['solved']))


if __name__ == '__main__':
    main()
