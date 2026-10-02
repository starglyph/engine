#!/usr/bin/env python3
"""Compare centroid-only masks with sky-only statistics; retain full detector diagnostics."""
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from smartphone_gate import check_inputs, finite_numbers, read_json, require, sha256


def canonical_report(report):
    result = copy.deepcopy(report)
    result.pop('timing_ms', None)
    result['detections'] = sorted(result.get('detections', []), key=lambda d: (-d['flux'], d['x'], d['y']))
    return result


def original_detections(diagnostic, width, height):
    return [{**d, 'x': (d['x']+0.5)*width/diagnostic['width']-0.5,
             'y': (d['y']+0.5)*height/diagnostic['height']-0.5}
            for d in diagnostic['result']['detections']]


def point_in_polygon(x, y, polygon):
    inside = False
    for a, b in zip(polygon, polygon[1:]+polygon[:1]):
        if (a[1] > y) != (b[1] > y) and x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:
            inside = not inside
    return inside


def probe_metrics(artifact, annotation, radius):
    require(artifact['source_sha256'] == annotation['source_sha256'], 'probe image hash mismatch')
    require([artifact['width'], artifact['height']] == [annotation['width'], annotation['height']], 'probe geometry mismatch')
    results = []
    for diagnostic in artifact['detection_diagnostics']:
        detections = original_detections(diagnostic, artifact['width'], artifact['height'])
        edges = sorted(((p['x']-d['x'])**2+(p['y']-d['y'])**2, i, j)
                       for i, p in enumerate(annotation['points']) for j, d in enumerate(detections)
                       if (p['x']-d['x'])**2+(p['y']-d['y'])**2 <= radius**2)
        matches, used = {}, set()
        for distance2, i, j in edges:
            if i not in matches and j not in used:
                matches[i] = (j, distance2**0.5)
                used.add(j)
        probes = []
        for i, p in enumerate(annotation['points']):
            match = matches.get(i)
            d = detections[match[0]] if match else None
            probes.append({'id': p['id'], 'label': p['label'], 'detected_before_top_k': d is not None,
                           'retained': d is not None and d['rank'] < diagnostic['max_detections'],
                           'distance_original_px': match[1] if match else None,
                           'rank': d['rank'] if d else None})
        primary = [p for p in probes if p['label'] == 'visible_compact_source']
        cloud_counts = {c['id']: sum(d['rank'] < diagnostic['max_detections'] and point_in_polygon(d['x'], d['y'], c['polygon'])
                                     for d in detections) for c in annotation.get('cloud_regions', [])}
        results.append({'width': diagnostic['width'], 'height': diagnostic['height'], 'tier': diagnostic['tier'],
                        'stats': diagnostic['result']['stats'], 'n_before_top_k': len(detections),
                        'n_retained': min(len(detections), diagnostic['max_detections']),
                        'primary_total': len(primary), 'primary_before_top_k': sum(p['detected_before_top_k'] for p in primary),
                        'primary_retained': sum(p['retained'] for p in primary),
                        'cloud_region_retained': cloud_counts, 'probes': probes})
    return results


def experiment_modes(sky_fill_pair):
    return ('statistics', 'sky-fill') if sky_fill_pair else ('centroid', 'statistics')


def check_pair_configs(configs, masks_path, sky_fill_pair):
    configs = [dict(c) for c in configs]
    require([c.pop('sky_statistics', False) for c in configs] == ([True, True] if sky_fill_pair else [False, True]), 'statistics mode mismatch')
    require([c.pop('sky_fill', False) for c in configs] == ([False, True] if sky_fill_pair else [False, False]), 'sky-fill mode mismatch')
    require(configs[0] == configs[1] and configs[0]['sky_masks'] == str(masks_path), 'A/B configuration mismatch')


def compare(args):
    baseline = read_json(args.baseline)
    entries = check_inputs(args.manifest, baseline)
    masks = {m['id']: m for m in read_json(args.masks)['masks']}
    probe_file = read_json(args.probes)
    probes = {p['id']: p for p in probe_file['frames']}
    modes = experiment_modes(args.sky_fill_pair)
    summaries = [read_json(args.out_dir/mode/'summary.json') for mode in modes]
    configs = [dict(s['config']) for s in summaries]
    check_pair_configs(configs, args.masks, args.sky_fill_pair)
    for summary in summaries:
        require(summary['dataset']['n_selected'] == len(entries), 'incomplete dataset')
    rows = []
    for entry in entries:
        frame_id = entry['id']
        artifacts = {mode: read_json(args.out_dir/mode/'solve-reports'/f'{frame_id}.json') for mode in modes}
        for mode, a in artifacts.items():
            require(a['source_sha256'] == entry['sha256'] and finite_numbers(a), f'{frame_id}: invalid artifact')
            require([a['width'], a['height']] == baseline['frames'][frame_id]['dimensions'], 'oriented dimensions mismatch')
            require(a.get('sky_mask') == masks.get(frame_id), 'mask provenance mismatch')
            require(a.get('sky_statistics', False) == (mode != 'centroid' and frame_id in masks), 'per-frame statistics mode mismatch')
            require(a.get('sky_fill', False) == (mode == 'sky-fill' and frame_id in masks), 'per-frame sky-fill mode mismatch')
            require(a['report']['status'] in ('solved', 'failed'), 'invalid solve status')
            if frame_id in baseline['required_solved']:
                require(a['report']['status'] == 'solved', f'{frame_id}: prior success lost')
        if frame_id not in masks:
            a, b = artifacts.values()
            require(canonical_report(a['report']) == canonical_report(b['report']) and a['camera'] == b['camera'], f'{frame_id}: unannotated frame changed')
        row = {'id': frame_id, 'mask_applied': frame_id in masks}
        for mode, a in artifacts.items():
            r = a['report']
            row[mode] = {'status': r['status'], 'n_detections': len(r.get('detections', [])),
                         'quality': r.get('quality'), 'failure': r.get('failure'), 'timing_ms': r['timing_ms']}
            if frame_id in probes:
                row[mode]['probe_diagnostics'] = probe_metrics(a, probes[frame_id], probe_file['match_radius_original_px'])
        rows.append(row)
    solved = {mode: sum(row[mode]['status'] == 'solved' for row in rows) for mode in modes}
    for mode, summary in zip(modes, summaries):
        require(summary['solver_track']['solved'] == solved[mode], 'summary status mismatch')
    return {'schema_version': 1, 'mask_sha256': sha256(args.masks), 'probes_sha256': sha256(args.probes),
            'solved': solved, 'required_successes_preserved': True, 'unannotated_reports_unchanged': True, 'frames': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('../data/input/smartphone/manifest.json'))
    parser.add_argument('--masks', type=Path, default=Path('../data/input/smartphone/sky-masks.json'))
    parser.add_argument('--probes', type=Path, default=Path('../data/input/smartphone/point-source-probes.json'))
    parser.add_argument('--baseline', type=Path, default=Path('eval/baseline-smartphone.json'))
    parser.add_argument('--binary', type=Path, default=Path('target/release/starglyph'))
    parser.add_argument('--catalog', type=Path, default=Path('../data/catalogs/hyg_v42.csv.gz'))
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--compare-only', action='store_true')
    parser.add_argument('--sky-fill-pair', action='store_true', help='compare sky statistics with global vs sky-only adaptive occupancy')
    args = parser.parse_args()
    modes = experiment_modes(args.sky_fill_pair)
    check_inputs(args.manifest, read_json(args.baseline))
    if not args.compare_only:
        if any((args.out_dir/mode).exists() for mode in modes):
            parser.error('use a fresh output directory or --compare-only')
        args.out_dir.mkdir(parents=True, exist_ok=True)
        provenance = {'binary_sha256': sha256(args.binary), 'mask_sha256': sha256(args.masks),
                      'probes_sha256': sha256(args.probes), 'catalog_sha256': sha256(args.catalog),
                      'platform': platform.platform(), 'available_cpus': len(os.sched_getaffinity(0)),
                      'rayon_num_threads': os.environ.get('RAYON_NUM_THREADS'),
                      'cache_before': sorted(str(p) for p in Path('artifacts/cache').glob('*.bin')), 'runs': {}}
        for mode in modes:
            command = [str(args.binary), 'eval', '--manifest', str(args.manifest), '--catalog', str(args.catalog),
                       '--out-dir', str(args.out_dir/mode), '--sky-masks', str(args.masks), '--detection-diagnostics']
            if mode != 'centroid':
                command.append('--sky-statistics')
            if mode == 'sky-fill':
                command.append('--sky-fill')
            print(f'Running {mode}', flush=True)
            started = time.monotonic()
            with (args.out_dir/f'{mode}.log').open('w') as log:
                outcome = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'STARGLYPH_SOLVE_DEBUG': '1'})
            provenance['runs'][mode] = {'command': command, 'wall_seconds': time.monotonic()-started, 'exit_code': outcome.returncode}
            (args.out_dir/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
            outcome.check_returncode()
    result = compare(args)
    (args.out_dir/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['solved']))


if __name__ == '__main__':
    main()
