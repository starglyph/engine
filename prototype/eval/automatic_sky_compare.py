"""Integrity checks and metrics for the three-arm automatic skyline experiment."""
from pathlib import Path
import math

import numpy as np

from smartphone_gate import check_inputs, check_reports, finite_numbers, read_json, require, sha256
from sky_mask_experiment import in_sky
from sky_statistics_experiment import canonical_report, original_detections, probe_metrics

MODES = ('unmasked', 'manual', 'automatic')


def rasterize(polygon, width, height):
    """Evaluate normalized pixel centers; exclude polygon boundaries like core."""
    x, y = np.meshgrid((np.arange(width)+0.5)/width, (np.arange(height)+0.5)/height)
    inside = np.zeros((height, width), dtype=bool)
    boundary = inside.copy()
    for a, b in zip(polygon, polygon[1:]+polygon[:1]):
        cross = (x-a[0])*(b[1]-a[1])-(y-a[1])*(b[0]-a[0])
        boundary |= ((np.abs(cross) <= 1e-12) & (x >= min(a[0], b[0])) &
                     (x <= max(a[0], b[0])) & (y >= min(a[1], b[1])) & (y <= max(a[1], b[1])))
        if a[1] != b[1]:
            inside ^= ((a[1] > y) != (b[1] > y)) & (x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])
    return inside & ~boundary


def mask_overlap(manual, automatic):
    scale = 320/max(manual['width'], manual['height'])
    width, height = max(1, round(manual['width']*scale)), max(1, round(manual['height']*scale))
    a = rasterize(manual['sky_polygon'], width, height)
    b = (rasterize(automatic['sky_polygon'], width, height) if automatic else np.ones_like(a))
    intersection, union = int((a & b).sum()), int((a | b).sum())
    return {'grid_dimensions': [width, height], 'iou_with_manual': intersection/union if union else None,
            'manual_region_retained': intersection/int(a.sum()) if a.any() else None,
            'selected_outside_manual_fraction': int((b & ~a).sum())/int(b.sum()) if b.any() else None}


def transitions(reference, candidate):
    return {'gained': sorted(set(candidate)-set(reference)), 'lost': sorted(set(reference)-set(candidate))}


def reference_inlier_recovery(reference, candidate, radius=12):
    """Track detections used by a reference solve, NOT independent star truth."""
    require((reference['source_sha256'], reference['width'], reference['height']) ==
            (candidate['source_sha256'], candidate['width'], candidate['height']), 'recovery image mismatch')
    points = [d for d in reference['report']['detections'] if d.get('inlier')]
    rows = []
    for diagnostic in candidate['detection_diagnostics']:
        detections = original_detections(diagnostic, candidate['width'], candidate['height'])
        edges = sorted(((p['x']-d['x'])**2+(p['y']-d['y'])**2, i, j)
                       for i, p in enumerate(points) for j, d in enumerate(detections)
                       if (p['x']-d['x'])**2+(p['y']-d['y'])**2 <= radius**2)
        matches, used = {}, set()
        for _, i, j in edges:
            if i not in matches and j not in used:
                matches[i] = detections[j]['rank']
                used.add(j)
        rows.append({'width': diagnostic['width'], 'tier': diagnostic['tier'],
                     'reference_inliers': len(points), 'matched_before_top_k': len(matches),
                     'retained': sum(rank < diagnostic['max_detections'] for rank in matches.values()),
                     'ranks_by_reference_inlier': [matches.get(i) for i in range(len(points))]})
    return {'scope': 'reference_solve_detections_not_independent_truth', 'radius_original_px': radius, 'tiers': rows}


def same_annotation(actual, expected):
    """Keep identity exact, allowing only two ULPs from Python/Rust JSON floats.

    Input-file hashes remain exact. serde_json without float_roundtrip can move
    generated fractional coordinates by an ULP on parse/serialize.
    """
    if actual is None or expected is None:
        return actual is expected
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    a, b = dict(actual), dict(expected)
    ap, bp = a.pop('sky_polygon', None), b.pop('sky_polygon', None)
    if a != b or not isinstance(ap, list) or not isinstance(bp, list) or len(ap) != len(bp):
        return False
    for p, q in zip(ap, bp):
        if not isinstance(p, list) or not isinstance(q, list) or len(p) != 2 or len(q) != 2:
            return False
        for x, y in zip(p, q):
            if type(x) not in (int, float) or type(y) not in (int, float) or not math.isfinite(x+y):
                return False
            if abs(x-y) > 2*max(math.ulp(x), math.ulp(y)):
                return False
    return True


def validate_artifact(artifact, record, entry, pinned, mask):
    frame_id = entry['id']
    require(artifact['id'] == record['id'] == frame_id, f'{frame_id}: report ID mismatch')
    require(artifact['source_sha256'] == entry['sha256'], f'{frame_id}: source mismatch')
    require([artifact['width'], artifact['height']] == pinned['dimensions'], f'{frame_id}: dimensions mismatch')
    require(artifact['pixel_convention'] == 'top_left_zero_based', f'{frame_id}: pixel convention mismatch')
    require(same_annotation(artifact.get('sky_mask'), mask), f'{frame_id}: mask provenance mismatch')
    require(artifact.get('sky_statistics', False) == (mask is not None) and
            artifact.get('sky_fill', False) == (mask is not None), f'{frame_id}: flags mismatch')
    require(finite_numbers(artifact), f'{frame_id}: non-finite artifact')
    report = artifact['report']
    require(report['status'] in ('solved', 'failed') and record['status'] == report['status'],
            f'{frame_id}: status mismatch')
    if report['status'] == 'solved':
        for section, keys in {'pose': ('ra_deg', 'dec_deg', 'roll_deg'),
                              'fov': ('fov_x_deg', 'fov_y_deg', 'focal_px'),
                              'quality': ('n_inliers', 'rms_px', 'log_odds', 'confidence')}.items():
            require(isinstance(report.get(section), dict) and
                    all(type(report[section].get(k)) in (int, float) for k in keys),
                    f'{frame_id}: missing {section}')
        camera = artifact.get('camera')
        require(isinstance(camera, dict) and [camera['width'], camera['height']] == pinned['dimensions'] and
                type(camera.get('focal_px')) in (int, float) and camera['focal_px'] > 0,
                f'{frame_id}: invalid camera')
        rotation = camera.get('world_to_camera')
        require(type(camera.get('k1')) in (int, float) and isinstance(rotation, list) and len(rotation) == 3 and
                all(isinstance(row, list) and len(row) == 3 and all(type(v) in (int, float) for v in row)
                    for row in rotation), f'{frame_id}: invalid rotation')


def compare(output):
    plan = read_json(output/'plan.json')
    provenance = read_json(output/'provenance.json')
    for path, digest in plan['inputs'].items():
        require(sha256(Path(path)) == digest, f'input changed: {path}')
    for path, digest in provenance['artifacts'].items():
        require(sha256(output/path) == digest, f'artifact changed: {path}')
    require(provenance['plan_sha256'] == sha256(output/'plan.json'), 'plan changed')
    baseline = read_json(Path(plan['baseline']))
    entries = check_inputs(Path(plan['manifest']), baseline)
    expected_artifacts = {'automatic-masks.json', 'generation.json'}
    for mode in MODES:
        require(provenance['runs'][mode]['exit_code'] == 0, f'{mode}: failed evaluation')
        expected_artifacts.add(f'{mode}/summary.json')
        for entry in entries:
            for kind in ('per-frame', 'solve-reports'):
                expected_artifacts.add(f"{mode}/{kind}/{entry['id']}.json")
    require(expected_artifacts <= set(provenance['artifacts']), 'incomplete artifact inventory')
    manual_path = Path(plan['manual_masks'])
    masks = {'unmasked': {}, 'manual': {m['id']: m for m in read_json(manual_path)['masks']},
             'automatic': {m['id']: m for m in read_json(output/'automatic-masks.json')['masks']}}
    generation_file = read_json(output/'generation.json')
    if 'algorithm' in plan:
        require(generation_file.get('algorithm') == plan['algorithm'] and
                generation_file.get('config') == plan['config'], 'generation algorithm/config mismatch')
    generation = generation_file['frames']
    ids = {entry['id'] for entry in entries}
    require(set(generation) == ids and set(masks['automatic']) <= ids, 'automatic membership mismatch')
    require(set(plan['split']['annotated']) == set(masks['manual']) and
            set(plan['split']['validation']) == ids-set(masks['manual']), 'split mismatch')
    probes = read_json(Path(plan['probes']))
    probe_frames = {p['id']: p for p in probes['frames']}
    check_reports(Path(plan['manifest']), baseline, output/'unmasked')
    check_reports(Path(plan['manifest']), read_json(Path(plan['manual_baseline'])), output/'manual',
                  sky_fill_masks=manual_path)
    summaries = {mode: read_json(output/mode/'summary.json') for mode in MODES}
    configs = []
    for mode, summary in summaries.items():
        require(summary['dataset']['n_selected'] == len(entries) and summary['dataset']['tracks'] == ['solver'],
                f'{mode}: incomplete selection')
        config = dict(summary['config'])
        expected = None if mode == 'unmasked' else str(manual_path if mode == 'manual' else output/'automatic-masks.json')
        if mode == 'automatic' and not masks['automatic']:
            expected = None
        require(config.pop('sky_masks', None) == expected, f'{mode}: mask config mismatch')
        require(config.pop('sky_statistics', False) == (expected is not None) and
                config.pop('sky_fill', False) == (expected is not None), f'{mode}: experiment flags mismatch')
        configs.append(config)
    require(all(c == configs[0] for c in configs), 'solve configurations differ')
    rows, solved = [], {mode: [] for mode in MODES}
    for entry in entries:
        frame_id = entry['id']
        artifacts = {mode: read_json(output/mode/'solve-reports'/f'{frame_id}.json') for mode in MODES}
        row = {'id': frame_id, 'split': 'annotated' if frame_id in plan['split']['annotated'] else 'validation',
               'mask_generation': generation[frame_id]}
        if artifacts['manual']['report']['status'] == 'solved' and artifacts['automatic']['report']['status'] == 'failed':
            row['manual_inlier_recovery'] = {mode: reference_inlier_recovery(artifacts['manual'], artifacts[mode])
                for mode in ('manual', 'automatic') if 'detection_diagnostics' in artifacts[mode]}
        if frame_id in masks['manual']:
            row['mask_overlap'] = mask_overlap(masks['manual'][frame_id], masks['automatic'].get(frame_id))
        for mode, artifact in artifacts.items():
            validate_artifact(artifact, read_json(output/mode/'per-frame'/f'{frame_id}.json'),
                              entry, baseline['frames'][frame_id], masks[mode].get(frame_id))
            report = artifact['report']
            if report['status'] == 'solved':
                solved[mode].append(frame_id)
            row[mode] = {'status': report['status'], 'n_detections': len(report.get('detections', [])),
                         'quality': report.get('quality'), 'failure': report.get('failure'),
                         'timing_ms': report['timing_ms']}
            if frame_id in masks['manual']:
                row[mode]['detections_outside_manual'] = sum(not in_sky(d, masks['manual'][frame_id])
                                                           for d in report.get('detections', []))
            if frame_id in probe_frames:
                row[mode]['probe_diagnostics'] = probe_metrics(artifact, probe_frames[frame_id], probes['match_radius_original_px'])
            if mode != 'unmasked' and frame_id not in masks[mode]:
                original = artifacts['unmasked']
                require(canonical_report(report) == canonical_report(original['report']) and
                        artifact['camera'] == original['camera'], f'{frame_id}: unmasked fallback changed')
        rows.append(row)
    for mode in MODES:
        require(summaries[mode]['solver_track']['solved'] == len(solved[mode]), f'{mode}: solved count mismatch')
    return {'schema_version': 1, 'scope': 'solver_acceptance_not_astrometric_truth',
            'algorithm': plan.get('algorithm', 'rgb_skyline_v1'),
            'analysis_scripts_sha256': {name: sha256(Path(__file__).with_name(name)) for name in
                ('automatic_sky_compare.py', 'sky_statistics_experiment.py', 'sky_mask_experiment.py', 'smartphone_gate.py')},
            'solved': {mode: len(ids) for mode, ids in solved.items()}, 'solved_ids': solved,
            'automatic_vs_unmasked': transitions(solved['unmasked'], solved['automatic']),
            'automatic_vs_manual': transitions(solved['manual'], solved['automatic']),
            'splits': {split: {mode: sum(r['split'] == split and r[mode]['status'] == 'solved' for r in rows)
                              for mode in MODES} for split in plan['split']}, 'frames': rows}


def compare_reference(output, reference, result):
    """Verify same engine/data and unchanged controls before comparing algorithms."""
    current_plan, previous_plan = [read_json(root/'plan.json') for root in (output, reference)]
    for key in ('binary', 'manifest', 'catalog', 'baseline', 'manual_baseline', 'manual_masks', 'probes'):
        require(current_plan['inputs'][current_plan[key]] == previous_plan['inputs'][previous_plan[key]],
                f'reference {key} differs')
    previous = compare(reference)
    require(set(result['solved_ids']) == set(previous['solved_ids']), 'reference modes differ')
    for frame in result['frames']:
        for mode in ('unmasked', 'manual'):
            a, b = [read_json(root/mode/'solve-reports'/f"{frame['id']}.json") for root in (output, reference)]
            require(canonical_report(a['report']) == canonical_report(b['report']) and a['camera'] == b['camera'],
                    f"{frame['id']}: reference {mode} control changed")
    return {'algorithm': previous['algorithm'], 'solved': previous['solved']['automatic'],
            'controls_identical_except_timing_and_equal_flux_order': True,
            **transitions(previous['solved_ids']['automatic'], result['solved_ids']['automatic'])}
