#!/usr/bin/env python3
"""Run/verify the pinned smartphone set without accepting aggregate-only success."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_numbers(value):
    if isinstance(value, dict):
        return all(finite_numbers(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_numbers(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def check_inputs(manifest_path, baseline):
    require(baseline['schema_version'] == 1, 'unsupported baseline version')
    entries = read_json(manifest_path)
    ids = [entry['id'] for entry in entries]
    expected = baseline['frames']
    require(len(ids) == len(set(ids)) and set(ids) == set(expected), 'dataset membership mismatch')
    require(set(baseline['required_solved']) <= set(ids), 'unknown required solved ID')
    require(bool(baseline['required_solved']), 'empty required solved set')
    for entry in entries:
        frame_id = entry['id']
        pinned = expected[frame_id]
        require(entry['track'] == 'solver', f'{frame_id}: track mismatch')
        require(entry['sha256'] == pinned['sha256'], f'{frame_id}: manifest SHA-256 mismatch')
        require(sha256(manifest_path.parent / entry['file']) == pinned['sha256'],
                f'{frame_id}: input SHA-256 mismatch')
        dimensions = [entry['width'], entry['height']]
        if entry['orientation'] in (5, 6, 7, 8):
            dimensions.reverse()
        require(dimensions == pinned['dimensions'], f'{frame_id}: manifest dimensions mismatch')
    return entries


def check_sky_fill_masks(baseline, masks_path):
    require(baseline.get('mode') == 'sky_fill', 'sky-fill gate requires its experimental baseline')
    require(sha256(masks_path) == baseline['masks_sha256'], 'mask file SHA-256 mismatch')
    document = read_json(masks_path)
    require(document['schema_version'] == 1 and
            document['coordinates'] == 'exif_oriented_normalized_image_edges', 'unsupported mask format')
    masks = {mask['id']: mask for mask in document['masks']}
    require(len(masks) == len(document['masks']) and set(masks) == set(baseline['mask_ids']),
            'mask membership mismatch')
    for frame_id, mask in masks.items():
        pinned = baseline['frames'][frame_id]
        require(mask['source_sha256'] == pinned['sha256'], f'{frame_id}: mask source SHA-256 mismatch')
        require([mask['width'], mask['height']] == pinned['dimensions'], f'{frame_id}: mask dimensions mismatch')
        require(finite_numbers(mask), f'{frame_id}: non-finite mask')
    return masks


def check_reports(manifest_path, baseline, run_dir, *, sky_fill_masks=None):
    entries = check_inputs(manifest_path, baseline)
    summary = read_json(run_dir / 'summary.json')
    require(not summary['config'].get('quantile_threshold') and not summary['config'].get('blob_concentration'),
            'experimental detector configuration cannot satisfy an existing gate')
    masks = None
    if sky_fill_masks is not None:
        masks = check_sky_fill_masks(baseline, sky_fill_masks)
        require(summary['config'].get('sky_fill') is True and summary['config'].get('sky_statistics') is True,
                'sky-fill gate requires both experimental flags')
        require(isinstance(summary['config'].get('sky_masks'), str), 'sky-fill mask configuration missing')
        require(sha256(Path(summary['config']['sky_masks'])) == baseline['masks_sha256'],
                'reported mask file SHA-256 mismatch')
    else:
        require(baseline.get('mode', 'default') == 'default', 'experimental baseline cannot satisfy the default gate')
        require(not summary['config'].get('sky_fill', False), 'experimental sky fill cannot satisfy the default baseline')
        require(not summary['config'].get('sky_statistics', False), 'experimental sky statistics cannot satisfy the default baseline')
        require(summary['config'].get('sky_masks') is None, 'experimental sky masks cannot satisfy the default baseline')
    require(summary['dataset']['n_selected'] == len(entries), 'selected count mismatch')
    require(summary['dataset']['tracks'] == ['solver'], 'selected tracks mismatch')
    for key, value in baseline['config'].items():
        require(summary['config'][key] == value, f'configuration mismatch: {key}')
    require(sha256(Path(summary['config']['catalog'])) == baseline['catalog_sha256'],
            'catalog SHA-256 mismatch')
    solved = []
    for entry in entries:
        frame_id = entry['id']
        pinned = baseline['frames'][frame_id]
        record = read_json(run_dir / 'per-frame' / f'{frame_id}.json')
        artifact = read_json(run_dir / 'solve-reports' / f'{frame_id}.json')
        require(not artifact.get('quantile_threshold') and not artifact.get('blob_concentration'),
                f'{frame_id}: experimental detector configuration in existing baseline')
        if masks is not None:
            applied = frame_id in masks
            require(artifact.get('sky_mask') == masks.get(frame_id), f'{frame_id}: mask provenance mismatch')
            require(artifact.get('sky_fill') is applied and artifact.get('sky_statistics') is applied,
                    f'{frame_id}: per-frame experimental flags mismatch')
        else:
            require(not artifact.get('sky_fill', False), f'{frame_id}: experimental sky fill in default baseline')
            require(not artifact.get('sky_statistics', False), f'{frame_id}: experimental sky statistics in default baseline')
            require(artifact.get('sky_mask') is None, f'{frame_id}: experimental mask in default baseline')
        require(record['id'] == artifact['id'] == frame_id, f'{frame_id}: report ID mismatch')
        require(artifact['source_sha256'] == pinned['sha256'], f'{frame_id}: report SHA-256 mismatch')
        require([artifact['width'], artifact['height']] == pinned['dimensions'],
                f'{frame_id}: oriented dimensions mismatch')
        require(artifact['pixel_convention'] == 'top_left_zero_based', f'{frame_id}: pixel convention')
        status = record['status']
        require(status in ('solved', 'failed'), f'{frame_id}: input failed to load')
        require(artifact['report']['status'] == status, f'{frame_id}: status mismatch')
        require(finite_numbers(artifact), f'{frame_id}: non-finite report')
        if status == 'solved':
            for section, keys in {'pose': ('ra_deg', 'dec_deg', 'roll_deg'),
                                  'fov': ('fov_x_deg', 'fov_y_deg', 'focal_px'),
                                  'quality': ('n_detections', 'n_inliers', 'rms_px', 'log_odds', 'confidence')}.items():
                values = artifact['report'].get(section)
                require(isinstance(values, dict) and all(isinstance(values.get(key), (int, float)) for key in keys),
                        f'{frame_id}: missing numeric {section}')
            camera = artifact['camera']
            require(isinstance(camera, dict), f'{frame_id}: missing camera')
            require([camera['width'], camera['height']] == pinned['dimensions'], f'{frame_id}: camera dimensions')
            require(isinstance(camera['focal_px'], (float, int)) and camera['focal_px'] > 0,
                    f'{frame_id}: invalid focal length')
            require(isinstance(camera['k1'], (float, int)), f'{frame_id}: invalid distortion')
            rotation = camera['world_to_camera']
            require(isinstance(rotation, list) and len(rotation) == 3 and
                    all(isinstance(row, list) and len(row) == 3 and
                        all(isinstance(v, (float, int)) for v in row) for row in rotation),
                    f'{frame_id}: invalid rotation')
            solved.append(frame_id)
    missing = set(baseline['required_solved']) - set(solved)
    require(not missing, f'previously solved frames lost: {sorted(missing)}')
    require(summary['solver_track']['solved'] == len(solved), 'solved count mismatch')
    return solved


def main(*, sky_fill=False):
    label = 'SMARTPHONE SKY-FILL GATE' if sky_fill else 'SMARTPHONE GATE'
    parser = argparse.ArgumentParser(description=('Experimental manual-mask acceptance regression; not astrometric ground truth.' if sky_fill else __doc__))
    parser.add_argument('--manifest', type=Path, default=Path('../data/input/smartphone/manifest.json'))
    parser.add_argument('--baseline', type=Path, default=Path('eval/baseline-smartphone-sky-fill.json' if sky_fill else 'eval/baseline-smartphone.json'))
    parser.add_argument('--run-dir', type=Path, help='check an existing run instead of executing')
    parser.add_argument('--binary', type=Path, default=Path('target/release/starglyph'))
    parser.add_argument('--out-dir', type=Path, default=Path('artifacts/eval/smartphone-sky-fill' if sky_fill else 'artifacts/eval/smartphone'))
    if sky_fill:
        parser.add_argument('--masks', type=Path, default=Path('../data/input/smartphone/sky-masks.json'))
    args = parser.parse_args()
    try:
        baseline = read_json(args.baseline)
        check_inputs(args.manifest, baseline)
        if sky_fill:
            check_sky_fill_masks(baseline, args.masks)
        else:
            require(baseline.get('mode', 'default') == 'default', 'experimental baseline cannot satisfy the default gate')
        run_dir = args.run_dir
        if run_dir is None:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            run_dir = Path(tempfile.mkdtemp(prefix='run-', dir=args.out_dir)).resolve()
            command = [str(args.binary.resolve()), 'eval', '--manifest', str(args.manifest.resolve()),
                       '--catalog', '../data/catalogs/hyg_v42.csv.gz', '--out-dir', str(run_dir)]
            if sky_fill:
                command += ['--sky-masks', str(args.masks.resolve()), '--sky-statistics', '--sky-fill']
            provenance = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                          'working_tree_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'])).hexdigest(),
                          'binary_sha256': sha256(args.binary),
                          'gate_sha256': sha256(Path(__file__)),
                          'available_cpus': len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count(),
                          'rayon_num_threads': os.environ.get('RAYON_NUM_THREADS'),
                          'platform': platform.platform(), 'cpu_count': os.cpu_count(),
                          'cpu_model': next((line.split(':', 1)[1].strip() for line in (Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else '').splitlines() if line.startswith('model name')), platform.processor()),
                          'python': platform.python_version(),
                          'rustc': subprocess.check_output(['rustc', '--version'], text=True).strip(),
                          'cache_before': sorted(str(p) for p in Path('artifacts/cache').glob('*.bin')),
                          'command': command,
                          'baseline_sha256': sha256(args.baseline), 'started_unix': time.time()}
            if sky_fill:
                provenance.update(mode='sky_fill', masks_sha256=sha256(args.masks),
                                  scope='solver_acceptance_not_astrometric_truth')
            (run_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
            print(f'Artifacts: {run_dir}', flush=True)
            started = time.monotonic()
            with (run_dir / 'solve.log').open('w') as log:
                outcome = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                         env={**os.environ, 'STARGLYPH_SOLVE_DEBUG': '1'})
            provenance.update(wall_seconds=time.monotonic() - started, exit_code=outcome.returncode)
            (run_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
            outcome.check_returncode()
            provenance['generated_by'] = read_json(run_dir / 'summary.json')['generated_by']
            (run_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        solved = check_reports(args.manifest, baseline, run_dir,
                               sky_fill_masks=args.masks if sky_fill else None)
        print(f'{label} PASS: {len(solved)}/{len(baseline["frames"])}; {run_dir}')
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f'{label} FAIL: {error}') from error


if __name__ == '__main__':
    main()
