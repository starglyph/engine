#!/usr/bin/env python3
"""Reproducible unmasked/manual/automatic skyline comparison. Run from prototype/."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import PIL

from automatic_sky_mask import algorithm_settings, generate
from automatic_sky_compare import MODES, compare, compare_reference
from smartphone_gate import check_inputs, check_sky_fill_masks, read_json, require, sha256


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare(args):
    """Freeze split, inputs and algorithm before generating masks or seeing solves."""
    paths = {name: getattr(args, name).resolve() for name in
             ('manifest', 'baseline', 'manual_baseline', 'manual_masks', 'probes', 'catalog', 'binary')}
    baseline = read_json(paths['baseline'])
    entries = check_inputs(paths['manifest'], baseline)
    manual_baseline = read_json(paths['manual_baseline'])
    check_inputs(paths['manifest'], manual_baseline)
    masks = check_sky_fill_masks(manual_baseline, paths['manual_masks'])
    require(sha256(paths['catalog']) == baseline['catalog_sha256'] == manual_baseline['catalog_sha256'],
            'catalog hash mismatch')
    ids = {e['id'] for e in entries}
    plan = {name: str(path) for name, path in paths.items()}
    plan.update(schema_version=1, algorithm=args.algorithm, config=asdict(algorithm_settings(args.algorithm)[0]),
                split={'annotated': sorted(masks), 'validation': sorted(ids-set(masks))},
                split_limitations='All 17 frames have been observed during research. Both groups are development/evaluation data, not a holdout.',
                inputs={str(path): sha256(path) for path in paths.values()})
    if args.reference_run is not None:
        reference = args.reference_run.resolve()
        require(reference != args.out_dir, 'reference run must be separate')
        compare(reference)  # Reject stale or incomplete reference before the new run.
        plan['reference_run'] = str(reference)
        for name in ('plan.json', 'provenance.json'):
            plan['inputs'][str(reference/name)] = sha256(reference/name)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.out_dir/'plan.json', plan)
    return entries, plan


def execute(args):
    entries, plan = prepare(args)
    root = Path(__file__).parent
    scripts = ['automatic_sky_mask.py', 'gradient_sky_mask.py', 'automatic_sky_compare.py', 'automatic_sky_experiment.py',
               'sky_statistics_experiment.py', 'sky_mask_experiment.py', 'smartphone_gate.py']
    provenance = {'plan_sha256': sha256(args.out_dir/'plan.json'),
                  'scripts_sha256': {name: sha256(root/name) for name in scripts},
                  'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  'working_tree_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'])).hexdigest(),
                  'python': platform.python_version(), 'numpy': np.__version__, 'pillow': PIL.__version__,
                  'platform': platform.platform(), 'cpu_count': os.cpu_count(),
                  'available_cpus': len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count(),
                  'rayon_num_threads': os.environ.get('RAYON_NUM_THREADS'),
                  'cache_before': sorted(str(p) for p in Path('artifacts/cache').glob('*.bin')),
                  'runs': {}, 'artifacts': {}}
    write_json(args.out_dir/'provenance.json', provenance)
    masks, generation = generate(Path(plan['manifest']), entries, algorithm=plan['algorithm'])
    write_json(args.out_dir/'automatic-masks.json', masks)
    write_json(args.out_dir/'generation.json', generation)
    # Write hashes before solving, not after examining candidate results.
    for name in ('automatic-masks.json', 'generation.json'):
        provenance['artifacts'][name] = sha256(args.out_dir/name)
    write_json(args.out_dir/'provenance.json', provenance)
    for mode in MODES:
        command = [plan['binary'], 'eval', '--manifest', plan['manifest'], '--catalog', plan['catalog'],
                   '--out-dir', str(args.out_dir/mode), '--detection-diagnostics']
        mask_path = plan['manual_masks'] if mode == 'manual' else str(args.out_dir/'automatic-masks.json')
        if mode == 'manual' or (mode == 'automatic' and masks['masks']):
            command += ['--sky-masks', mask_path, '--sky-statistics', '--sky-fill']
        print(f'Running {mode}', flush=True)
        started = time.monotonic()
        with (args.out_dir/f'{mode}.log').open('w') as log:
            outcome = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                     env={**os.environ, 'STARGLYPH_SOLVE_DEBUG': '1'})
        provenance['runs'][mode] = {'command': command, 'wall_seconds': time.monotonic()-started,
                                    'exit_code': outcome.returncode}
        for path in (args.out_dir/mode).rglob('*.json'):
            provenance['artifacts'][str(path.relative_to(args.out_dir))] = sha256(path)
        write_json(args.out_dir/'provenance.json', provenance)
        outcome.check_returncode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('../data/input/smartphone/manifest.json'))
    parser.add_argument('--baseline', type=Path, default=Path('eval/baseline-smartphone.json'))
    parser.add_argument('--manual-baseline', type=Path, default=Path('eval/baseline-smartphone-sky-fill.json'))
    parser.add_argument('--manual-masks', type=Path, default=Path('../data/input/smartphone/sky-masks.json'))
    parser.add_argument('--probes', type=Path, default=Path('../data/input/smartphone/point-source-probes.json'))
    parser.add_argument('--catalog', type=Path, default=Path('../data/catalogs/hyg_v42.csv.gz'))
    parser.add_argument('--binary', type=Path, default=Path('target/release/starglyph'))
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--algorithm', choices=('rgb_skyline_v1', 'rgb_gradient_v2'), default='rgb_skyline_v1')
    parser.add_argument('--reference-run', type=Path, help='pin a previous automatic experiment for per-ID comparison')
    parser.add_argument('--compare-only', action='store_true', help='verify existing artifacts; never rerun or regenerate masks')
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    if not args.compare_only:
        execute(args)
    result = compare(args.out_dir)
    plan = read_json(args.out_dir/'plan.json')
    if plan.get('reference_run'):
        result['reference_comparison'] = compare_reference(args.out_dir, Path(plan['reference_run']), result)
    write_json(args.out_dir/'comparison.json', result)
    print(json.dumps({key: result[key] for key in ('solved', 'automatic_vs_unmasked', 'automatic_vs_manual', 'splits')}))
    # An honest negative experiment is valid output, not an excuse to retune.
    # Regressions are explicit in comparison.json; this command is NOT a release gate.


if __name__ == '__main__':
    main()
