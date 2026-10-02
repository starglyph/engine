#!/usr/bin/env python3
"""Offline reduced/deep prefix experiment; this is not a full-frame solve gate."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def cases(detections, width, height, seed):
    """One measured list and one reproducible null geometry with identical fluxes."""
    rng = random.Random(seed)
    negative = copy.deepcopy(detections)
    for d in negative:
        d.update(x=rng.uniform(8, width-8), y=rng.uniform(8, height-8))
    return [dict(name='automatic', detections=detections),
            dict(name='uniform_negative', detections=negative)]


def reduced_deep(report):
    matches = [d for d in report['detection_diagnostics']
               if max(d['width'], d['height']) == 1600 and d['tier'] == 'deep']
    if len(matches) != 1:
        raise ValueError('expected one reduced deep diagnostic at long edge 1600')
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--test-binary', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    binary = args.test_binary.resolve()
    run = args.run.resolve()
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    catalog = root / 'data/catalogs/hyg_v42.csv.gz'
    manifest_path = root / 'data/input/smartphone/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    inputs = []
    sources = [binary, catalog, manifest_path, Path(__file__).resolve(),
               root / 'prototype/crates/starglyph-core/src/solve.rs',
               root / 'prototype/crates/starglyph-core/src/solve/replay.rs',
               root / 'prototype/Cargo.lock']
    for index, entry in enumerate(manifest):
        frame_id = entry['id']
        report_path = run / 'automatic/solve-reports' / f'{frame_id}.json'
        report = json.loads(report_path.read_text())
        image = manifest_path.parent / entry['file']
        if sha256(image) != report['source_sha256'] or report['source_sha256'] != entry['sha256']:
            raise ValueError(f'{frame_id}: source identity mismatch')
        diagnostic = reduced_deep(report)
        width, height = diagnostic['width'], diagnostic['height']
        detections = diagnostic['result']['detections'][:diagnostic['max_detections']]
        seed = 20261002 + index
        data = dict(image=str(image), catalog=str(catalog), cache=str(root / 'prototype/artifacts/cache'),
                    width=width, height=height, extra_prefixes=[40, 50], extra_dense=True,
                    cases=cases(detections, width, height, seed))
        path = output / f'{frame_id}.input.json'
        write_json(path, data)
        sources.extend([image, report_path, path])
        inputs.append(dict(id=frame_id, input=str(path), seed=seed, count=len(detections)))
    plan = dict(scope='Reduced 1600px deep only. Candidates before refinement, not full-image solve counts.',
                policy='Current ladder plus independently measured 40/50 prefixes; bootstrap and dense. Full detections used for verification.',
                negative_limit='17 uniform random geometries are a small synthetic smoke check, not a false-positive-rate estimate.',
                inputs=inputs, sha256={str(p): sha256(p) for p in sources})
    write_json(output / 'plan.json', plan)  # Freeze every input before running any case.
    rows = []
    for item in inputs:
        frame_id = item['id']
        result_path = output / f'{frame_id}.result.json'
        command = [str(binary), 'solve::replay::saved_detections', '--ignored', '--nocapture']
        started = time.monotonic()
        with (output / f'{frame_id}.log').open('w') as log:
            subprocess.run(command, cwd=root / 'prototype', check=True, stdout=log, stderr=subprocess.STDOUT,
                           env={**os.environ, 'STARGLYPH_REPLAY_INPUT': item['input'],
                                'STARGLYPH_REPLAY_OUTPUT': str(result_path)})
        result = json.loads(result_path.read_text())
        for case in result['cases']:
            accepted = [probe for probe in case['extra_prefixes']
                        if (probe['result'].get('candidate') or {}).get('accepted')]
            rows.append(dict(id=frame_id, case=case['name'], baseline=case['chosen_before_refinement'],
                             accepted_extra=accepted, baseline_ms=case['matching_ms'],
                             all_extra_ms=sum(p['elapsed_ms'] for p in case['extra_prefixes'])))
        print(frame_id, [(c['name'], c['chosen_before_refinement'] is not None) for c in result['cases']],
              f'{time.monotonic()-started:.2f}s', flush=True)
    artifacts = {p.name: sha256(p) for p in output.glob('*') if p.is_file()}
    write_json(output / 'comparison.json', dict(rows=rows, artifacts=artifacts, plan=plan))


if __name__ == '__main__':
    main()
