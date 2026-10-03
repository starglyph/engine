#!/usr/bin/env python3
"""Paired robustness iteration: frozen detector variant, explicit split, compact results.

Run from prototype/. Holdout requires the protocol written before development.
No parameters are fitted; no WCS solve is launched. Raw reports stay in artifacts/.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from collection_run import ROOT, digest, select_records, write_json

SOURCES = [
    'prototype/crates/starglyph-core/src/detect.rs',
    'prototype/crates/starglyph-core/src/solve.rs',
    'prototype/crates/starglyph-cli/src/main.rs',
    'prototype/crates/starglyph-cli/src/eval_cmd.rs',
    'prototype/eval/collection_run.py', 'prototype/eval/robustness_iteration.py',
    'prototype/Cargo.lock',
]
MANIFEST = ROOT / 'data/samples/sky-samples/manifest.json'
BINARY = ROOT / 'prototype/target/release/starglyph'
CATALOG = ROOT / 'data/catalogs/hyg_v42.csv.gz'
BASELINE = ROOT / 'data/samples/sky-samples/robustness-results.json'


def frozen_hashes():
    return {name: digest(ROOT / name) for name in SOURCES + [
        str(p.relative_to(ROOT)) for p in (MANIFEST, BINARY, CATALOG, BASELINE)]}


def new_protocol():
    return {
        'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'hashes': frozen_hashes(),
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'experiment': 'occupancy_quantile_v1',
        'parameters': {'max_fill': .02, 'max_threshold_multiplier': 16,
                       'initial_threshold': 'unchanged k_sigma * sigma_convolved',
                       'rank': 'n - floor(n/50) - 1; strict >; clamp to [initial, 16*initial]'},
        'controls': 'Same release binary, pixels, catalog, cache, search ladder, acceptance, refinement; no masks or hints.',
        'criteria': [
            'Compare every ID, separating solver/stress/scene and negative examples.',
            'Retain every baseline success; no new accepted negative, process error or wall timeout.',
            'New solve counts are provisional without independently checked stellar geometry.',
            'Report detector counts, inliers, internal RMS, pending external metrics and timings separately.',
            'No default activation if any old success is lost or independent geometry remains unconfirmed.',
            'Same 120 s per-process ceiling; no added solver attempts. Timings are descriptive single pairs.',
            'One fixed variant; do not tune on holdout or rerun holdout to select a variant.',
        ],
        'environment': {'python': sys.version, 'platform': platform.platform(),
                        'available_cpus': len(os.sched_getaffinity(0)),
                        'rayon_num_threads': os.environ.get('RAYON_NUM_THREADS'),
                        'cache': {p.name: digest(p) for p in sorted((ROOT/'prototype/artifacts/cache').glob('*.bin'))}},
    }


def summarize(out, protocol):
    records = {r['id']: r for r in json.loads(MANIFEST.read_text())}
    old = {r['id']: r for r in json.loads(BASELINE.read_text())['frames']}
    pairs = {}
    inputs = None
    for mode in ('control', 'quantile'):
        folder = out / mode
        plan = json.loads((folder/'plan.json').read_text())
        summary = json.loads((folder/'summary.json').read_text())
        if summary['plan_sha256'] != digest(folder/'plan.json'):
            raise ValueError('changed plan')
        selected = plan['inputs']
        if inputs is not None and selected != inputs:
            raise ValueError('paired input mismatch')
        inputs = selected
        if (summary['completed'] != len(inputs) or summary['input_count'] != len(inputs)
                or [r['id'] for r in summary['frames']] != [r['id'] for r in inputs]):
            raise ValueError('incomplete or duplicated run')
        if bool(plan.get('quantile_threshold')) != (mode == 'quantile'):
            raise ValueError('wrong experimental arm')
        for row in summary['frames']:
            rid = row['id']
            rec = records[rid]
            pair = pairs.setdefault(rid, {'id': rid, 'split': row['split'], 'track': rec['track'],
                'negative': rec['research'].get('negative', False), 'source_sha256': rec['clean_sha256'],
                'published_status': old[rid]['starglyph'].get('solve_status'),
                'external_reference_status': old[rid]['external_wcs']['status'],
                'ground_truth': 'unconfirmed'})
            result = {k: row[k] for k in ('status', 'elapsed_s', 'solve_status', 'exit_code') if k in row}
            if 'report' in row:
                path = Path(row['report'])
                if digest(path) != row['report_sha256']:
                    raise ValueError('changed report')
                artifact = json.loads(path.read_text())
                if artifact['source_sha256'] != rec['clean_sha256']:
                    raise ValueError('report input mismatch')
                report = artifact['report']
                result.update(report_sha256=digest(path), detections=len(report['detections']),
                              quality=report.get('quality'), failure=report.get('failure'),
                              timing_ms=report['timing_ms'])
                # Full local TAN/SIP comparison, never promoted to ground truth.
                refpath = ROOT/'prototype/artifacts/robustness-baseline/wcs-run'/rid/'reference.json'
                if refpath.exists():
                    from compare_local_wcs import compare
                    result['external_geometry'] = compare(json.loads(refpath.read_text()), artifact)
            log = folder/f'{rid}.log'
            result['attempt_timeouts'] = log.read_text().count('err=Timeout') if log.exists() else 0
            pair[mode] = result
    frames = list(pairs.values())
    def solved(frame, mode):
        return frame[mode].get('solve_status') == 'solved'
    stats = {}
    for track in ('solver', 'stress', 'scene'):
        group = [f for f in frames if f['track'] == track]
        stats[track] = {'count': len(group), **{mode: sum(solved(f, mode) for f in group)
                                             for mode in ('control', 'quantile')}}
    result = {'protocol_sha256': digest(out/'protocol.json'), 'protocol': protocol,
              'summaries': stats,
              'gained': [f['id'] for f in frames if solved(f, 'quantile') and not solved(f, 'control')],
              'lost': [f['id'] for f in frames if solved(f, 'control') and not solved(f, 'quantile')],
              'negative_accepts': {mode: [f['id'] for f in frames if f['negative'] and solved(f, mode)]
                                   for mode in ('control', 'quantile')},
              'frames': frames}
    write_json(out/'results.json', result)
    print(json.dumps({k: result[k] for k in ('summaries', 'gained', 'lost', 'negative_accepts')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('development', 'holdout'), required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, help='frozen development protocol (required for holdout)')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if args.summarize_only:
        summarize(out, json.loads((out/'protocol.json').read_text()))
        return
    if args.split == 'holdout' and args.protocol is None:
        parser.error('holdout requires a frozen development protocol')
    protocol = json.loads(args.protocol.read_text()) if args.protocol else new_protocol()
    if protocol['hashes'] != frozen_hashes():
        parser.error('frozen code, binary, catalog or inputs changed')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'protocol.json', protocol)
    records = select_records(json.loads(MANIFEST.read_text()), args.split)
    write_json(out/'selection.json', {'split': args.split, 'ids': [r['id'] for r in records]})
    for mode in ('control', 'quantile'):
        command = [sys.executable, str(ROOT/'prototype/eval/collection_run.py'), 'starglyph',
                   '--split', args.split, '--out-dir', str(out/mode)]
        if mode == 'quantile':
            command.append('--quantile-threshold')
        subprocess.run(command, check=True, cwd=ROOT/'prototype',
                       env={**os.environ, 'STARGLYPH_SOLVE_DEBUG': '1'})
    summarize(out, protocol)


if __name__ == '__main__':
    main()
