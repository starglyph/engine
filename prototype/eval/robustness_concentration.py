#!/usr/bin/env python3
"""Frozen development-only A/B of component concentration (iteration 2).

Use prepare, then run each arm in bounded batches, then summarize. No holdout
mode, downloads, ground-truth promotion, or parameter fitting is provided.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from collection_run import ROOT, digest, select_records, write_json
from robustness_iteration import MANIFEST, BINARY, CATALOG, BASELINE

MODES = ('control', 'blob')


def source_hashes():
    files = sorted((ROOT / 'prototype/crates').glob('*/src/**/*.rs'))
    files += [ROOT / 'prototype/Cargo.lock', BINARY, CATALOG, MANIFEST, BASELINE,
              MANIFEST.parent / 'collection-wcs-review.json']
    files += [ROOT / 'prototype/eval' / name for name in (
        'collection_run.py', 'robustness_concentration.py', 'robustness_source_trace.py',
        'centroid_replay.py', 'robustness_iteration.py', 'compare_local_wcs.py',
        'robustness_trace_report.py')]
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def prepare(out):
    records = select_records(json.loads(MANIFEST.read_text()), 'development')
    inputs = [dict(id=r['id'], source_sha256=r['clean_sha256'], track=r['track'],
                   negative=r['research'].get('negative', False)) for r in records]
    for r in records:
        if digest(MANIFEST.parent / r['file']) != r['clean_sha256']:
            raise ValueError('input hash mismatch')
    protocol = dict(schema_version=1, experiment='component_concentration_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), split='development',
        base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        hashes=source_hashes(), inputs=inputs,
        parameters=dict(concentration='peak * component_area / sum_positive_residual_in_component',
                        min_concentration=2.5, min_area=25, max_area=150,
                        quantile_threshold=False, masks=False, wall_limit_s=120),
        unchanged=['saturation exemption', 'core resegmentation', 'centroid and flux ranking',
                   'matching search budget', 'acceptance and refinement'],
        criteria=['Retain every previous success; no accepted development negative.',
                  'No new process errors or wall timeouts; report attempt timeouts and elapsed time.',
                  'New solutions remain provisional pending independent identity and edge verification.',
                  'A detection proxy gain alone is insufficient; no default activation on regressions.',
                  'One fixed variant, no parameter sweep, no holdout access.'],
        cache_hashes={p.name: digest(p) for p in sorted((ROOT/'prototype/artifacts/cache').glob('*.bin'))})
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'protocol.json', protocol)


def checked_protocol(out):
    protocol = json.loads((out / 'protocol.json').read_text())
    if protocol['hashes'] != source_hashes():
        raise ValueError('frozen source, binary or data changed')
    if protocol['split'] != 'development':
        raise ValueError('only development is allowed')
    return protocol


def run(out):
    """Alternate six-frame batches, resuming only completed prefixes."""
    protocol = checked_protocol(out)
    remaining = set(MODES)
    while remaining:
        for mode in MODES:
            if mode not in remaining:
                continue
            checked_protocol(out)
            folder = out / mode
            summary_path = folder / 'summary.json'
            prior = json.loads(summary_path.read_text()) if summary_path.exists() else None
            before = prior['completed'] if prior else 0
            if before == len(protocol['inputs']):
                remaining.remove(mode)
                continue
            command = [sys.executable, str(ROOT/'prototype/eval/collection_run.py'), 'starglyph',
                       '--split', 'development', '--out-dir', str(folder), '--batch-size', '6']
            if mode == 'blob':
                command.append('--blob-concentration')
            if prior:
                command.append('--resume')
            print(f'Running {mode} after {before} completed frames', flush=True)
            subprocess.run(command, cwd=ROOT/'prototype', check=True,
                           env={**os.environ, 'STARGLYPH_SOLVE_DEBUG': '1'})
            if json.loads(summary_path.read_text())['completed'] <= before:
                raise ValueError('executor made no progress')


def summarize(out):
    protocol = checked_protocol(out)
    expected = [r['id'] for r in protocol['inputs']]
    original = {r['id']: r for r in json.loads(BASELINE.read_text())['frames']}
    frames = {r['id']: {**r, 'published_status': original[r['id']]['starglyph'].get('solve_status'),
                       'ground_truth': 'unconfirmed'} for r in protocol['inputs']}
    for mode in MODES:
        folder = out / mode
        plan = json.loads((folder / 'plan.json').read_text())
        summary = json.loads((folder / 'summary.json').read_text())
        if (plan.get('selection', {}).get('split') != 'development' or plan.get('quantile_threshold')
                or plan.get('sky_masks_sha256') or plan.get('detection_diagnostics')
                or bool(plan.get('blob_concentration')) != (mode == 'blob')):
            raise ValueError('wrong experiment configuration')
        for key, path in (('binary_sha256', BINARY), ('catalog_sha256', CATALOG),
                          ('manifest_sha256', MANIFEST)):
            if plan[key] != protocol['hashes'][str(path.relative_to(ROOT))]:
                raise ValueError('run does not match frozen protocol')
        if plan['wall_limit_s'] != protocol['parameters']['wall_limit_s']:
            raise ValueError('changed run budget')
        if (summary['plan_sha256'] != digest(folder/'plan.json')
                or [r['id'] for r in summary['frames']] != expected
                or [r['id'] for r in plan['inputs']] != expected
                or summary['completed'] != len(expected)):
            raise ValueError('incomplete or changed run')
        for row in summary['frames']:
            rid = row['id']
            result = {k: row[k] for k in ('status', 'solve_status', 'elapsed_s', 'exit_code', 'reason') if k in row}
            if 'report' in row:
                path = Path(row['report'])
                if digest(path) != row['report_sha256']:
                    raise ValueError('changed report')
                artifact = json.loads(path.read_text())
                if (artifact['source_sha256'] != frames[rid]['source_sha256']
                        or bool(artifact.get('blob_concentration')) != (mode == 'blob')
                        or artifact.get('quantile_threshold')):
                    raise ValueError('wrong report provenance')
                report = artifact['report']
                result.update(report_sha256=digest(path), detections=len(report['detections']),
                              quality=report.get('quality'), failure=report.get('failure'),
                              timing_ms=report.get('timing_ms'))
                refpath = ROOT / f'prototype/artifacts/robustness-baseline/wcs-run/{rid}/reference.json'
                if refpath.exists():
                    from compare_local_wcs import compare
                    result['pending_external_geometry'] = compare(json.loads(refpath.read_text()), artifact)
            log = folder / f'{rid}.log'
            result['attempt_timeouts'] = log.read_text().count('err=Timeout') if log.exists() else 0
            frames[rid][mode] = result
    rows = list(frames.values())
    def solved(row, mode):
        return row[mode].get('solve_status') == 'solved'
    totals = {track: dict(count=sum(r['track'] == track for r in rows),
        **{mode: sum(r['track'] == track and solved(r, mode) for r in rows) for mode in MODES})
        for track in ('solver', 'stress', 'scene')}
    result = dict(protocol=protocol, protocol_sha256=digest(out/'protocol.json'), totals=totals,
        gained=[r['id'] for r in rows if solved(r, 'blob') and not solved(r, 'control')],
        lost=[r['id'] for r in rows if solved(r, 'control') and not solved(r, 'blob')],
        negative_accepts={m: [r['id'] for r in rows if r['negative'] and solved(r,m)] for m in MODES},
        frames=rows)
    write_json(out/'results.json', result)
    print(json.dumps({k: v for k,v in result.items() if k not in ('protocol', 'frames')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'check', 'run', 'summarize'))
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    {'prepare': prepare, 'check': checked_protocol, 'run': run,
     'summarize': summarize}[args.action](args.out.resolve())


if __name__ == '__main__':
    main()
