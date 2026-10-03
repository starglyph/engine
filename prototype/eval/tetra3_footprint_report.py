#!/usr/bin/env python3
"""Summarize frozen development A/B; external references remain pending."""
import argparse
import json
from pathlib import Path
import statistics

from collection_run import ROOT, digest, select_records, write_json


def summarize(out, split='development'):
    protocol = json.loads((out / 'protocol.json').read_text())
    if split not in ('development', 'holdout') or protocol['split'] != split:
        raise ValueError('explicit matching split required')
    for path, sha in protocol['hashes'].items():
        if digest(Path(path)) != sha:
            raise ValueError(f'frozen input changed: {path}')
    manifest = ROOT / 'data/samples/sky-samples/manifest.json'
    records = select_records(json.loads(manifest.read_text()), split)
    if [r['id'] for r in records] != protocol['ids']:
        raise ValueError('selection changed')
    baseline = {r['id']: r for r in json.loads((manifest.parent / 'robustness-results.json').read_text())['frames']}
    frames = {r['id']: dict(id=r['id'], track=r['track'], negative=r['research'].get('negative', False),
        source_sha256=r['clean_sha256'], published_status=baseline[r['id']]['starglyph'].get('solve_status'),
        independent_ground_truth=False) for r in records}
    for arm in ('control', 'footprint'):
        folder = out / split / arm
        plan = json.loads((folder / 'plan.json').read_text())
        summary = json.loads((folder / 'summary.json').read_text())
        if (plan['selection']['split'] != split or plan['selection']['ids'] != protocol['ids']
                or summary['completed'] != len(records) or summary['plan_sha256'] != digest(folder / 'plan.json')
                or [r['id'] for r in summary['frames']] != protocol['ids']):
            raise ValueError('incomplete or mixed run')
        if (plan['binary_sha256'] != protocol['hashes'][protocol['binaries'][arm]] or plan['wall_limit_s'] != 120
                or any(plan.get(k) for k in ('quantile_threshold', 'blob_concentration', 'sky_masks_sha256'))):
            raise ValueError('wrong variant or budget')
        for row in summary['frames']:
            rid = row['id']
            result = {k: row[k] for k in ('status', 'solve_status', 'elapsed_s', 'exit_code', 'reason') if k in row}
            if 'report' in row:
                path = Path(row['report'])
                if digest(path) != row['report_sha256']:
                    raise ValueError('report changed')
                artifact = json.loads(path.read_text())
                if artifact['source_sha256'] != frames[rid]['source_sha256']:
                    raise ValueError('wrong image')
                report = artifact['report']
                result.update(report_sha256=digest(path), detections=len(report['detections']),
                    quality=report.get('quality'), failure=report.get('failure'), timing_ms=report.get('timing_ms'))
                refpath = ROOT / f'prototype/artifacts/robustness-baseline/wcs-run/{rid}/reference.json'
                if refpath.exists():
                    from compare_local_wcs import compare
                    reference = json.loads(refpath.read_text())
                    for key in ('wcs_file', 'correspondences_file'):
                        if key in reference:
                            reference[key] = str(ROOT / 'prototype' / reference[key])
                    result['pending_external_geometry'] = compare(reference, artifact)
            log = folder / f'{rid}.log'
            result['attempt_timeouts'] = log.read_text().count('err=Timeout') if log.exists() else None
            frames[rid][arm] = result
    rows = list(frames.values())
    solved = lambda r, arm: r[arm].get('solve_status') == 'solved'
    totals = {t: dict(count=sum(r['track'] == t for r in rows), **{
        a: sum(r['track'] == t and solved(r, a) for r in rows) for a in ('control', 'footprint')})
        for t in ('solver', 'stress', 'scene')}
    timings = {a: dict(total_s=sum(r[a].get('elapsed_s', 0) for r in rows),
                      median_s=statistics.median(r[a]['elapsed_s'] for r in rows if 'elapsed_s' in r[a]),
                      attempt_timeouts=sum(r[a]['attempt_timeouts'] or 0 for r in rows))
               for a in ('control', 'footprint')}
    return dict(iteration=4, split=split, protocol_sha256=digest(out / 'protocol.json'),
        patch_sha256=digest(Path(__file__).with_name('tetra3_verify_footprint.patch')),
        binaries={a: dict(path=p, sha256=protocol['hashes'][p]) for a,p in protocol['binaries'].items()},
        totals=totals, timing=timings, gained=[r['id'] for r in rows if solved(r,'footprint') and not solved(r,'control')],
        lost=[r['id'] for r in rows if solved(r,'control') and not solved(r,'footprint')],
        negative_accepts={a: [r['id'] for r in rows if r['negative'] and solved(r,a)] for a in ('control','footprint')},
        baseline_status_mismatches=[r['id'] for r in rows if r['control'].get('solve_status') != r['published_status']],
        frames=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('report', type=Path)
    p.add_argument('--split', choices=('development', 'holdout'), default='development')
    a = p.parse_args()
    result = summarize(a.out, a.split)
    write_json(a.report, result)
    print(json.dumps({k:result[k] for k in ('totals','gained','lost','negative_accepts','timing')}, indent=2))


if __name__ == '__main__':
    main()
