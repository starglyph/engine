#!/usr/bin/env python3
"""Compare complete paired mask runs across solver binaries, preserving per-ID checks."""
import argparse
import json
from pathlib import Path
import statistics

from automatic_sky_compare import MODES, transitions
from sky_statistics_experiment import canonical_report
from smartphone_gate import read_json, require, sha256


def verify_run(directory, binary):
    plan = read_json(directory / 'plan.json')
    provenance = read_json(directory / 'provenance.json')
    require(sha256(directory / 'plan.json') == provenance['plan_sha256'], 'plan hash mismatch')
    for name, digest in plan['inputs'].items():
        path = binary if name == plan['binary'] else Path(name)
        require(sha256(path) == digest, f'input hash mismatch: {path}')
    for name, digest in provenance['artifacts'].items():
        require(sha256(directory / name) == digest, f'artifact hash mismatch: {name}')
    require(all(provenance['runs'][mode]['exit_code'] == 0 for mode in MODES), 'incomplete run')
    return plan, provenance


def compare(before, after, before_binary, after_binary):
    a_plan, a_provenance = verify_run(before, before_binary)
    b_plan, b_provenance = verify_run(after, after_binary)
    for field in ('manifest', 'catalog', 'baseline', 'manual_baseline', 'manual_masks', 'probes'):
        require(a_plan['inputs'][a_plan[field]] == b_plan['inputs'][b_plan[field]], f'{field} differs')
    for field in ('algorithm', 'config', 'split'):
        require(a_plan[field] == b_plan[field], f'{field} differs')
    require(sha256(before / 'automatic-masks.json') == sha256(after / 'automatic-masks.json'), 'automatic masks differ')
    entries = read_json(Path(a_plan['manifest']))
    output = {}
    for mode in MODES:
        configs = [read_json(p / mode / 'summary.json')['config'] for p in (before, after)]
        for config in configs:
            mask = config.pop('sky_masks', None)
            config['sky_masks_sha256'] = sha256(Path(mask)) if mask else None
        require(configs[0] == configs[1], f'{mode}: solver configurations differ')
        solved = [[], []]
        times = [[], []]
        changed = []
        rows = []
        for entry in entries:
            frame_id = entry['id']
            artifacts = [read_json(p / mode / 'solve-reports' / f'{frame_id}.json') for p in (before, after)]
            reports = [a['report'] for a in artifacts]
            for i, (artifact, report) in enumerate(zip(artifacts, reports)):
                require(artifact['source_sha256'] == entry['sha256'], f'{frame_id}: source mismatch')
                times[i].append(report['timing_ms']['total'])
                if report['status'] == 'solved':
                    solved[i].append(frame_id)
            if reports[0]['status'] == 'solved' and (
                    artifacts[0]['camera'] != artifacts[1]['camera'] or
                    canonical_report(reports[0]) != canonical_report(reports[1])):
                changed.append(frame_id)
            rows.append(dict(id=frame_id, before=reports[0].get('quality'), after=reports[1].get('quality'),
                             statuses=[r['status'] for r in reports],
                             total_ms=[r['timing_ms']['total'] for r in reports]))
        delta = transitions(*solved)
        require(not delta['lost'], f'{mode}: lost solutions {delta["lost"]}')
        output[mode] = dict(solved_counts=list(map(len, solved)), **delta,
                            prior_solutions_changed=changed, rows=rows,
                            median_total_ms=list(map(statistics.median, times)),
                            max_total_ms=list(map(max, times)))
    return dict(modes=output, before_plan_sha256=a_provenance['plan_sha256'],
                after_plan_sha256=b_provenance['plan_sha256'],
                before_binary_sha256=sha256(before_binary), after_binary_sha256=sha256(after_binary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--before-binary', type=Path, required=True)
    parser.add_argument('--after-binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.before, args.after, args.before_binary, args.after_binary)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
