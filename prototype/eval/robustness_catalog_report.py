#!/usr/bin/env python3
"""Compact iteration-3 evidence. Pattern membership is necessary, not sufficient."""
import argparse
from collections import Counter
import json
from pathlib import Path

from collection_run import ROOT, digest, write_json


def prefixes(n):
    return sorted({min(k, n) for k in (8, 10, 12, 16, 20, 30) if min(k, n) >= 4})


def pattern_count(patterns, ids):
    present = set(ids) - {None}
    return sum(set(p) <= present for p in patterns)


def compact(out):
    stages = []
    for name, folder in [('corr_subset', out), ('expanded_identities', out / 'expanded'),
                         ('restore_dschubba', out / 'restore')]:
        inputs = json.loads((folder / 'input.json').read_text())
        results = json.loads((folder / 'replay-final.json').read_text())
        freeze = json.loads((folder / 'freeze-final.json').read_text())
        for path, sha in freeze['hashes'].items():
            if digest(Path(path)) != sha:
                raise ValueError(f'frozen research code or input changed: {path}')
        if inputs['protocol']['split'] != 'development':
            raise ValueError('development required')
        databases = results['databases']
        cases = []
        if len(inputs['cases']) != len(results['cases']):
            raise ValueError('incomplete replay')
        for source, result in zip(inputs['cases'], results['cases']):
            if source['name'] != result['name']:
                raise ValueError('case order changed')
            coverage = [dict(k=k, known_ids=sum(i is not None for i in source['search_ids'][:k]),
                patterns_by_database=[pattern_count(d['probe_patterns'], source['search_ids'][:k]) for d in databases])
                for k in prefixes(len(source['search']))]
            cases.append(dict(name=source['name'], detections=result['search_count'],
                verification_detections=result['verification_count'],
                search_ids=source['search_ids'], prefix_coverage=coverage,
                outcome='candidate' if result['chosen_before_refinement'] else 'no_candidate',
                chosen_before_refinement=result['chosen_before_refinement'],
                geometry_independently_confirmed=False,
                inliers=None if result['chosen_before_refinement'] is None else result['chosen_before_refinement']['hits'],
                elapsed_ms=result['elapsed_ms'], attempts=len(result['attempts']),
                attempt_status=dict(Counter(a['result'].get('error', 'candidate') for a in result['attempts'])),
                candidates=[a for a in result['attempts'] if 'error' not in a['result']]))
        stages.append(dict(stage=name, protocol=inputs['protocol'],
            raw_sha256={f: digest(folder / f) for f in ('input.json', 'replay-final.json', 'freeze-final.json', 'run-final.json')},
            run=json.loads((folder / 'run-final.json').read_text()),
            databases=[dict(file=d['file'], stars=d['stars'], properties=d['properties'],
                present_probe_count=len(d['probe_ids_present']),
                absent_probe_ids=sorted(set(inputs['probe_ids'])-set(d['probe_ids_present'])),
                patterns_within_reviewed_pool=len(d['probe_patterns']),
                common_ids_in_all_reviewed_patterns=sorted(set.intersection(*(set(p) for p in d['probe_patterns']))) if d['probe_patterns'] else [],
                probe_pattern_participation=d['probe_pattern_participation']) for d in databases],
            cases=cases))
    source = json.loads((out / 'sources.json').read_text())
    trace_input = json.loads((out / 'expanded/trace-input.json').read_text())
    traces = json.loads((out / 'expanded/traces.json').read_text())['tiers']
    trace_rows = []
    for tier in traces:
        if tier['quantile'] or tier['blob_concentration']:
            continue
        sources = []
        for identity, probe in zip(trace_input['hyg_ids'], tier['probes']):
            c = probe['component'] or {}
            sources.append(dict(hyg_id=identity, outcome=c.get('outcome', 'no_component'),
                rank=c.get('rank_before_top_k'), area=c.get('area'),
                measured_area=c.get('measured_area'), concentration=c.get('concentration')))
        trace_rows.append(dict(width=tier['width'], tier=tier['tier'], sources=sources,
            dschubba=tier['probes'][trace_input['hyg_ids'].index(78165)]))
    return dict(iteration=3, id='wm_r_161606532', split='development',
        conclusion='no automatic improvement; detector loses catalogue anchors, their restoration alone does not solve',
        external_wcs_status='pending; identity evidence supports visible sky, not global geometry',
        pattern_count_scope='known catalogue ID membership before tetra3 thinning and geometry/probability checks',
        source_associations=source['associations'],
        noncorr_predictions=json.loads((out / 'noncorr-predictions.json').read_text()),
        detector_identity_associations=json.loads((out / 'expanded/associations.json').read_text()),
        source_traces=trace_rows, stages=stages,
        positive_control=dict(id='wm_r_112929230', split='development',
            replay=json.loads((out / 'positive-replay.json').read_text())['cases'],
            raw_sha256=digest(out / 'positive-replay.json')),
        limitations=['catalogue-informed source selection is diagnostic only',
            'all failure interventions concern one development image, not independent samples',
            'membership alone cannot locate internal tetra3 rejection',
            'no new full-collection or holdout evaluation; no production algorithm changed'])


def rounded(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/experiments/robustness-iteration-3-diagnosis.json')
    args = parser.parse_args()
    write_json(args.output, rounded(compact(args.artifacts)))


if __name__ == '__main__':
    main()
