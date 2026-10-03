#!/usr/bin/env python3
"""Summarize observation-only internal traces, retaining actual target rejections."""
import argparse
from collections import Counter
import json
from pathlib import Path

from collection_run import digest, write_json


def read_trace(path):
    counts = Counter()
    target = []
    context = None
    current = {}
    for line in path.read_text().splitlines():
        if not line.startswith('SGTRACE '):
            continue
        row = json.loads(line[8:])
        stage, data = row['stage'], row['data']
        counts[stage] += 1
        if stage == 'attempt':
            context = data
            current = {}
        elif stage == 'lookup':
            current = dict(attempt=context, lookup=data)
        elif stage in ('ratios', 'pairing', 'rotation', 'verification_pool'):
            current[stage] = data
        elif stage == 'verification':
            current['verification'] = data
            target.append(current)
            current = {}
    return dict(event_counts=dict(counts), target_verifications=target)


def compact(out):
    arms = {}
    for arm, log, result in [('control', 'trace.log', 'trace-result.json'),
                             ('footprint', 'footprint-trace.log', 'footprint-trace-result.json')]:
        trace = read_trace(out / log)
        cases = json.loads((out / result).read_text())['cases']
        rows = []
        for t in trace['target_verifications']:
            pool = t['verification_pool']
            rows.append(dict(attempt=t['attempt'], seed_fov=t['pairing']['fov_seed'],
                measured_fov=t['pairing']['fov_refined'], pairing=t['pairing'], rotation=t['rotation'],
                image_ratios=t['ratios']['image'], catalog_ratios=t['ratios']['catalog'],
                probability=t['verification']['prob_mismatch'], threshold=t['verification']['threshold'],
                matched_pairs=t['verification']['pairs'], matched_count=t['verification']['matches'],
                accepted_by_tetra3=t['verification']['pass'], catalogue_before_budget=pool['before'],
                budget=pool['limit'], retained_catalogue_ids=pool['ids_before'][:pool['limit']]))
        arms[arm] = dict(log_sha256=digest(out / log), result_sha256=digest(out / result),
            event_counts=trace['event_counts'], target_verifications=rows, cases=cases)
    def example(arm):
        return next(r for r in arms[arm]['target_verifications']
            if r['attempt'] == dict(case='4080-default-control', db=0, k=30, fov=25.0)
            and r['seed_fov'] == 33.5)
    before, after = example('control'), example('footprint')
    same_attempt = dict(control=before, footprint=after,
        removed_from_limited_pool=sorted(set(before['retained_catalogue_ids'])-set(after['retained_catalogue_ids'])),
        added_to_limited_pool=sorted(set(after['retained_catalogue_ids'])-set(before['retained_catalogue_ids'])))
    for arm in arms.values():
        arm['target_verifications'] = [{key: row[key] for key in (
            'attempt', 'seed_fov', 'measured_fov', 'probability', 'threshold',
            'matched_count', 'accepted_by_tetra3', 'catalogue_before_budget', 'budget')}
            for row in arm['target_verifications']]
    return dict(iteration=4, split='development', id='wm_r_161606532',
        target_catalogue_ids=[73486, 74164, 76644, 78029],
        conclusion='correct indexed quartet reaches internal verification; out-of-frame catalogue stars consume the 2N brightness budget',
        external_reference='pending, stellar identities conditional on iteration-3 review',
        same_attempt=same_attempt,
        arms=arms)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('report', type=Path)
    a = p.parse_args()
    write_json(a.report, compact(a.out))


if __name__ == '__main__':
    main()
