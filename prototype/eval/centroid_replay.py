#!/usr/bin/env python3
"""Prepare controlled reduced/deep matching cases; no reference pose is used."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def prepare_cases(lists, radius, names=('control', 'variant')):
    """Factor membership, common coordinates and common order without pose labels."""
    # Greedy one-to-one geometric correspondence, independent of solved inlier labels.
    edges = sorted(((a['x']-b['x'])**2 + (a['y']-b['y'])**2, i, j)
                   for i, a in enumerate(lists[0]) for j, b in enumerate(lists[1]))
    used_a, used_b, pairs = set(), set(), []
    for distance, i, j in edges:
        if distance <= radius**2 and i not in used_a and j not in used_b:
            pairs.append((i, j))
            used_a.add(i)
            used_b.add(j)
    cases = [dict(name=mode, detections=ds) for mode, ds in zip(names, lists)]
    for side, mode in enumerate(names):
        ds = copy.deepcopy(lists[side])
        for pair in pairs:
            other = lists[1-side][pair[1-side]]
            ds[pair[side]].update(x=other['x'], y=other['y'])
        cases.append(dict(name=f'{mode}_swapped_common_coordinates', detections=ds))
        # Preserve all membership, coordinates, unmatched slots and slot brightness.
        # Permute only common sources within their original common-source slots.
        ds = copy.deepcopy(lists[side])
        slots = sorted(pair[side] for pair in pairs)
        reordered = sorted(pairs, key=lambda p: p[1-side])
        for slot, pair in zip(slots, reordered):
            d = copy.deepcopy(lists[side][pair[side]])
            d.update(rank=slot, flux=lists[side][slot]['flux'])
            ds[slot] = d
        cases.append(dict(name=f'{mode}_swapped_common_order', detections=ds))
    # Fixed common membership: factorial test of coordinates and brightness ordering.
    for coordinates in range(2):
        for ordering in range(2):
            ds = []
            for rank, pair in enumerate(sorted(pairs, key=lambda p: p[ordering])):
                d = copy.deepcopy(lists[coordinates][pair[coordinates]])
                d.update(rank=rank, flux=lists[ordering][pair[ordering]]['flux'])
                ds.append(d)
            cases.append(dict(name=f'common_coordinates_{coordinates}_order_{ordering}', detections=ds))
    return cases, pairs


def prepare(run, frame_id, radius=4.8):
    sources = [run / mode / 'solve-reports' / f'{frame_id}.json'
               for mode in ('manual', 'automatic')]
    reports = [json.loads(p.read_text()) for p in sources]
    diagnostics = [next(d for d in r['detection_diagnostics']
                        if d['width'] == 1600 and d['tier'] == 'deep') for r in reports]
    assert diagnostics[0]['height'] == diagnostics[1]['height']
    lists = [d['result']['detections'][:d['max_detections']] for d in diagnostics]
    cases, pairs = prepare_cases(lists, radius, ('manual', 'automatic'))
    root = Path(__file__).resolve().parents[2]
    return dict(image=str(root / 'data/input/smartphone' / f'{frame_id}.jpg'),
                catalog=str(root / 'data/catalogs/hyg_v42.csv.gz'),
                cache=str(root / 'prototype/artifacts/cache'), width=1600,
                height=diagnostics[0]['height'], cases=cases,
                provenance=dict(source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in sources}, radius_working_px=radius,
                                pairs=pairs, counts=list(map(len, lists)),
                                note='0=manual, 1=automatic; matching only, before refinement; observed research frame'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--frame-id', default='20260829_220020')
    args = parser.parse_args()
    args.output.write_text(json.dumps(prepare(args.run.resolve(), args.frame_id), indent=2) + '\n')
