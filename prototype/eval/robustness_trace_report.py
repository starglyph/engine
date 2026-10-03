#!/usr/bin/env python3
"""Compact the two development probe traces and matcher replays; no solves."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from collection_run import digest
from robustness_source_trace import IDS


def compact(run):
    frames = []
    for rid in IDS:
        input_path = run / f'{rid}-trace-input.json'
        trace_path = run / f'{rid}-traces.json'
        source = json.loads(input_path.read_text())
        trace = json.loads(trace_path.read_text())
        if source['split'] != 'development' or digest(Path(source['image'])) != source['image_sha256']:
            raise ValueError('changed or unscoped source')
        from PIL import Image
        with Image.open(source['image']) as im:
            original_size = im.size
        frame = dict(id=rid, source_sha256=source['image_sha256'],
            input_sha256=digest(input_path), trace_sha256=digest(trace_path),
            provenance=source['provenance'], original_probes=source['probes'], tiers=[])
        for tier in trace['tiers']:
            components = {}
            probes = []
            outcomes = Counter()
            for i, probe in enumerate(tier['probes']):
                component = probe['component']
                outcome = component['outcome'] if component else 'no_component_in_radius'
                outcomes[outcome] += 1
                row = dict(probe=i, component=component['component'] if component else None,
                           distance_to_component_working_px=probe['distance_to_component'])
                if component:
                    components[str(component['component'])] = component
                    center = component['centroid']
                    if center:
                        point = source['probes'][i]
                        dx = (center[0]+.5)*original_size[0]/tier['width']-.5-point[0]
                        dy = (center[1]+.5)*original_size[1]/tier['height']-.5-point[1]
                        row['centroid_displacement_original_px'] = math.hypot(dx,dy)
                probes.append(row)
            frame['tiers'].append(dict(width=tier['width'], height=tier['height'], tier=tier['tier'],
                quantile=tier['quantile'], blob_concentration=tier['blob_concentration'],
                stats=tier['result']['stats'], detections=len(tier['result']['detections']),
                outcomes=dict(outcomes), components=components, probes=probes))
        replay_tier = 'default' if rid == IDS[0] else 'deep'
        replay_input = run / f'{rid}-1600-{replay_tier}-replay-input.json'
        replay_output = run / f'{rid}-1600-{replay_tier}-replay.json'
        inp = json.loads(replay_input.read_text())
        result = json.loads(replay_output.read_text())
        result.setdefault('tier', replay_tier)
        frame['replay'] = dict(input_sha256=digest(replay_input), output_sha256=digest(replay_output),
            provenance=inp['provenance'],
            counts=[len(case['detections']) for case in inp['cases']], **result)
        frames.append(frame)
    return dict(schema_version=1, split='development',
        scope='Source filtering and matcher acceptance only; no certified star identities or full-field truth.',
        coordinate_convention='top_left_zero_based; half-pixel resize',
        component_ids='Local to a detector pass; match across runs by probe/bounds, not component integer.',
        radius_original_px=12, script_sha256=digest(Path(__file__)),
        frames=frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    # Full-precision source traces remain hashed local artifacts. Six decimal
    # places are far below a pixel and keep the published diagnostic compact.
    def rounded(value):
        if isinstance(value, float):
            return round(value, 6)
        if isinstance(value, dict):
            return {k: rounded(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rounded(v) for v in value]
        return value
    result = rounded(compact(args.run))
    result['numeric_precision'] = 'six decimals; raw traces retain full precision'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False,
                                      separators=(',', ':')) + '\n')


if __name__ == '__main__':
    main()
