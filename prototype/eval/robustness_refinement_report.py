#!/usr/bin/env python3
"""Measure every candidate/refinement stage on frozen iteration-5 observations."""
import argparse
import json
from pathlib import Path

import numpy as np

from collection_run import ROOT, digest, write_json
from compare_local_wcs import project, stats
from robustness_geometry import report_path, selected
from robustness_refinement import RID


def lift(xy, camera, width, height):
    return (np.asarray(xy) + .5) * [width/camera['width'], height/camera['height']] - .5


def summarize_stage(stage, sources, width, height):
    camera = stage['camera']
    world = [[s['ra_deg'], s['dec_deg']] for s in sources]
    measured = np.array([s['xy'] for s in sources])
    projected = lift(project(camera, world), camera, width, height)
    errors = np.linalg.norm(projected-measured, axis=1)
    pairs = []
    for pair in stage['pairs']:
        xy = lift(pair['xy'], camera, width, height)
        projection = lift(pair['projected_xy'], camera, width, height)
        distance = np.linalg.norm(measured-xy, axis=1)
        nearest = int(np.argmin(distance))
        pairs.append(dict(hyg_ids=pair['hyg_ids'], xy=xy.tolist(),
            residual_px=float(np.linalg.norm(projection-xy)),
            nearest_reviewed_source=sources[nearest]['id'], nearest_reviewed_distance_px=float(distance[nearest]),
            reviewed_identity_agrees=(sources[nearest]['hyg_id'] in pair['hyg_ids']) if distance[nearest] < 12 else None))
    masks = dict(all_reviewed=np.ones(len(sources), dtype=bool),
        outside_both_inputs=np.array([s['outside_both_inputs'] for s in sources]),
        left_10pct=measured[:, 0] < width*.1, right_10pct=measured[:, 0] > width*.9,
        top_10pct=measured[:, 1] < height*.1, bottom_10pct=measured[:, 1] > height*.9)
    return dict(stage=stage['stage'], camera=camera,
        groups={k:stats(errors[m]) if m.any() else dict(count=0) for k,m in masks.items()},
        source_errors_px=errors.tolist(), projected_xy=projected.tolist(), fit_pairs=pairs)


def compare_saved(tier, candidate, saved):
    camera = candidate['stages'][-1]['camera']
    expected = saved['camera']
    errors = {k:float(np.max(np.abs(np.array(camera[k])-np.array(expected[k]))))
              for k in ('focal_px','k1','world_to_camera','width','height')}
    if max(errors.values()) > 1e-8:
        raise ValueError(f'selected camera differs from saved pipeline: {errors}')
    actual = lift([[d['x'],d['y']] for d in tier['detections']], tier, saved['width'], saved['height'])
    previous = np.array([[d['x'],d['y']] for d in saved['report']['detections']])
    if actual.shape != previous.shape or not np.allclose(actual, previous, atol=.00501, rtol=0):
        raise ValueError('selected detector input differs from saved pipeline')
    if len(candidate['stages'][-1]['pairs']) != saved['report']['quality']['n_inliers']:
        raise ValueError('selected inlier count differs')
    return dict(status='passed', camera_max_abs_difference=errors,
                detections_max_abs_difference_px=float(np.max(np.abs(actual-previous))),
                detections_tolerance_px=.00501)


def summarize(out, output):
    protocol = json.loads((out/'protocol.json').read_text())
    if protocol['split'] != 'development' or protocol['id'] != RID:
        raise ValueError('frozen development ID required')
    selected([RID])
    for path, expected in protocol['hashes'].items():
        if digest(Path(path)) != expected:
            raise ValueError('frozen input changed: '+path)
    review_path = ROOT/'docs/experiments/robustness-iteration-5-review.json'
    review = next(f for f in json.loads(review_path.read_text())['frames'] if f['id']==RID)
    prior_geometry = json.loads((ROOT/'docs/experiments/robustness-iteration-5-geometry.json').read_text())
    prior_protocol_path = ROOT/'docs/experiments/robustness-iteration-5-protocol.json'
    if digest(prior_protocol_path) != prior_geometry['protocol_sha256']:
        raise ValueError('iteration-5 protocol changed')
    prior_protocol = json.loads(prior_protocol_path.read_text())
    for arm in ('control', 'footprint'):
        saved_path = report_path(RID,arm)
        if digest(saved_path) != prior_protocol['hashes'][str(saved_path.relative_to(ROOT))]:
            raise ValueError('saved comparison camera changed')
    prior = next(f for f in prior_geometry['frames'] if f['id']==RID)
    sources = []
    for source in prior['sources']:
        point = next(p for p in review['points'] if p['id']==source['source_id'])
        sources.append(dict(id=source['source_id'], hyg_id=source['hyg_id'], name=source['name'],
            xy=source['xy'], outside_both_inputs=source['outside_both_inputs'],ra_deg=point['ra_deg'],dec_deg=point['dec_deg']))
    results, checks = [], {}
    traces = {}
    for arm, selected_tier in [('control','deep'),('footprint','default')]:
        trace = json.loads((out/arm/'trace.json').read_text())
        if trace['split'] != 'development' or trace['id'] != RID:
            raise ValueError('trace selection differs')
        traces[arm] = trace
        for tier in trace['tiers']:
            for candidate in tier['candidates']:
                production_chosen = tier['tier']==selected_tier and candidate['chosen']
                if production_chosen:
                    checks[arm] = compare_saved(tier,candidate,json.loads(report_path(RID,arm).read_text()))
                results.append(dict(arm=arm,tier=tier['tier'],index=candidate['index'],label=candidate['label'],
                    production_chosen=production_chosen,chosen_within_tier=candidate['chosen'],hard=candidate['hard'],
                    verified=candidate['verified'],initial_hits=candidate['hits'],initial_log_odds=candidate['log_odds'],
                    matching_ms=tier['matching_ms'],refinement_ms=candidate['refinement_ms'],
                    stages=[summarize_stage(s,sources,review['width'],review['height']) for s in candidate['stages']]))
    if set(checks) != {'control','footprint'}:
        raise ValueError('missing production replay check')
    for a,b in zip(traces['control']['tiers'],traces['footprint']['tiers']):
        if a['tier'] != b['tier'] or a['detections'] != b['detections']:
            raise ValueError('arms have different detector inputs')
    write_json(output,dict(iteration=6,id=RID,split='development',protocol_sha256=digest(out/'protocol.json'),
        review_sha256=digest(review_path),sources=sources,candidates=results,reproduction_checks=checks,
        detector_coordinates_original={t['tier']:lift([[d['x'],d['y']] for d in t['detections']],
            t,review['width'],review['height']).tolist() for t in traces['control']['tiers']},
        detector_inputs_identical_between_arms=True,
        tiers={a:[dict(tier=t['tier'],n_detections=len(t['detections']),chosen_index=t['chosen_index'],
                       matching_ms=t['matching_ms'],candidate_count=len(t['candidates'])) for t in trace['tiers']]
               for a,trace in traces.items()},
        trace_sha256={a:digest(out/a/'trace.json') for a in traces},
        limitations=['unchosen candidate refinements and forced deep tier are diagnostic extra work',
                    'identities conditional on iteration-5 visual review; no independent absolute ground truth',
                    'edge groups may contain fit sources; outside-both-input group remains fixed from iteration 5']))


def compact_report(summary):
    """Keep every source error and fit identity; omit reconstructible projections."""
    compact = json.loads(json.dumps(summary))
    compact['fit_pair_columns'] = ['hyg_ids', 'x_original', 'y_original', 'residual_original_px']
    for candidate in compact['candidates']:
        for stage in candidate['stages']:
            del stage['projected_xy']
            stage['source_errors_px'] = [round(x, 6) for x in stage['source_errors_px']]
            stage['fit_pairs'] = [[p['hyg_ids'], *[round(x, 6) for x in p['xy']], round(p['residual_px'], 6)]
                                  for p in stage['fit_pairs']]
    return compact


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compact-output',type=Path)
    a=p.parse_args()
    summarize(a.out.resolve(),a.output)
    if a.compact_output is not None:
        compact = compact_report(json.loads(a.output.read_text()))
        a.compact_output.write_text(json.dumps(compact, ensure_ascii=False, separators=(',', ':'))+'\n')
