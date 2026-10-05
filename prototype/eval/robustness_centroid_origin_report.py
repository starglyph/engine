#!/usr/bin/env python3
"""Per-ID A/B outcomes and frozen reviewed field geometry; no new WCS or fit."""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np

from collection_run import ROOT, digest, write_json
from compare_local_wcs import compare, project, stats
from robustness_centroid_origin import ARMS, MANIFEST, records, validate


def read_arm(out, protocol, arm):
    folder = out / 'development' / arm
    plan = json.loads((folder / 'plan.json').read_text())
    summary = json.loads((folder / 'summary.json').read_text())
    if (plan['selection'] != dict(split='development', ids=protocol['ids'])
            or summary['completed'] != len(protocol['ids'])
            or summary['plan_sha256'] != digest(folder / 'plan.json')
            or [r['id'] for r in summary['frames']] != protocol['ids']):
        raise ValueError('incomplete or mixed development run')
    if (plan['binary_sha256'] != protocol['hashes'][protocol['binaries'][arm]]
            or plan['wall_limit_s'] != protocol['wall_limit_s']
            or any(plan.get(k) for k in ('quantile_threshold', 'blob_concentration', 'sky_masks_sha256'))):
        raise ValueError('variant or budget changed')
    results, artifacts = {}, {}
    for row in summary['frames']:
        rid = row['id']
        result = {k:row[k] for k in ('status', 'solve_status', 'elapsed_s', 'exit_code', 'reason') if k in row}
        if 'report' in row:
            path = Path(row['report'])
            if digest(path) != row['report_sha256']:
                raise ValueError('saved report changed')
            artifact = json.loads(path.read_text())
            expected = next(r for r in plan['inputs'] if r['id'] == rid)
            if artifact['source_sha256'] != expected['sha256']:
                raise ValueError('wrong source image')
            artifacts[rid] = artifact
            report = artifact['report']
            result.update(report_sha256=digest(path), detections=len(report['detections']),
                          quality=report.get('quality'), failure=report.get('failure'), timing_ms=report.get('timing_ms'))
            reference = ROOT / f'prototype/artifacts/robustness-baseline/wcs-run/{rid}/reference.json'
            if reference.exists() and report['status'] == 'solved':
                ref = json.loads(reference.read_text())
                if ref['status'] == 'solved_candidate':
                    for key in ('wcs_file', 'correspondences_file'):
                        ref[key] = str(ROOT / 'prototype' / ref[key])
                    result['pending_external_geometry'] = compare(ref, artifact)
        log = folder / f'{rid}.log'
        result['attempt_timeouts'] = log.read_text().count('err=Timeout') if log.exists() else 0
        results[rid] = result
    return results, artifacts


def geometry_groups(xy, detections, width, height, exclusion):
    outside = np.ones(len(xy), dtype=bool)
    for det in detections:
        if len(det):
            outside &= np.min(np.linalg.norm(xy[:, None] - det[None, :], axis=2), axis=1) >= exclusion
    x, y = xy.T
    return dict(all_reviewed=np.ones(len(x), dtype=bool), outside_current_inputs=outside,
                left_10pct=x < width*.1, right_10pct=x > width*.9,
                top_10pct=y < height*.1, bottom_10pct=y > height*.9)


def reviewed_geometry(artifacts, settings):
    frozen = json.loads((ROOT / 'docs/experiments/robustness-iteration-13-geometry.json').read_text())
    alternate = json.loads((ROOT / 'docs/experiments/robustness-iteration-5-geometry.json').read_text())
    results = []
    for frame in frozen['frames']:
        rid = frame['id']
        if any(rid not in artifacts[a] or artifacts[a][rid]['report']['status'] != 'solved' for a in ARMS):
            results.append(dict(id=rid, status='not_comparable'))
            continue
        reports = {a:artifacts[a][rid] for a in ARMS}
        width, height = reports['control']['width'], reports['control']['height']
        xy = np.array([s['xy'] for s in frame['sources']])
        radec = np.array([s['radec'] for s in frame['sources']])
        worlds = np.array([s['world'] for s in frame['sources']])
        alternate_rows = {s['source_id']:s for f in alternate['frames'] if f['id'] == rid for s in f['sources']}
        probes = {'external_centroid':xy}
        for radius in (8, 12):
            probes[f'native_r{radius}'] = np.array([alternate_rows[s['source_id']][f'native_r{radius}_xy'] for s in frame['sources']])
        detections = [np.array([[d['x'], d['y']] for d in reports[a]['report']['detections']]) for a in ARMS]
        masks = geometry_groups(xy, detections, width, height, settings['excluded_detection_radius_px'])
        projected = {}
        try:
            for a in ARMS:
                camera = reports[a]['camera']
                rotated = worlds @ np.asarray(camera['world_to_camera']).T
                radius2 = np.sum((rotated[:, :2] / rotated[:, 2, None])**2, axis=1)
                if camera['k1'] < 0 and np.any(radius2 >= -1/(3*camera['k1'])):
                    raise ValueError('reviewed source outside monotone camera domain')
                projected[a] = project(camera, radec)
        except ValueError as error:
            results.append(dict(id=rid, status='invalid_projection', reason=str(error)))
            continue
        measures = {}
        for centroid_name, positions in probes.items():
            errors = {a:np.linalg.norm(projected[a]-positions, axis=1) for a in ARMS}
            groups = {}
            for name, mask in masks.items():
                groups[name] = dict(count=int(sum(mask)))
                if mask.any():
                    groups[name].update({a:stats(errors[a][mask]) for a in ARMS})
                    groups[name]['rms_change_px'] = groups[name]['centered']['rms_px']-groups[name]['control']['rms_px']
            measures[centroid_name] = groups
        results.append(dict(id=rid, status='measured', independent_ground_truth=False,
            groups=measures,
            sources=[dict(source_id=s['source_id'], hyg_id=s['hyg_id'], xy=s['xy'],
                outside_current_inputs=bool(masks['outside_current_inputs'][i]),
                predicted_xy={a:projected[a][i].tolist() for a in ARMS}) for i,s in enumerate(frame['sources'])]))
    return results


def summarize(out):
    protocol = validate(out)
    selected = records()
    baseline = {r['id']:r for r in json.loads((MANIFEST.parent / 'robustness-results.json').read_text())['frames']}
    arms, artifacts = {}, {}
    for arm in ARMS:
        arms[arm], artifacts[arm] = read_arm(out, protocol, arm)
    frames = [dict(id=r['id'], track=r['track'], negative=r['research'].get('negative', False),
                   source_sha256=r['clean_sha256'], published_status=baseline[r['id']]['starglyph'].get('solve_status'),
                   **{a:arms[a][r['id']] for a in ARMS}) for r in selected]
    solved = lambda r,a: r[a].get('solve_status') == 'solved'
    gained = [r['id'] for r in frames if solved(r,'centered') and not solved(r,'control')]
    lost = [r['id'] for r in frames if solved(r,'control') and not solved(r,'centered')]
    geometry = reviewed_geometry(artifacts, protocol['geometry'])
    regressions, improvements = [], []
    for frame in geometry:
        if frame['status'] != 'measured':
            regressions.append(dict(id=frame['id'], reason=frame['status']))
            continue
        for centroid, groups in frame['groups'].items():
            for name, group in groups.items():
                if group['count'] and group['rms_change_px'] > protocol['geometry']['max_group_rms_regression_px']:
                    regressions.append(dict(id=frame['id'], centroid=centroid, group=name, rms_change_px=group['rms_change_px']))
        group = frame['groups']['external_centroid']['outside_current_inputs']
        if (group['count'] >= protocol['geometry']['minimum_outside_probes_for_improvement']
                and group['centered']['rms_px'] <= (1-protocol['geometry']['improvement_fraction'])*group['control']['rms_px']):
            improvements.append(frame['id'])
    negative_accepts = {a:[r['id'] for r in frames if r['negative'] and solved(r,a)] for a in ARMS}
    process_failures = {a:[r['id'] for r in frames if r[a]['status'] not in ('completed','not_attempted')] for a in ARMS}
    totals = {track:dict(count=sum(r['track']==track for r in frames), **{
        a:sum(r['track']==track and solved(r,a) for r in frames) for a in ARMS}) for track in ('solver','stress','scene')}
    write_json(out / 'results.json', dict(iteration=18, split='development', protocol_sha256=digest(out/'protocol.json'),
        totals=totals, gained=gained, lost=lost, negative_accepts=negative_accepts, process_failures=process_failures,
        baseline_status_mismatches=[r['id'] for r in frames if r['control'].get('solve_status') != r['published_status']],
        timing={a:dict(total_s=sum(r[a].get('elapsed_s',0) for r in frames),
                       median_s=statistics.median(r[a]['elapsed_s'] for r in frames if 'elapsed_s' in r[a]),
                       attempt_timeouts=sum(r[a]['attempt_timeouts'] for r in frames)) for a in ARMS},
        geometry_regressions=regressions, geometry_improvements=improvements,
        development_checks=dict(retain_control_successes=not lost,
            negative_rejections_preserved=not negative_accepts['centered'],
            no_additional_process_failures=not (set(process_failures['centered'])-set(process_failures['control'])),
            reviewed_geometry_no_regressions=not regressions),
        new_solves_requiring_review=gained, automatic_improvement_confirmed=False,
        frames=frames, reviewed_geometry=geometry,
        limitations=['external WCS remain pending, not independent accepted ground truth',
                     '117 reviewed identities remain conditional; other field areas are not certified',
                     'new solves need separate field review before any promotion']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.out_dir.resolve())


if __name__ == '__main__':
    main()
