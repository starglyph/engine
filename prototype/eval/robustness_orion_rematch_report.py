#!/usr/bin/env python3
"""Audit rematch identities and common unused geometry, never acceptance."""
import argparse
import csv
import gzip
from pathlib import Path

import numpy as np

from collection_run import digest, write_json
from compare_local_wcs import project, stats
from robustness_orion_fit import METHODS, RID, WIDTH, HEIGHT, previous, read
from robustness_orion_rematch import validate
from robustness_reference_audit import CATALOG
from robustness_reference_coverage import extent
from robustness_refinement_crossover_report import close
from robustness_refinement_report import lift


def unused_sources(sources, fitted):
    """Common assessment excludes identity OR physical reuse in either arm."""
    used = {p['hyg_id'] for p in fitted}
    xy = np.array([p['xy'] for p in fitted])
    return [s['id'] for s in sources if s['hyg_id'] not in used and all(
        np.linalg.norm(xy-s['coordinates'][method], axis=1).min() > 12 for method in METHODS)]


def pair_summary(pairs):
    return {label: extent([dict(x=p['xy'][0], y=p['xy'][1]) for p in pairs
        if p['reviewed_identity_agrees'] is value], WIDTH, HEIGHT)
        for label, value in [('agree', True), ('disagree', False), ('unknown', None)]}


def verdict(before, after):
    a, b = before['pair_summary'], after['pair_summary']
    coverage = (b['agree']['count'] >= a['agree']['count'] and
                b['agree']['width_fraction'] is not None and a['agree']['width_fraction'] is not None and
                b['agree']['width_fraction'] > a['agree']['width_fraction'] and
                b['disagree']['count'] <= a['disagree']['count'])
    methods = {}
    for method in METHODS:
        old = before['geometry'][method]['groups']; new = after['geometry'][method]['groups']
        if old.keys() != new.keys() or any(old[k]['count'] != new[k]['count'] for k in old):
            raise ValueError('assessment membership changed')
        key = 'outside_common/all_reviewed'
        enough = old[key]['count'] >= 8
        ratio = new[key]['rms_px']/old[key]['rms_px'] if enough and old[key]['rms_px'] > 0 else None
        regressions = {k: new[k]['rms_px']-v['rms_px'] for k, v in old.items()
                       if v['count'] and new[k]['rms_px']-v['rms_px'] > .5}
        methods[method] = dict(count=old[key]['count'], rms_ratio=ratio, regressions=regressions,
                               passed=ratio is not None and ratio <= .9 and not regressions)
    return dict(correct_pair_coverage_passed=coverage, geometry=methods,
                passed=coverage and all(m['passed'] for m in methods.values()))


def summarize(out):
    protocol = validate(out); raw = read(out/'replay.json'); saved = read(previous('41-trace'))
    cases = raw['cases']
    if [c['pose_arm'] for c in cases] != protocol['arms'] or any(c['match_arm'] != 'control' for c in cases):
        raise ValueError('two frozen cases required')
    replay_error = close(cases[0]['steps'], saved['steps'])
    close(cases[0]['verification'], saved['steps'][-1]['verification'])
    close(cases[0]['steps'][0], cases[1]['steps'][0], tolerance=0)
    sources = read(previous('40-results'))['sources']; ids = [s['id'] for s in sources]
    with gzip.open(CATALOG, 'rt') as f:
        stars = [p for p in csv.DictReader(f) if p['mag'] and float(p['mag']) <= 6.8]
    ra, dec = np.array([[float(p['rarad']), float(p['decrad'])] for p in stars]).T
    units = np.column_stack((np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)))
    identity_cache = {}
    def bind(rows, camera):
        result = []
        for m in rows:
            key = tuple(m['world'])
            if key not in identity_cache:
                indices = np.flatnonzero(np.linalg.norm(units-m['world'], axis=1) < 1e-12)
                if len(indices) != 1:
                    raise ValueError('catalogue identity not unique')
                identity_cache[key] = stars[int(indices[0])]
            star = identity_cache[key]; xy = lift(m['xy'], camera, WIDTH, HEIGHT)
            distances = np.linalg.norm(np.array([s['coordinates']['external_centroid'] for s in sources])-xy, axis=1)
            i = int(distances.argmin()); s = sources[i]
            result.append(dict(hyg_id=int(star['id']), mag=float(star['mag']), xy=xy.tolist(),
                nearest_reviewed_id=s['id'], reviewed_distance_px=float(distances[i]),
                reviewed_identity_agrees=(int(star['id']) == s['hyg_id']) if distances[i] < 12 else None))
        return result
    rows = []; fitted = []
    for case in cases:
        stages = []
        for step in case['steps']:
            v = step.get('input', step.get('verification'))
            pairs = bind(step['fit_matches'] if v is None else v['matches'], step['camera'])
            is_fit = step['stage'] == 'first_lm' or (step['stage'] == 'rematch' and step['refined'])
            if is_fit:
                fitted.extend(pairs)
            stages.append(dict(stage=step['stage'], radius_working_px=step.get('radius_px'), camera=step['camera'],
                used_in_fit=is_fit, pairs=pairs, pair_summary=pair_summary(pairs)))
        pairs = bind(case['verification']['matches'], case['camera'])
        stages.append(dict(stage='final_full_catalog', radius_working_px=2.5, camera=case['camera'],
            used_in_fit=False, pairs=pairs, pair_summary=pair_summary(pairs)))
        rows.append(dict(arm=case['pose_arm'], rematch_catalog_count=case['rematch_catalog_count'],
            elapsed_ms=case['elapsed_ms'], lm_calls=sum(s['used_in_fit'] for s in stages), stages=stages))
    unused = unused_sources(sources, fitted)
    groups = {'all47/all_reviewed': ids, 'outside_common/all_reviewed': unused}
    for group in sorted({g for s in sources for g in s['groups']}):
        members = [s['id'] for s in sources if group in s['groups']]
        groups['all47/'+group] = members
        groups['outside_common/'+group] = [sid for sid in members if sid in unused]
    for row in rows:
        for stage in row['stages']:
            pred = lift(project(stage['camera'], [s['radec'] for s in sources]), stage['camera'], WIDTH, HEIGHT)
            stage['predicted_xy_original'] = pred.tolist(); stage['geometry'] = {}
            for method in METHODS:
                residual = pred-[s['coordinates'][method] for s in sources]
                errors = np.linalg.norm(residual, axis=1)
                stage['geometry'][method] = dict(residual_xy_px=residual.tolist(), groups={g:
                    stats(errors[[ids.index(i) for i in members]]) if members else dict(count=0)
                    for g, members in groups.items()})
    write_json(out/'results.json', dict(iteration=54, id=RID, split='development', holdout=False,
        protocol_sha256=digest(out/'protocol.json'), control_steps_max_delta=replay_error,
        assessment_groups=groups, outside_common_ids=unused, cases=rows,
        descriptive_transfer=verdict(rows[0]['stages'][-1], rows[1]['stages'][-1]),
        sources=[dict(id=s['id'], hyg_id=s['hyg_id'], name=s['name']) for s in sources],
        run=read(out/'run.json'), new_solver_calls=0, new_wcs_calls=0, production_changed=False,
        independent_ground_truth=False, automatic_improvement_confirmed=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out-dir', type=Path, required=True)
    summarize(p.parse_args().out_dir.resolve())
