#!/usr/bin/env python3
"""One frozen radius-aware Huber roll candidate; observed and fresh noise cohorts."""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import shutil

from collection_run import ROOT, digest, write_json
from robustness_candidate_trace import command
import robustness_roll_synthetic as previous

COHORTS = dict(observed=list(range(8)),fresh=list(range(1000,1032)))
PARAMETERS = dict(huber_c=1.345,mad_to_sigma=1.4826,scale_floor_px=1e-6,
    max_steps=8,step_tolerance_rad=1e-12,
    scale='fixed MAD of initial radius-times-angular residuals, centered on their median',
    initialization='iteration23 wrapped angular median',
    weights='radius squared times Huber residual weight; normalize radii by maximum',
    radius='observed distance from geometric image center')


def cases():
    templates=[c for c in previous.cases() if c['seed']==0]
    rows=[]
    for cohort,seeds in COHORTS.items():
        for template in templates:
            for seed in seeds[:1] if template['sigma_px']==0 else seeds:
                rows.append(dict(template,seed=seed,cohort=cohort,id=f'{cohort}-{len(rows):05d}'))
    return rows


def candidate_adapter(source):
    candidate=previous.candidate_adapter(source)
    replacements={
        'median_pose_from_solution':'huber_pose_from_solution',
        'rolls.push(roll);':'rolls.push((roll, (det.x - cx).hypot(cy - det.y)));',
        'robust_roll::estimate(&rolls)':'huber_roll::estimate(&rolls)',
    }
    for before,after in replacements.items():
        if candidate.count(before)!=1:
            raise ValueError('unexpected generated adapter')
        candidate=candidate.replace(before,after)
    return candidate


def prepare(out):
    # Reuse isolated workspace preparation; no iteration23 experiment is run.
    previous.prepare(out)
    previous_protocol=json.loads((out/'protocol.json').read_text())
    pipeline=out/'pipeline'
    source=ROOT/'prototype/crates/starglyph-core/src/solve.rs'
    target=pipeline/'crates/starglyph-core/src/solve.rs'
    target.write_text(source.read_text()+'\n#[cfg(test)]\nmod roll_huber_synthetic;\n')
    module=Path(__file__).with_name('huber_roll.rs')
    shutil.copyfile(module,target.parent/'solve/huber_roll.rs')
    harness=Path(__file__).with_name('roll_synthetic.rs').read_text()
    if harness.count('mod robust_roll;')!=1:
        raise ValueError('unexpected synthetic harness')
    harness=harness.replace('mod robust_roll;','mod robust_roll;\n#[path = "huber_roll.rs"]\nmod huber_roll;')
    harness=harness.replace('"median"','"huber"').replace('adapter::median_pose_from_solution','adapter::huber_pose_from_solution')
    (target.parent/'solve/roll_huber_synthetic.rs').write_text(harness)
    (target.parent/'solve/roll_candidate_adapter.rs').write_text(candidate_adapter(source.read_text()))
    write_json(out/'cases.json',cases())
    files=[ROOT/name for name in previous_protocol['hashes']]
    files.extend([Path(__file__).resolve(),module,target.parent/'solve/huber_roll.rs',
        target.parent/'solve/roll_huber_synthetic.rs',
        ROOT/'docs/experiments/robustness-iteration-23-cases-results.jsonl'])
    write_json(out/'protocol.json',dict(iteration=24,split='synthetic',holdout=False,real_images=False,
        plan=previous.PLAN,cohorts=COHORTS,parameters=PARAMETERS,criteria=previous.CRITERIA,
        case_count=len(cases()),probes=63,fit_or_search=False,
        candidate='Huber location of radius-weighted angular residuals, with frozen robust scale and bounded IRLS',
        progression='all gates must pass separately on observed and fresh cohorts before development; no parameter search',
        input_generator='unchanged iteration23 TAN fixtures, Gaussian noise and corrupted catalogue pairs',
        observed_control_tolerance_px=1e-12,wall_limit_s=120,build_wall_limit_s=600,
        hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(set(files))}))


def validate(out):
    p=json.loads((out/'protocol.json').read_text())
    if p['split']!='synthetic' or p['holdout'] or p['real_images']:
        raise ValueError('synthetic only')
    expected=dict(plan=previous.PLAN,cohorts=COHORTS,parameters=PARAMETERS,
                  criteria=previous.CRITERIA,case_count=len(cases()))
    if any(p[k]!=v for k,v in expected.items()):
        raise ValueError('frozen plan, parameters or criteria changed')
    for name,sha in p['hashes'].items():
        if digest(ROOT/name)!=sha:
            raise ValueError('changed frozen input: '+name)
    return p


def run(out):
    p=validate(out)
    if (out/'run.json').exists() or (out/'raw.json').exists():
        raise ValueError('run already attempted')
    pipeline=out/'pipeline'
    env={**os.environ,'CARGO_TARGET_DIR':str(pipeline/'target')}
    command(out,'build',['cargo','test','--offline','--locked','--release','--manifest-path',
        str(pipeline/'Cargo.toml'),'-p','starglyph-core','--lib','--no-run','--message-format=json'],env,p['build_wall_limit_s'])
    artifacts=[]
    for line in (out/'build.log').read_text().splitlines():
        if line.startswith('{'):
            row=json.loads(line)
            if row.get('reason')=='compiler-artifact' and row.get('executable') and row['target']['name']=='starglyph_core':
                artifacts.append(row['executable'])
    if len(artifacts)!=1:
        raise ValueError('expected one core test executable')
    binary=out/'roll-test'
    shutil.copy2(artifacts[0],binary)
    write_json(out/'build-artifact.json',dict(binary=str(binary.relative_to(ROOT)),sha256=digest(binary)))
    env.update(STARGLYPH_ROLL_CASES=str(out/'cases.json'),STARGLYPH_ROLL_OUTPUT=str(out/'raw.json'))
    command(out,'run',[str(binary),'solve::roll_huber_synthetic','--include-ignored','--nocapture'],env,p['wall_limit_s'])


def evaluate(rows):
    # Reuse exactly iteration23's gates, adapting only the estimator's arm name.
    cohorts={}
    for cohort in COHORTS:
        subset=[dict(r,arms=dict(control=r['arms']['control'],median=r['arms']['huber']))
                for r in rows if r['cohort']==cohort]
        result=previous.evaluate(subset,previous.CRITERIA)
        for group in result['clean_groups']:
            group['rms_px']['huber']=group['rms_px'].pop('median')
        for group in result['contaminated_groups']:
            group['median_rms_px']['huber']=group['median_rms_px'].pop('median')
        cohorts[cohort]=result
    return dict(cohorts=cohorts,eligible_for_development=all(r['eligible_for_development'] for r in cohorts.values()))


def case_key(row):
    return json.dumps({k:row[k] for k in ['size','fov_deg','sky','support','pairs','sigma_px',
                                        'seed','bad_count','bad_angle_deg']},sort_keys=True)


def summarize(out):
    p=validate(out)
    raw=json.loads((out/'raw.json').read_text())['cases']
    expected={c['id']:c for c in cases()}
    if len(raw)!=len(expected) or {r['id'] for r in raw}!=set(expected):
        raise ValueError('incomplete or duplicated case matrix')
    if json.loads((out/'run.json').read_text())['exit_code']!=0:
        raise ValueError('run failed')
    rows=[]
    for row in raw:
        meta=expected[row['id']]
        indices=row['corrupted_indices']
        if len(set(indices))!=meta['bad_count'] or any(i<0 or i>=meta['pairs'] for i in indices):
            raise ValueError('invalid corrupted membership')
        if set(row['arms'])!={'control','huber'}:
            raise ValueError('missing comparison arm')
        if any(not math.isfinite(m[k]) for m in row['arms'].values()
               for k in ['roll_deg','rms_px','max_px','elapsed_us']):
            raise ValueError('nonfinite result')
        rows.append(dict(meta,**{k:v for k,v in row.items() if k!='id'}))
    old=[json.loads(line) for line in (ROOT/'docs/experiments/robustness-iteration-23-cases-results.jsonl').read_text().splitlines()]
    lookup={case_key(r):r for r in old}
    maximum=0.
    for row in rows:
        if row['cohort']=='observed':
            original=lookup[case_key(row)]
            if row['corrupted_indices']!=original['corrupted_indices']:
                raise ValueError('observed corruption not reproduced')
            maximum=max(maximum,*(abs(row['arms']['control'][k]-original['arms']['control'][k])
                                  for k in ['roll_deg','rms_px','max_px']))
    if maximum>p['observed_control_tolerance_px']:
        raise ValueError('observed control not reproduced')
    write_json(out/'results.json',dict(iteration=24,split='synthetic',holdout=False,case_count=len(rows),
        protocol_sha256=digest(out/'protocol.json'),raw_sha256=digest(out/'raw.json'),
        run=json.loads((out/'run.json').read_text()),observed_control_max_numeric_difference=maximum,
        **evaluate(rows)))
    data=''.join(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n' for r in rows).encode()
    (out/'cases-results.jsonl.gz').write_bytes(gzip.compress(data,mtime=0))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','summarize','validate'])
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args()
    globals()[args.stage](args.out_dir.resolve())


if __name__=='__main__':
    main()
