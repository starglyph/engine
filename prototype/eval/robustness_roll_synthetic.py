#!/usr/bin/env python3
"""Preregistered synthetic comparison of circular mean and wrapped median roll."""
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics

from collection_run import ROOT, digest, write_json
from robustness_candidate_trace import command

MODULES = ['roll_synthetic.rs', 'roll_synthetic_fixture.rs', 'robust_roll.rs', 'centre_transfer_fixture.rs']
PLAN = dict(sizes=[[1600,1200],[1989,1498]], fovs_deg=[22.,85.],
    skies=[[30.,20.,15.],[210.,-35.,179.8]], supports=['full','upper'], pairs=[13,20],
    sigmas_px=[.25,1.], seeds=list(range(8)), bad_angles_deg=[1.,5.])
CRITERIA = dict(noiseless_max_px=.01, clean_group_rms_ratio_max=1.10,
    clean_case_rms_increase_max_px=.5, contaminated_group_median_rms_ratio_max=.75)


def cases():
    rows=[]
    for size,fov,sky,support,count in itertools.product(PLAN['sizes'],PLAN['fovs_deg'],
            PLAN['skies'],PLAN['supports'],PLAN['pairs']):
        base=dict(size=size,fov_deg=fov,sky=sky,support=support,pairs=count)
        rows.append(dict(base,sigma_px=0.,seed=0,bad_count=0,bad_angle_deg=0.))
        for sigma,seed in itertools.product(PLAN['sigmas_px'],PLAN['seeds']):
            rows.append(dict(base,sigma_px=sigma,seed=seed,bad_count=0,bad_angle_deg=0.))
            for angle,bad_count in itertools.product(PLAN['bad_angles_deg'],[1,count//4]):
                rows.append(dict(base,sigma_px=sigma,seed=seed,bad_count=bad_count,bad_angle_deg=angle))
    return [dict(row,id=f'case-{i:04d}') for i,row in enumerate(rows)]


def candidate_adapter(source):
    start=source.index('fn pose_from_solution(')
    end=source.index('/// Project verification stars',start)
    function=source[start:end]
    replacements={
        'fn pose_from_solution(':'pub(super) fn median_pose_from_solution(',
        '    let mut sum_sin = 0.0;\n    let mut sum_cos = 0.0;':'    let mut rolls = Vec::new();',
        '        sum_sin += roll.sin();\n        sum_cos += roll.cos();':'        rolls.push(roll);',
        'roll_deg: sum_sin.atan2(sum_cos).to_degrees(),':'roll_deg: robust_roll::estimate(&rolls)?.to_degrees(),',
    }
    for before,after in replacements.items():
        if function.count(before)!=1:
            raise ValueError('unexpected adapter source')
        function=function.replace(before,after)
    return '// Generated from frozen production adapter; only roll aggregation differs.\nuse super::*;\n'+function.rstrip()+'\n'


def prepare(out):
    if out.exists():
        raise ValueError('fresh output directory required')
    pipeline=out/'pipeline'
    pipeline.mkdir(parents=True)
    for name in ['starglyph-core','simulator-core']:
        shutil.copytree(ROOT/'prototype/crates'/name,pipeline/'crates'/name)
    manifest=(ROOT/'prototype/Cargo.toml').read_text()
    manifest,count=re.subn(r'members = \[.*?\]',
        'members = ["crates/starglyph-core", "crates/simulator-core"]',manifest,count=1,flags=re.S)
    if count!=1:
        raise ValueError('unexpected workspace')
    (pipeline/'Cargo.toml').write_text(manifest)
    shutil.copyfile(ROOT/'prototype/Cargo.lock',pipeline/'Cargo.lock')
    source=ROOT/'prototype/crates/starglyph-core/src/solve.rs'
    target=pipeline/'crates/starglyph-core/src/solve.rs'
    target.write_text(source.read_text()+'\n#[cfg(test)]\nmod roll_synthetic;\n')
    for name in MODULES:
        shutil.copyfile(Path(__file__).with_name(name),target.parent/'solve'/name)
    (target.parent/'solve/roll_candidate_adapter.rs').write_text(candidate_adapter(source.read_text()))
    # Existing core regression tests resolve committed data through this layout.
    (out/'data').symlink_to(ROOT/'data',target_is_directory=True)
    (out/'prototype').symlink_to(pipeline,target_is_directory=True)
    (pipeline/'artifacts').mkdir()
    (pipeline/'artifacts/cache').symlink_to(ROOT/'prototype/artifacts/cache',target_is_directory=True)
    command(out,'lock-resolution',['cargo','update','--offline','--workspace','--manifest-path',str(pipeline/'Cargo.toml')])
    write_json(out/'cases.json',cases())
    files=[Path(__file__).resolve(),Path(__file__).with_name('collection_run.py'),
        Path(__file__).with_name('robustness_candidate_trace.py'),out/'cases.json',
        *[Path(__file__).with_name(name) for name in MODULES]]
    for base in [ROOT/'prototype',pipeline]:
        files.extend([base/'Cargo.toml',base/'Cargo.lock'])
        for name in ['starglyph-core','simulator-core']:
            files.extend((base/'crates'/name/'src').rglob('*.rs'))
            files.append(base/'crates'/name/'Cargo.toml')
    files.extend(ROOT/'data/samples/sky-samples'/name for name in
        ['manifest.json','robustness-results.json','collection-wcs-review.json'])
    write_json(out/'protocol.json',dict(iteration=23,split='synthetic',holdout=False,real_images=False,
        plan=PLAN,criteria=CRITERIA,case_count=len(cases()),probes=63,fit_or_search=False,
        candidate='median of pair roll angles unwrapped about their circular mean; even counts average middle two',
        corruption='rotate catalogue identity around optical axis by signed 1 or 5 degrees for 1 or floor(n/4) pairs; detections unchanged',
        randomness='SplitMix64 seed 20261005+seed; Box-Muller Cartesian noise then Fisher-Yates identity selection; sign alternates by seed',
        geometry='independent existing tetra3 TAN fixtures; fixed known centre alignment +0.5 px; no distortion or boresight/FOV error',
        progression='development only if all synthetic gates pass; no tuning after failed gates; never promote from synthetic alone',
        wall_limit_s=120,build_wall_limit_s=600,
        hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(set(files))}))


def validate(out):
    p=json.loads((out/'protocol.json').read_text())
    if p['split']!='synthetic' or p['holdout'] or p['real_images']:
        raise ValueError('synthetic only')
    if p['plan']!=PLAN or p['criteria']!=CRITERIA or p['case_count']!=len(cases()):
        raise ValueError('fixed plan or criteria changed')
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
    args=['cargo','test','--offline','--locked','--release','--manifest-path',str(pipeline/'Cargo.toml'),
        '-p','starglyph-core','--lib','--no-run','--message-format=json']
    command(out,'build',args,env,p['build_wall_limit_s'])
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
    command(out,'run',[str(binary),'solve::roll_synthetic','--include-ignored','--nocapture'],env,p['wall_limit_s'])


def evaluate(rows, criteria):
    clean=[r for r in rows if r['bad_count']==0 and r['sigma_px']>0]
    ideal=[r for r in rows if r['sigma_px']==0]
    groups=[]
    for noise,support,count in itertools.product(PLAN['sigmas_px'],PLAN['supports'],PLAN['pairs']):
        subset=[r for r in clean if (r['sigma_px'],r['support'],r['pairs'])==(noise,support,count)]
        rms={arm:math.sqrt(statistics.mean(r['arms'][arm]['rms_px']**2 for r in subset)) for arm in ['control','median']}
        groups.append(dict(sigma_px=noise,support=support,pairs=count,count=len(subset),rms_px=rms,
            ratio=rms['median']/rms['control']))
    bad_groups=[]
    for angle,fraction in itertools.product(PLAN['bad_angles_deg'],['single','quarter']):
        subset=[r for r in rows if r['bad_angle_deg']==angle and
            r['bad_count']==(1 if fraction=='single' else r['pairs']//4)]
        rms={arm:statistics.median(r['arms'][arm]['rms_px'] for r in subset) for arm in ['control','median']}
        bad_groups.append(dict(bad_angle_deg=angle,corruption=fraction,count=len(subset),median_rms_px=rms,
            ratio=rms['median']/rms['control']))
    worst=max(clean,key=lambda r:r['arms']['median']['rms_px']-r['arms']['control']['rms_px'])
    increase=worst['arms']['median']['rms_px']-worst['arms']['control']['rms_px']
    maximum=max(r['arms'][arm]['max_px'] for r in ideal for arm in ['control','median'])
    gates=dict(noiseless=maximum<=criteria['noiseless_max_px'],
        clean_groups=all(g['ratio']<=criteria['clean_group_rms_ratio_max'] for g in groups),
        clean_individual=increase<=criteria['clean_case_rms_increase_max_px'],
        contaminated_groups=all(g['ratio']<=criteria['contaminated_group_median_rms_ratio_max'] for g in bad_groups))
    return dict(gates=gates,eligible_for_development=all(gates.values()),clean_groups=groups,
        contaminated_groups=bad_groups,noiseless_max_px=maximum,
        worst_clean_case_id=worst['id'],worst_clean_rms_increase_px=increase)


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
        if set(row['arms'])!={'control','median'}:
            raise ValueError('missing comparison arm')
        for arm in row['arms'].values():
            if any(not math.isfinite(arm[k]) for k in ['roll_deg','rms_px','max_px','elapsed_us']):
                raise ValueError('nonfinite result')
        rows.append(dict(meta,**{k:v for k,v in row.items() if k!='id'}))
    summary=evaluate(rows,p['criteria'])
    write_json(out/'results.json',dict(iteration=23,split='synthetic',holdout=False,case_count=len(rows),
        protocol_sha256=digest(out/'protocol.json'),raw_sha256=digest(out/'raw.json'),
        run=json.loads((out/'run.json').read_text()),**summary))
    # One compact JSON row per case; all parameters, corruption identities and metrics retained.
    (out/'cases-results.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n' for r in rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','summarize','validate'])
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args()
    globals()[args.stage](args.out_dir.resolve())


if __name__=='__main__':
    main()
