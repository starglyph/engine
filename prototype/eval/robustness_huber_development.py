#!/usr/bin/env python3
"""Frozen development A/B of the synthetic-qualified radius-aware Huber roll."""
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import sys

from collection_run import ROOT, bounded, digest, write_json
from robustness_candidate_trace import command
from robustness_centroid_origin import prepare as prepare_workspace, records, MANIFEST
from robustness_huber_roll import candidate_adapter, validate as validate_synthetic

ARMS=('control','huber')
GEOMETRY=dict(excluded_detection_radius_px=12,max_group_rms_regression_px=.5,
              improvement_fraction=.1,minimum_outside_probes_for_improvement=8)


def patch_source(source):
    start=source.index('fn pose_from_solution(')
    end=source.index('/// Project verification stars',start)
    adapter=candidate_adapter(source)
    adapter=adapter[adapter.index('pub(super) fn huber_pose_from_solution('):]
    adapter=adapter.replace('pub(super) fn huber_pose_from_solution(', 'fn pose_from_solution(',1)
    return source[:start]+adapter+'\n'+source[end:]+'\nmod robust_roll;\nmod huber_roll;\n'


def prepare(out, synthetic):
    validate_synthetic(synthetic)
    results=json.loads((synthetic/'results.json').read_text())
    if not results['eligible_for_development']:
        raise ValueError('synthetic gates did not pass')
    prepare_workspace(out)
    pipeline=out/'pipeline'
    original=ROOT/'prototype/crates/starglyph-core/src/solve.rs'
    target=pipeline/'crates/starglyph-core/src/solve.rs'
    (target.parent/'solve/centroid_origin.rs').unlink()
    (out/'binaries').mkdir()
    env={**os.environ,'CARGO_TARGET_DIR':str(pipeline/'target')}
    for arm in ARMS:
        target.write_text(original.read_text() if arm=='control' else patch_source(original.read_text()))
        if arm=='huber':
            for name in ['robust_roll.rs','huber_roll.rs']:
                shutil.copyfile(Path(__file__).with_name(name),target.parent/'solve'/name)
        shutil.copyfile(target,out/f'solve-{arm}.rs')
        command(out,'build-'+arm,['cargo','build','--offline','--locked','--release','--manifest-path',
            str(pipeline/'Cargo.toml'),'-p','starglyph-cli'],env,600)
        shutil.copy2(pipeline/'target/release/starglyph',out/'binaries'/arm)
    selected=records()
    files=[Path(__file__).resolve(),Path(__file__).with_name('robustness_huber_roll.py'),
        synthetic/'protocol.json',synthetic/'results.json',
        out/'solve-control.rs',out/'solve-huber.rs',*[out/'binaries'/a for a in ARMS],
        MANIFEST,MANIFEST.parent/'robustness-results.json',MANIFEST.parent/'collection-wcs-review.json',
        ROOT/'data/catalogs/hyg_v42.csv.gz']
    files.extend(Path(__file__).with_name(name) for name in ['huber_roll.rs','robust_roll.rs',
        'robustness_centroid_origin.py','robustness_centroid_origin_report.py','collection_run.py',
        'compare_local_wcs.py','smartphone_gate.py','sky_fill_gate.py','robustness_candidate_trace.py',
        'baseline-smartphone.json','baseline-smartphone-sky-fill.json','baseline-ci.json'])
    for base in [ROOT/'prototype',pipeline]:
        files.extend([base/'Cargo.toml',base/'Cargo.lock'])
        for name in ['starglyph-core','starglyph-cli','simulator-core']:
            files.extend((base/'crates'/name/'src').rglob('*.rs'))
            files.append(base/'crates'/name/'Cargo.toml')
    for r in selected:
        files.append(MANIFEST.parent/r['file'])
        reference=ROOT/f"prototype/artifacts/robustness-baseline/wcs-run/{r['id']}/reference.json"
        if reference.exists():
            files.append(reference)
            ref=json.loads(reference.read_text())
            if ref['status']=='solved_candidate':
                files.extend(ROOT/'prototype'/ref[k] for k in ['wcs_file','correspondences_file'])
    phone=ROOT/'data/input/smartphone'
    files.extend([phone/'manifest.json',phone/'sky-masks.json'])
    files.extend(phone/r['file'] for r in json.loads((phone/'manifest.json').read_text()))
    files.extend(ROOT/'docs/experiments'/f'robustness-iteration-{i}-geometry.json' for i in [5,13])
    files.extend((ROOT/'prototype/artifacts/cache').glob('*.bin'))
    write_json(out/'protocol.json',dict(iteration=24,split='development',holdout=False,
        ids=[r['id'] for r in selected],wall_limit_s=120,batch_size=6,
        binaries={a:str((out/'binaries'/a).relative_to(ROOT)) for a in ARMS},
        hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(set(files))},geometry=GEOMETRY,
        variant='only replace roll aggregation with iteration24 synthetic-qualified Huber; original centroid convention retained',
        order='alternate first arm in batches of six, one CLI at a time, same cached catalogues',
        criteria=['retain all control successes and all eight development negative rejections',
                  'no additional process errors or wall timeouts',
                  'no conditional reviewed group RMS regression over 0.5px',
                  'retain smartphone/sky-fill and core/CLI/extended-dense/CI regression gates',
                  'improvement needs a newly independently reviewed solve or >=10% outside-input RMS improvement with >=8 probes',
                  'pending WCS and synthetic success cannot certify new solves; no holdout tuning']))


def validate(out):
    p=json.loads((out/'protocol.json').read_text())
    if p['split']!='development' or p['holdout'] or p['ids']!=[r['id'] for r in records()]:
        raise ValueError('frozen development selection required')
    for name,sha in p['hashes'].items():
        if digest(ROOT/name)!=sha:
            raise ValueError('changed frozen input: '+name)
    return p


def run(out):
    import fcntl
    p=validate(out)
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        batch=0
        while True:
            complete=True
            for arm in ARMS if batch%2==0 else ARMS[::-1]:
                folder=out/'development'/arm
                prior=json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else None
                if prior and prior['completed']==len(p['ids']):
                    continue
                complete=False
                args=[sys.executable,str(ROOT/'prototype/eval/collection_run.py'),'starglyph',
                    '--split','development','--binary',str(ROOT/p['binaries'][arm]),
                    '--out-dir',str(folder),'--batch-size',str(p['batch_size'])]
                if prior:
                    args.append('--resume')
                print(f'batch {batch} {arm}',flush=True)
                command(out,f'batch-{batch}-{arm}',args,{**os.environ,'STARGLYPH_SOLVE_DEBUG':'1'},800)
            if complete:
                break
            batch+=1
    validate(out)


def gates(out):
    p=validate(out)
    binary=ROOT/p['binaries']['huber']
    cargo=['cargo','test','--offline','--locked','--release','--manifest-path',str(out/'pipeline/Cargo.toml')]
    checks=[('rust-tests',cargo+['-p','starglyph-core','-p','starglyph-cli']),
        ('extended-dense',cargo+['-p','starglyph-core','--test','extended_dense','--','--ignored']),
        ('eval-ci',[str(binary),'eval','--manifest','../data/samples/sky-samples/manifest.json',
            '--ids','tetra3_alt40,tetra3_alt60','--catalog','../data/catalogs/hyg_v42.csv.gz',
            '--out-dir',str(out/'gate-ci'),'--baseline','eval/baseline-ci.json']),
        ('smartphone',[sys.executable,'eval/smartphone_gate.py','--binary',str(binary),'--out-dir',str(out/'gate-smartphone')]),
        ('sky-fill',[sys.executable,'eval/sky_fill_gate.py','--binary',str(binary),'--out-dir',str(out/'gate-sky-fill')])]
    for name,args in checks:
        path=out/(name+'.json')
        if path.exists():
            continue
        result=bounded(args,out/(name+'.log'),600)
        write_json(path,result)
        print(name,result['status'],flush=True)


def rename_arm(value):
    if isinstance(value,dict):
        return {('huber' if k=='centered' else k):rename_arm(v) for k,v in value.items()}
    if isinstance(value,list):
        return [rename_arm(v) for v in value]
    return value


def summarize(out):
    from robustness_centroid_origin_report import read_arm, reviewed_geometry
    p=validate(out)
    arms,artifacts={},{}
    for arm in ARMS:
        arms[arm],artifacts[arm]=read_arm(out,p,arm)
    baseline={r['id']:r for r in json.loads((MANIFEST.parent/'robustness-results.json').read_text())['frames']}
    frames=[dict(id=r['id'],track=r['track'],negative=r['research'].get('negative',False),
        published_status=baseline[r['id']]['starglyph'].get('solve_status'),source_sha256=r['clean_sha256'],
        **{a:arms[a][r['id']] for a in ARMS}) for r in records()]
    solved=lambda r,a:r[a].get('solve_status')=='solved'
    gained=[r['id'] for r in frames if solved(r,'huber') and not solved(r,'control')]
    lost=[r['id'] for r in frames if solved(r,'control') and not solved(r,'huber')]
    geometry=rename_arm(reviewed_geometry(dict(control=artifacts['control'],centered=artifacts['huber']),p['geometry']))
    regressions=[]
    improvements=[]
    for frame in geometry:
        if frame['status']!='measured':
            regressions.append(dict(id=frame['id'],reason=frame['status']))
            continue
        for centroid,groups in frame['groups'].items():
            for name,g in groups.items():
                if g['count'] and g['rms_change_px']>GEOMETRY['max_group_rms_regression_px']:
                    regressions.append(dict(id=frame['id'],centroid=centroid,group=name,rms_change_px=g['rms_change_px']))
        g=frame['groups']['external_centroid']['outside_current_inputs']
        if g['count']>=GEOMETRY['minimum_outside_probes_for_improvement'] and g['huber']['rms_px']<=(1-GEOMETRY['improvement_fraction'])*g['control']['rms_px']:
            improvements.append(frame['id'])
    negative={a:[r['id'] for r in frames if r['negative'] and solved(r,a)] for a in ARMS}
    errors={a:[r['id'] for r in frames if r[a]['status'] not in ('completed','not_attempted')] for a in ARMS}
    gates={name:json.loads((out/(name+'.json')).read_text()) for name in
        ['rust-tests','extended-dense','eval-ci','smartphone','sky-fill'] if (out/(name+'.json')).exists()}
    write_json(out/'results.json',dict(iteration=24,split='development',holdout=False,
        protocol_sha256=digest(out/'protocol.json'),automatic_improvement_confirmed=False,
        totals={track:dict(count=sum(r['track']==track for r in frames),
            **{a:sum(r['track']==track and solved(r,a) for r in frames) for a in ARMS}) for track in ['solver','stress','scene']},
        gained=gained,lost=lost,negative_accepts=negative,process_failures=errors,
        baseline_status_mismatches=[r['id'] for r in frames if r['control'].get('solve_status')!=r['published_status']],
        timing={a:dict(total_s=sum(r[a].get('elapsed_s',0) for r in frames),
            median_s=statistics.median(r[a]['elapsed_s'] for r in frames if 'elapsed_s' in r[a]),
            attempt_timeouts=sum(r[a]['attempt_timeouts'] for r in frames)) for a in ARMS},
        geometry_regressions=regressions,geometry_improvements=improvements,
        development_checks=dict(retain_control_successes=not lost,negative_rejections_preserved=not negative['huber'],
            no_additional_process_failures=not(set(errors['huber'])-set(errors['control'])),reviewed_geometry_no_regressions=not regressions),
        regression_gates=gates,new_solves_requiring_review=gained,frames=frames,reviewed_geometry=geometry,
        limitations=['external WCS and reviewed identities remain conditional, not accepted independent ground truth',
                     'new solved statuses require separate field review; no default promotion from counts']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','gates','summarize','validate'])
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--synthetic-dir',type=Path)
    args=parser.parse_args()
    os.chdir(ROOT/'prototype')
    out=args.out_dir.resolve()
    if args.stage=='prepare':
        if args.synthetic_dir is None:
            parser.error('prepare requires --synthetic-dir')
        prepare(out,args.synthetic_dir.resolve())
    else:
        globals()[args.stage](out)


if __name__=='__main__':
    main()
