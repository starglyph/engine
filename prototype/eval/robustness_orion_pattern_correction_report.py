#!/usr/bin/env python3
"""Publish lossless scalar curves; full five-component ratios remain local."""
import argparse
import copy
import json
from pathlib import Path

from collection_run import ROOT, digest
from robustness_orion_pattern_correction import read, validate


def publish(out):
    protocol=validate(out)
    raw=read(out/'raw.json');results=read(out/'results.json');inp=read(out/'input.json')
    for name in ['run','python-tests','rust-tests','clippy']:
        if read(out/(name+'.json'))['exit_code']!=0:raise ValueError('failed check: '+name)
    if results['protocol_sha256']!=digest(out/'protocol.json'):raise ValueError('result protocol changed')
    compact=copy.deepcopy(raw)
    count=0
    for before,after in zip(raw['cases'],compact['cases']):
        for old,arm in zip(before['arms'],after['arms']):
            arm['curves']=[[r['max_abs_delta'],r['ratios_pass']] for r in old['curves']]
            if len(arm['curves'])!=len(inp['fovs_rad']):raise ValueError('incomplete FOV curve')
            count+=len(arm['curves'])
    compact.update(curve_columns=['max_abs_delta','ratios_pass'],fovs_rad=inp['fovs_rad'],
        full_local_raw_sha256=digest(out/'raw.json'),id=inp['id'],split=inp['split'],holdout=False,
        compaction='All FOVs, scalar differences and pass flags retained without rounding. Full five-ratio vectors remain local; fixed-camera vectors are in results.')
    ref=next(c for c in raw['cases'] if c['name']=='distributed/external_centroid')['arms'][0]['curves']
    centroid_delta={}
    for method in ['native_r8','native_r12','detector']:
        other=next(c for c in raw['cases'] if c['name']=='distributed/'+method)['arms'][0]['curves']
        centroid_delta[method]=max(abs(a-b) for x,y in zip(ref,other) for a,b in zip(x['ratios'],y['ratios']))
    artifacts={}
    for name,value in [(n,read(out/(n+'.json'))) for n in ['protocol','input','binary','results']]+[('curves',compact)]:
        path=ROOT/'docs/experiments'/('robustness-iteration-50-'+name+'.json')
        path.write_text(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n')
        artifacts[str(path.relative_to(ROOT))]=digest(path)
    for path in [*out.glob('*.json'),*out.glob('*.log'),Path(__file__).resolve(),Path(__file__).with_name('test_robustness_orion_pattern_correction.py')]:
        if path.name!='checks.json':artifacts[str(path.relative_to(ROOT))]=digest(path)
    checks=dict(iteration=50,id=inp['id'],split='development',holdout=False,
        python_tests=325,tetra3_rust_tests=54,clippy_passed=True,scalar_curve_points=count,
        raw49_max_ratio_error=results['replay49_max_ratio_error'],
        max_centroid_ratio_component_change=centroid_delta,
        max_inverse_roundtrip_after_f32_px=max(a['roundtrip_after_f32_px'] for c in raw['cases'] for a in c['arms']),
        protected_hashes=protocol['protected_hashes'],artifact_hashes=artifacts,
        new_fits=0,new_solver_calls=0,new_wcs_calls=0,process_failures=0,timeouts=0,
        production_changed=False,independent_ground_truth=False,automatic_improvement_confirmed=False)
    for path in [out/'checks.json',ROOT/'docs/experiments/robustness-iteration-50-checks.json']:
        path.write_text(json.dumps(checks,separators=(',',':'),allow_nan=False)+'\n')
    print(json.dumps(dict(curve_points=count,public_bytes=sum((ROOT/p).stat().st_size for p in artifacts if p.startswith('docs/')))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out-dir',type=Path,required=True)
    publish(parser.parse_args().out_dir.resolve())
