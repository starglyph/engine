#!/usr/bin/env python3
"""Frozen development-only component-growth veto, using the existing trace binary."""
import argparse
from collections import Counter
import math
import os
from pathlib import Path

from collection_run import ROOT, digest, select_records, write_json
from robustness_candidate_trace import command
from robustness_central_sources import read
from robustness_snow_sources import BINARY, OLD, DIAG as SNOW_DIAG, MANIFEST

IDS = ['wm_r_16053617', 'wm_r_112929230', 'wm_r_152165509', 'wm_r_124355480']
POSITIVE_DIAG = ROOT/'prototype/artifacts/robustness-iteration-1/regression-control-diagnostics/wm_r_112929230/solve-reports/wm_r_112929230.json'


def records():
    rows = select_records(read(MANIFEST), 'development', IDS)
    if {r['id'] for r in rows} != set(IDS):
        raise ValueError('four frozen development records required')
    return [next(r for r in rows if r['id'] == rid) for rid in IDS]


def prepare(out):
    if out.exists():
        raise ValueError('fresh output directory required')
    recs = records(); checks = read(ROOT/'docs/experiments/robustness-iteration-54-checks.json')
    for p in [OLD/'binary.json', OLD/'protocol.json']:
        if digest(p) != checks['artifact_hashes'][str(p.relative_to(ROOT))]:
            raise ValueError('saved binary provenance changed')
    binary = read(OLD/'binary.json'); old = read(OLD/'protocol.json')
    if digest(BINARY) != binary['sha256'] or digest(OLD/'protocol.json') != binary['protocol_sha256']:
        raise ValueError('saved binary changed')
    sources = sorted((ROOT/'prototype/crates/starglyph-core/src').rglob('*.rs'))
    for p in sources:
        if digest(p) != old['hashes'][str(p.relative_to(ROOT))]:
            raise ValueError('production source changed')
    images = [MANIFEST.parent/r['file'] for r in recs]
    for r,p in zip(recs,images):
        if digest(p) != r['clean_sha256']:
            raise ValueError('image changed')
    out.mkdir(parents=True)
    paths = [Path(__file__).resolve(),Path(__file__).with_name('test_robustness_component_growth.py'),
        *(Path(__file__).with_name(n) for n in ['collection_run.py','robustness_candidate_trace.py','robustness_central_sources.py','robustness_snow_sources.py']),
        MANIFEST, BINARY, OLD/'binary.json', OLD/'protocol.json',SNOW_DIAG,POSITIVE_DIAG,*images,*sources,
        *(ROOT/'data/samples/sky-samples'/n for n in ['robustness-results.json','collection-wcs-review.json'])]
    write_json(out/'protocol.json',dict(iteration=57,split='development',holdout=False,ids=IDS,
        inputs=[dict(id=r['id'],image=str(p),width=r['width'],height=r['height'],track=r['track'],
                     negative=r['research']['negative']) for r,p in zip(recs,images)],
        binary=str(BINARY.relative_to(ROOT)),radius_original_px=12,wall_limit_per_process_s=240,
        stages=['scout all default top40, no manual selection','trace selected centroids',
                'derive unique nearest integer threshold pixel; trace these anchors'],
        veto='only control default; exact default source and shared pixel; strictly lower deep threshold; deep outcome elongation',
        no_backfill=True,uncertain_action='retain',
        criteria=['all 12 trace modes per process equal uninstrumented detector (Rust assertion)',
            'all lists exactly stable across three stages; snow and positive control lists reproduce saved diagnostics',
            'no default source reassociation: centroid within 1e-8 px and rank equal; shared pixel distance below 1e-8',
            'P20 native snow vetoed; native snow controls ranks 0,2,3,34 retained; 21 saved working positive inliers retained',
            'inspect every vetoed source in neutral crops; compact stellar-looking veto is a safety failure',
            'no parameter retuning, matcher, WCS, holdout, solve-rate claim or production change'],
        hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}))


def validate(out):
    p=read(out/'protocol.json');records()
    if (p['ids'],p['split'],p['holdout']) != (IDS,'development',False):
        raise ValueError('development only')
    for name,sha in p['hashes'].items():
        if digest(ROOT/name)!=sha:
            raise ValueError('frozen input changed: '+name)
    return p


def key(t):
    return (t['quantile'],t['blob_concentration'],t['width'],t['height'],t['tier'])


def tiers(raw):
    rows=raw['tiers']; index={key(t):t for t in rows}
    if len(rows)!=12 or len(index)!=12:
        raise ValueError('12 unique trace tiers required')
    return index


def unique_anchor(trace):
    """Recover a pixel only when the saved exact distance has one integer solution.

    The trace already certifies that some component pixel has this distance.
    A unique lattice solution within its bounding box therefore is that pixel.
    Ties/absent components stay unknown; never use the box itself as membership.
    """
    c=trace['component']; d=trace['distance_to_component']
    if c is None or d is None or not math.isfinite(d) or d<0:
        return None
    x,y=trace['point'];left,top,right,bottom=c['bounds']
    candidates=[]
    for yy in range(max(top,math.ceil(y-d-1e-10)),min(bottom,math.floor(y+d+1e-10))+1):
        for xx in range(max(left,math.ceil(x-d-1e-10)),min(right,math.floor(x+d+1e-10))+1):
            if abs(math.hypot(xx-x,yy-y)-d)<1e-10:
                candidates.append([xx,yy])
    return candidates[0] if len(candidates)==1 else None


def lift(xy,width,height,original_width,original_height):
    return [(xy[0]+.5)*original_width/width-.5,(xy[1]+.5)*original_height/height-.5]


def exact_source(trace,source,rank):
    c=trace['component']
    return (c is not None and c['centroid'] is not None and c['rank_before_top_k']==rank
            and math.dist(c['centroid'],[source['x'],source['y']])<1e-8)


def veto_reason(high,low,anchor,source,rank,high_stats,low_stats):
    if anchor is None or not exact_source(high,source,rank):
        return 'retain_uncertain_default_association'
    if any(high_stats[k]!=low_stats[k] for k in ['sigma','sigma_convolved','background_median','statistics_pixels']):
        raise ValueError('preprocessing differs')
    if low_stats['threshold']>=high_stats['threshold']:
        return 'retain_threshold_not_lower'
    if any(t['component'] is None or t['distance_to_component'] is None or t['distance_to_component']>=1e-8 for t in [high,low]):
        return 'retain_unproven_shared_pixel'
    return 'veto_elongation' if low['component']['outcome']=='elongation' else 'retain_lower_component'


def trace_run(folder,stage,inp):
    write_json(folder/f'{stage}-input.json',inp)
    command(folder,stage,[str(BINARY),'--exact','solve::replay::source_traces','--ignored'],
        {**os.environ,'STARGLYPH_TRACE_INPUT':str(folder/f'{stage}-input.json'),
         'STARGLYPH_TRACE_OUTPUT':str(folder/f'{stage}-trace.json')},240)
    return read(folder/f'{stage}-trace.json')


def compare_lists(raw,reference):
    a,b=tiers(raw),tiers(reference)
    if a.keys()!=b.keys() or any(a[k]['result']!=b[k]['result'] for k in a):
        raise ValueError('detector lists changed between stages')


def run(out):
    protocol=validate(out)
    if any((out/rid).exists() for rid in IDS):
        raise ValueError('fresh measurement required')
    for rec in protocol['inputs']:
        rid=rec['id'];folder=out/rid;folder.mkdir()
        inp=dict(image=rec['image'],probes=[],radius_original_px=12)
        scout=trace_run(folder,'scout',inp);tiers(scout);probes=[]
        for t in scout['tiers']:
            if t['quantile'] or t['blob_concentration'] or t['tier']!='default':continue
            for rank,d in enumerate(t['result']['detections']):
                probes.append(dict(width=t['width'],height=t['height'],rank=rank,detection=d,
                    xy=lift([d['x'],d['y']],t['width'],t['height'],rec['width'],rec['height'])))
        write_json(folder/'selection.json',dict(id=rid,probes=probes))
        centers=trace_run(folder,'centers',{**inp,'probes':[p['xy'] for p in probes]})
        compare_lists(centers,scout);ci=tiers(centers);anchors=[]
        for i,p in enumerate(probes):
            tr=ci[(False,False,p['width'],p['height'],'default')]['probes'][i]
            anchor=unique_anchor(tr) if exact_source(tr,p['detection'],p['rank']) else None
            anchors.append(dict(pixel=anchor,xy=lift(anchor,p['width'],p['height'],rec['width'],rec['height']) if anchor else p['xy']))
        write_json(folder/'anchor-mapping.json',dict(id=rid,anchors=anchors))
        anchored=trace_run(folder,'anchors',{**inp,'probes':[p['xy'] for p in anchors]})
        compare_lists(anchored,scout)
        print(rid,'complete',flush=True)


def summarize(out):
    protocol=validate(out);frames=[];saved_lists=0
    positive=read(POSITIVE_DIAG)
    pd=next(d for d in positive['detection_diagnostics'] if d['width']==1600 and d['tier']=='default')
    reported=positive['report']['detections']
    if [d['flux'] for d in reported]!=[d['flux'] for d in pd['result']['detections'][:40]]:
        raise ValueError('positive inlier source order changed')
    for rec in protocol['inputs']:
        rid=rec['id'];folder=out/rid;scout=read(folder/'scout-trace.json');si=tiers(scout)
        centers=read(folder/'centers-trace.json');anchored=read(folder/'anchors-trace.json')
        compare_lists(centers,scout);compare_lists(anchored,scout);ai=tiers(anchored)
        if rid in IDS[:2]:
            saved=read(SNOW_DIAG if rid==IDS[0] else POSITIVE_DIAG)['detection_diagnostics']
            for d in saved:
                t=si[(False,False,d['width'],d['height'],d['tier'])]
                if t['result']['detections']!=d['result']['detections'][:d['max_detections']]:
                    raise ValueError('saved control mismatch')
                saved_lists+=1
        selected=read(folder/'selection.json')['probes'];anchors=read(folder/'anchor-mapping.json')['anchors'];rows=[]
        for i,(p,a) in enumerate(zip(selected,anchors)):
            high=ai[(False,False,p['width'],p['height'],'default')]
            low=ai[(False,False,p['width'],p['height'],'deep')]
            ht,lt=high['probes'][i],low['probes'][i]
            reason=veto_reason(ht,lt,a['pixel'],p['detection'],p['rank'],high['result']['stats'],low['result']['stats'])
            row={**p,'anchor':a['pixel'],'high':ht,'low':lt,'high_threshold':high['result']['stats']['threshold'],
                 'low_threshold':low['result']['stats']['threshold'],'decision':reason}
            row['saved_inlier_control']=rid==IDS[1] and p['width']==1600 and reported[p['rank']]['inlier']
            row['snow_morphology_control']=rid==IDS[0] and p['width']==2772 and p['rank'] in [0,2,3,34]
            rows.append(row)
        runs=[read(folder/f'{stage}.json') for stage in ['scout','centers','anchors']]
        frames.append(dict(id=rid,track=rec['track'],negative=rec['negative'],rows=rows,
            decisions=dict(Counter(r['decision'] for r in rows)),elapsed_s=sum(r['elapsed_s'] for r in runs),
            process_exit_codes=[r['exit_code'] for r in runs]))
    rows=[r for f in frames for r in f['rows']];vetoed=[r for r in rows if r['decision']=='veto_elongation']
    target=next(r for r in frames[0]['rows'] if r['width']==2772 and r['rank']==37)
    write_json(out/'results.json',dict(iteration=57,split='development',holdout=False,protocol_sha256=digest(out/'protocol.json'),
        frames=frames,exact_saved_control_lists=saved_lists,unchanged_trace_passes=144,
        target_vetoed=target['decision']=='veto_elongation',saved_inlier_controls=sum(r['saved_inlier_control'] for r in rows),
        vetoed_saved_inlier_controls=sum(r['saved_inlier_control'] for r in vetoed),
        vetoed_snow_morphology_controls=sum(r['snow_morphology_control'] for r in vetoed),
        production_changed=False,new_matcher_calls=0,new_wcs_calls=0,automatic_improvement_confirmed=False,
        independent_ground_truth=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','validate','run','summarize'])
    p.add_argument('--out-dir',type=Path,required=True);args=p.parse_args();globals()[args.stage](args.out_dir.resolve())
