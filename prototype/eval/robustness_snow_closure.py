#!/usr/bin/env python3
"""Bounded evidence/coverage audit of the saved snow-frame failure; no solver calls."""
import argparse
from pathlib import Path

from astropy.io import fits
from PIL import Image, ImageDraw

from collection_run import ROOT, digest, select_records, write_json
from robustness_central_sources import read
from robustness_component_growth import lift
from robustness_reference_coverage import extent
from robustness_snow_foreground import RID, MANIFEST, WCS, FRACTION

PRIOR = ROOT/'docs/experiments'
INPUT = PRIOR/'robustness-iteration-55-input.json'
RESULT = PRIOR/'robustness-iteration-55-results.json'


def selected():
    rows=select_records(read(MANIFEST),'development',[RID])
    if len(rows)!=1 or rows[0]['id']!=RID or rows[0]['track']!='solver':
        raise ValueError('one development solver required')
    return rows[0]


def sky_coverage(points,width,height):
    sky=[d for d in points if d['y']<FRACTION*height]
    if any(not (0<=d['x']<width and 0<=d['y']<height) for d in points):
        raise ValueError('points outside original frame')
    grid=[[0]*4 for _ in range(2)]
    for d in sky:
        grid[int(2*d['y']/(FRACTION*height))][int(4*d['x']/width)]+=1
    return dict(all_detections=len(points),conservative_sky=extent(sky,width,FRACTION*height),
        sky_grid_4x2=grid,occupied_sky_cells=sum(n>0 for row in grid for n in row),
        catalogue_verified_star_coverage=None)


def prepare(out):
    rec=selected()
    if set(p.name for p in out.iterdir())!={'publisher-review.json'}:
        raise ValueError('fresh output with publisher-review.json only required')
    image=MANIFEST.parent/rec['file']
    if digest(image)!=rec['clean_sha256'] or read(WCS)['source_sha256']!=rec['clean_sha256']:
        raise ValueError('image provenance changed')
    for number,paths in [(55,[INPUT,RESULT]),(56,[PRIOR/'robustness-iteration-56-results.json']),
                         (57,[PRIOR/'robustness-iteration-57-results.json'])]:
        checks=read(PRIOR/f'robustness-iteration-{number}-checks.json')
        for p in paths:
            if digest(p)!=checks['artifact_hashes'][str(p.relative_to(ROOT))]:
                raise ValueError('saved evidence changed')
    inventory=sorted(p for p in WCS.parent.rglob('*') if p.is_file())
    paths=[Path(__file__).resolve(),Path(__file__).with_name('test_robustness_snow_closure.py'),
        *(Path(__file__).with_name(n) for n in ['collection_run.py','robustness_central_sources.py',
          'robustness_component_growth.py','robustness_reference_coverage.py','robustness_snow_foreground.py']),
        MANIFEST,image,INPUT,RESULT,*inventory,out/'publisher-review.json',
        *(PRIOR/f'robustness-iteration-{n}-results.json' for n in [56,57]),
        *(MANIFEST.parent/n for n in ['robustness-results.json','collection-wcs-review.json'])]
    write_json(out/'protocol.json',dict(iteration=58,id=RID,split='development',holdout=False,
        scope='read-only audit of existing evidence; no new detector, matcher, WCS or image download',
        coverage='existing y<0.45*height region; extent and fixed4x2 grid; detections never equated with verified stars',
        visual_selection='all40 saved native default manual_sky sources, no reselection or coordinate adjustment',
        stop_rule='without independently confirmed stellar identities/field geometry, close snow heuristic direction; no further algorithm iteration',
        wcs_inventory=[str(p.relative_to(ROOT)) for p in inventory],
        hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}))


def validate(out):
    p=read(out/'protocol.json');selected()
    if (p['id'],p['split'],p['holdout'])!=(RID,'development',False):
        raise ValueError('development only')
    if p['wcs_inventory']!=sorted(str(p.relative_to(ROOT))for p in WCS.parent.rglob('*') if p.is_file()):
        raise ValueError('external evidence inventory changed')
    for name,sha in p['hashes'].items():
        if digest(ROOT/name)!=sha:raise ValueError('frozen input changed: '+name)
    return p


def audit(out):
    protocol=validate(out);rec=selected();inp=read(INPUT);old=read(RESULT)
    if (out/'results.json').exists():raise ValueError('fresh audit required')
    rows=[]
    for case in inp['cases']:
        if case['id']!=RID:continue
        points=[]
        for d in case['search']:
            x,y=lift([d['x'],d['y']],case['width'],case['height'],rec['width'],rec['height'])
            points.append(dict(x=x,y=y))
        prior=next(c for c in old['cases'] if c['name']==case['name']);best=prior['funnel']['best']
        rows.append(dict(name=case['name'],coverage=sky_coverage(points,rec['width'],rec['height']),
            statuses=prior['statuses'],best_rejected=dict(probability_ratio=best['best_ratio'],
                pair_count=len(best['best']['pairs']),catalog_ids_unverified=[p[1] for p in best['best']['pairs']],
                finalized=best['finalized'],probability_pass=best['probability_pass']),
            verified_catalogue_identities=[],independent_edge_residual_px=None))
    external=[]
    for name in protocol['wcs_inventory']:
        path=ROOT/name
        if path.suffix!='.axy':continue
        with fits.open(path) as hdus:
            tables=[dict(columns=list(h.columns.names),rows=len(h.data))for h in hdus if hasattr(h,'columns')]
            celestial_keys=sorted({k for h in hdus for k in h.header if k.startswith(('CRVAL','CRPIX','CTYPE','CD1_','CD2_'))})
        log=path.with_name('solve.log').read_text()
        external.append(dict(path=name,tables=tables,celestial_header_keys=celestial_keys,
            cpu_limit_reached='Total CPU time limit reached!' in log))
    write_json(out/'results.json',dict(iteration=58,id=RID,split='development',holdout=False,
        protocol_sha256=digest(out/'protocol.json'),cases=rows,external_reference_status=read(WCS)['status'],
        external_extractions=external,wcs_or_correspondence_files=[n for n in protocol['wcs_inventory'] if Path(n).suffix in ['.wcs','.corr','.match','.rdls']],
        publisher_review=read(out/'publisher-review.json'),prior55_comparison_valid=old['comparison_valid'],
        conclusion='independent_stellar_identities_not_established; stop_this_heuristic_direction',
        accepted_ground_truth=False,automatic_improvement_confirmed=False,production_changed=False,
        new_detector_calls=0,new_matcher_calls=0,new_wcs_calls=0))


def render(out):
    validate(out);rec=selected();case=next(c for c in read(INPUT)['cases'] if c['name']=='2772-default-manual_sky')
    with Image.open(MANIFEST.parent/rec['file']) as source:im=source.convert('RGB')
    for start in [0,20]:
        sheet=Image.new('RGB',(800,925));draw=ImageDraw.Draw(sheet)
        for j,d in enumerate(case['search'][start:start+20]):
            x,y=round(d['x']),round(d['y']);bx=j%5*160;by=j//5*185
            sheet.paste(im.crop((x-64,y-64,x+64,y+64)).resize((160,160)),(bx,by))
            draw.ellipse((bx+74,by+74,bx+86,by+86),outline='yellow')
            draw.text((bx+4,by+164),f"S{start+j:02d}",fill='white')
        sheet=sheet.crop((0,0,800,740));sheet.save(out/f'sources-{start:02d}.jpg',quality=95)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','audit','render','validate'])
    p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args();globals()[a.stage](a.out_dir.resolve())
