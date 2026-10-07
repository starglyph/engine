#!/usr/bin/env python3
"""Render every component-growth veto without modifying numerical inputs."""
import argparse
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from collection_run import digest,write_json
from robustness_component_growth import read,validate,lift


def render(out):
    protocol=validate(out);results=read(out/'results.json');pages=[];summary=[]
    for frame,rec in zip(results['frames'],protocol['inputs']):
        if frame['id']!=rec['id']:raise ValueError('frame order changed')
        removed=[r for r in frame['rows'] if r['decision']=='veto_elongation']
        with Image.open(rec['image']) as source:im=source.convert('RGB')
        for start in range(0,len(removed),8):
            subset=removed[start:start+8];sheet=Image.new('RGB',(896,((len(subset)+3)//4)*256))
            draw=ImageDraw.Draw(sheet)
            for j,r in enumerate(subset):
                x,y=r['xy'];left,top=round(x)-64,round(y)-64;bx=j%4*224;by=j//4*256
                patch=im.crop((left,top,left+128,top+128)).resize((224,224))
                sheet.paste(patch,(bx,by));draw.ellipse((bx+104,by+104,bx+120,by+120),outline='yellow')
                draw.text((bx+4,by+228),f"{start+j}: w{r['width']} rank {r['rank']}",fill='white')
            name=f"{frame['id']}-veto-{start//8:02d}.jpg";sheet.save(out/name,quality=95)
            pages.append(dict(file=name,id=frame['id'],rows=[dict(width=r['width'],rank=r['rank'],xy=r['xy'])for r in subset]))
        for r in removed:
            if not r['saved_inlier_control']:continue
            x,y=r['xy'];left,top=round(x)-64,round(y)-64
            focus=im.crop((left,top,left+128,top+128)).resize((512,512));draw=ImageDraw.Draw(focus)
            for field,color in [('low','cyan'),('high','yellow')]:
                l,t,rr,b=r[field]['component']['bounds']
                lo=lift([l-.5,t-.5],r['width'],r['height'],rec['width'],rec['height'])
                hi=lift([rr+.5,b+.5],r['width'],r['height'],rec['width'],rec['height'])
                draw.rectangle(((lo[0]-left)*4,(lo[1]-top)*4,(hi[0]-left)*4,(hi[1]-top)*4),outline=color,width=2)
            draw.text((8,8),'yellow: default bounds; cyan: deep bounds',fill='white')
            name=f"{frame['id']}-lost-inlier-{r['rank']}.jpg";focus.save(out/name,quality=95)
            pages.append(dict(file=name,id=frame['id'],rows=[dict(width=r['width'],rank=r['rank'],xy=r['xy'])],
                purpose='saved inlier loss; rectangles are component bounding boxes, not masks'))
        per_scale=[]
        for width in sorted({r['width'] for r in frame['rows']}):
            rows=[r for r in frame['rows'] if r['width']==width];veto=[r for r in rows if r['decision']=='veto_elongation']
            per_scale.append(dict(width=width,before=len(rows),after=len(rows)-len(veto),vetoed_ranks=[r['rank']for r in veto],
                decisions=dict(Counter(r['decision']for r in rows)),
                saved_inlier_controls=sum(r['saved_inlier_control'] for r in rows),
                retained_inlier_controls=sum(r['saved_inlier_control'] and r['decision']!='veto_elongation' for r in rows)))
        summary.append(dict(id=frame['id'],track=frame['track'],negative=frame['negative'],scales=per_scale,
            elapsed_s=frame['elapsed_s'],process_exit_codes=frame['process_exit_codes']))
    write_json(out/'review-index.json',dict(results_sha256=digest(out/'results.json'),frames=summary,pages=pages,
        display='neutral square 128px crops enlarged to224px, yellow source marker, no brightness adjustment'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args();render(args.out_dir.resolve())
