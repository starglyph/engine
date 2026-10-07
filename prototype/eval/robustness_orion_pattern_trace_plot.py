#!/usr/bin/env python3
"""Plot observed ratio mismatch, not a new FOV sweep or parameter fit."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from robustness_orion_fit import RID, read


def plot(data, output):
    if (data['id'],data['split'],data['holdout']) != (RID,'development',False):
        raise ValueError('development only')
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
    for ax,name,title in zip(axes,('distributed','local_control'),
                             ('Distributed: Sirius / Procyon / Rigel / Betelgeuse','Local control: Rigel / Alnilam / Tabit / Arneb')):
        points=sorted({(w['image']['fov'],w['assessment']['max_abs_ratio_delta'])
                       for w in data['windows'] if w['target']==name})
        ax.plot([x for x,y in points],[y for x,y in points],'.',ms=3,color='#345e8a',label='Observed max |ratio difference|')
        ax.axhline(.006,color='#b84535',ls='--',label='Dense tolerance 0.006')
        ax.axhline(.005,color='#956336',ls=':',label='Bootstrap tolerance 0.005')
        ax.set(title=title,xlabel='Visited FOV (degrees)',ylabel='Dimensionless ratio difference',ylim=(0,.025))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Development wm_r_154807046 — fixed quartets, existing search only',fontsize=12)
    output.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(output.with_suffix('.'+ext),dpi=160)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plot(read(a.results),a.output)
