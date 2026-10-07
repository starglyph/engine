#!/usr/bin/env python3
"""Plot exact numerator and denominator contributions for the frozen focus cases."""
import argparse
from pathlib import Path

from robustness_orion_pattern_edges import read, validate


def plot(out):
    validate(out)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    result=read(out/'results.json')
    fig,axes=plt.subplots(1,2,figsize=(13,5.3),layout='constrained',sharex=True)
    for ax,quartet,title in zip(axes,['check_quartet_1','local_control'],['Check 1 — no fit stars','Local control — 2 of 4 fit stars']):
        c=next(c for c in result['cases'] if c['quartet']==quartet and c['method']=='external_centroid')
        a=next(a for a in c['arms'] if a['name']=='seven');names={s['id']:s['name'] for s in c['sources']}
        labels=[' – '.join(names[i] for i in c['edge_source_ids'][e]) for e in c['catalog_order'][:5]]
        denominator=' – '.join(names[i] for i in c['edge_source_ids'][c['catalog_order'][-1]])
        y=np.arange(5)
        ax.barh(y-.19,a['numerator_contribution'],height=.32,color='#2863a3',label='Numerator contribution')
        ax.barh(y+.19,a['denominator_contribution'],height=.32,color='#d48720',label='Denominator contribution')
        ax.scatter(a['identified_ratio_delta'],y,marker='D',s=29,color='#111',label='Exact sum',zorder=4)
        for v in [-.006,.006]:ax.axvline(v,color='#b33',ls='--',lw=1)
        ax.axvline(0,color='#888',lw=.7);ax.set_yticks(y,labels);ax.invert_yaxis()
        ax.set(xlim=(-.014,.014),xlabel='Signed ratio error (observed − catalogue)',title=title+'\nDenominator: '+denominator)
        ax.grid(axis='x',alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3)
    fig.suptitle('Frozen Seven correction, FOV 62.711641° — external centroids\nDashed lines: ±0.006; conditional identities, no new solve')
    fig.savefig(out/'edge-contributions.png',dpi=160)
    fig.savefig(out/'edge-contributions.pdf',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',type=Path,required=True)
    plot(p.parse_args().out_dir.resolve())
