#!/usr/bin/env python3
"""Export residual vectors as a scientific figure; no interpolation or fit."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from collection_run import digest, write_json
from robustness_residual_fields import validate


def plot(out):
    protocol,_ = validate(out)
    result = json.loads((out/'results.json').read_text())
    if result['protocol_sha256'] != digest(out/'protocol.json'):
        raise ValueError('result protocol differs')
    plt.rcParams.update({'font.size':10, 'svg.hashsalt':'starglyph-iteration-12'})
    fig,axes = plt.subplots(3,2,figsize=(10,16),layout='constrained')
    gain = protocol['plot_arrow_magnification']
    for row,frame in enumerate(result['frames']):
        xy = np.array([s['xy'] for s in frame['sources']])
        outside = np.array([s['outside_both_inputs'] for s in frame['sources']])
        w,h = frame['width'],frame['height']
        for col,arm in enumerate(protocol['arms']):
            ax = axes[row,col]
            ax.add_patch(Rectangle((0,0),w,h,facecolor='#fafafa',edgecolor='#73808c',lw=.8,zorder=0))
            for fraction in (1/3,2/3):
                ax.plot([w*fraction]*2,[0,h],color='#dddddd',lw=.7,zorder=0)
                ax.plot([0,w],[h*fraction]*2,color='#dddddd',lw=.7,zorder=0)
            residual = np.array(frame['arms'][arm]['residual_xy_px'])
            colors = np.where(outside,'#d05a00','#155b9b')
            q = ax.quiver(xy[:,0],xy[:,1],residual[:,0],residual[:,1],color=colors,
                angles='xy',scale_units='xy',scale=1/gain,width=.004,headwidth=3.5,headlength=4.5)
            ax.scatter(xy[:,0],xy[:,1],s=7,c=colors)
            ax.plot(w/2,h/2,'+',color='#555555',ms=8)
            ax.quiverkey(q,.79,.94,10,'10 original px',coordinates='axes',labelpos='S',color='#333333')
            ax.set_xlim(-w*.13,w*1.13)
            ax.set_ylim(h*1.13,-h*.13)
            ax.set_aspect('equal')
            ax.set_xlabel('x, original pixels')
            ax.set_ylabel('y, original pixels (down)')
            rms = frame['arms'][arm]['groups']['all_reviewed']['rms_px']
            ax.set_title(f"{frame['id']} / {arm}\nN={len(xy)}, RMS={rms:.2f} px")
    fig.suptitle('Frozen development residual fields: predicted minus measured\n'
                 f'Arrows enlarged {gain:g}x; no fitted correction or interpolated field',fontsize=13)
    handles=[Line2D([0],[0],color='#155b9b',marker='o',label='Within previous input lists'),
             Line2D([0],[0],color='#d05a00',marker='o',label='Outside both previous input lists')]
    fig.legend(handles=handles,loc='outside lower center',ncol=2)
    fig.savefig(out/'vectors.png',dpi=160)
    fig.savefig(out/'vectors.svg',metadata={'Date':None})
    fig.savefig(out/'vectors.pdf',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    write_json(out/'plot.json',dict(matplotlib=matplotlib.__version__,numpy=np.__version__,
        results_sha256=digest(out/'results.json'),script_sha256=digest(Path(__file__)),
        files={name:digest(out/name) for name in ('vectors.png','vectors.svg','vectors.pdf')}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    plot(parser.parse_args().out.resolve())
