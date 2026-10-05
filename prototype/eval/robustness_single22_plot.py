#!/usr/bin/env python3
"""Export fixed-pair residual maps; scientific plot, no pixel image processing."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from collection_run import digest, write_json
from robustness_single22_residuals import ARMS, read, validate


def plot(out):
    protocol = validate(out)
    result = read(out/'results.json')
    if result['protocol_sha256'] != digest(out/'protocol.json'):
        raise ValueError('result protocol changed')
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'starglyph-iteration-29'})
    fig, axes = plt.subplots(2,2,figsize=(10,13),layout='constrained')
    w,h = result['width'], result['height']
    gain = protocol['plot_arrow_magnification']
    for row,name in enumerate(('fit22','probes35')):
        data = result['datasets'][name]
        base = data['base']
        xy = np.array(data['positions'][base])
        outside = np.zeros(len(xy),dtype=bool)
        outside[data['group_indices'].get('outside_all40_inputs',[])] = True
        colors = np.where(outside,'#cc5900','#185b9a')
        for col,arm in enumerate(ARMS):
            ax = axes[row,col]
            ax.add_patch(Rectangle((0,0),w,h,fill=False,edgecolor='#777',lw=.8))
            for fraction in (1/3,2/3):
                ax.axvline(w*fraction,color='#ddd',lw=.7,zorder=0)
                ax.axhline(h*fraction,color='#ddd',lw=.7,zorder=0)
            vectors = np.array(data['arms'][arm]['methods'][base]['residual_xy_px'])
            q = ax.quiver(xy[:,0],xy[:,1],vectors[:,0],vectors[:,1],color=colors,
                angles='xy',scale_units='xy',scale=1/gain,width=.004)
            ax.scatter(*xy.T,s=9,c=colors)
            ax.plot(w/2,h/2,'+',color='#555',ms=9)
            ax.quiverkey(q,.75,.15,10,'10 original px',coordinates='axes',labelpos='S')
            ax.set(xlim=(-w*.1,w*1.1),ylim=(h*1.05,-h*.06),aspect='equal',xlabel='x, original pixels',ylabel='y, original pixels (down)')
            rms = data['arms'][arm]['methods'][base]['groups']['all_reviewed']['rms_px']
            ax.set_title(f'{name} / {arm} camera\nRMS={rms:.2f} px')
    fig.suptitle(f"{result['id']}: predicted minus measured\nArrows enlarged {gain:g}x; orange = outside all40 inputs; no interpolation")
    for ext in ('png','svg','pdf'):
        metadata = {'Date':None} if ext=='svg' else {'CreationDate':None,'ModDate':None} if ext=='pdf' else None
        fig.savefig(out/f'vectors.{ext}',dpi=160,metadata=metadata,bbox_inches='tight')
        if ext == 'svg':
            svg = out/'vectors.svg'
            svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    write_json(out/'plot.json',dict(matplotlib=matplotlib.__version__,numpy=np.__version__,
        script_sha256=digest(Path(__file__)),results_sha256=digest(out/'results.json'),
        files={ext:digest(out/f'vectors.{ext}') for ext in ('png','svg','pdf')}))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    plot(parser.parse_args().out_dir.resolve())
