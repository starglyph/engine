#!/usr/bin/env python3
"""Plot saved candidate coverage and unscaled residuals; no photograph pixels."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from robustness_orion_fit import RID, WIDTH, HEIGHT, read


def plot(result, output):
    if (result['id'], result['split'], result['holdout']) != (RID, 'development', False):
        raise ValueError('development only')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    reviewed = np.array([s['coordinates']['external_centroid'] for s in result['sources']])
    dets = np.array(result['detector_xy_original'])
    pairs = np.array([p['xy'] for p in result['candidates'][0]['verification_pairs']])
    axes[0].scatter(*dets.T, s=25, facecolors='none', edgecolors='#547487', label='40 input detections')
    axes[0].scatter(*pairs.T, s=22, color='#be5b18', label='same 16 pairs in both')
    axes[0].legend(loc='upper center', bbox_to_anchor=(.5, -.22), fontsize=8)
    axes[0].set_title('Verification width: 14.17% of image')
    for ax, c in zip(axes[1:], result['candidates']):
        pred = np.array(c['predicted_xy_original'])
        ax.scatter(*reviewed.T, s=13, color='#34495e', label='47 reviewed centers')
        ax.quiver(*reviewed.T, *(pred-reviewed).T, angles='xy', scale_units='xy', scale=1,
                  width=.004, color='#b43a32', label='candidate residual (true scale)')
        rms = c['geometry']['external_centroid']['groups']['all47/all_reviewed']['rms_px']
        ax.set_title(f"Candidate {c['index']} {'(chosen)' if c['chosen'] else '(soft)'}: RMS {rms:.2f} px")
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.22), fontsize=8)
    for ax in axes:
        ax.axvspan(pairs[:, 0].min(), pairs[:, 0].max(), color='#ddaa66', alpha=.14)
        ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0), aspect='equal', xlabel='Original x (px)', ylabel='Original y (px)')
        ax.grid(alpha=.15)
    fig.suptitle('Development wm_r_154807046 — saved candidates; conditional identities, no ground truth', fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(output.with_suffix('.'+suffix), dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); plot(read(a.results), a.output)
