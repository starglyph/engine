#!/usr/bin/env python3
"""Build local Astrometry.net indices for a shared-catalog algorithm cross-check.

HYG is also used by Starglyph: these indices are NOT an independent catalog.
Generated FITS are CC BY-SA 4.0 derivatives and must remain local artifacts.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from astropy.io import fits


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=Path(__file__).resolve().parents[2]/'data/catalogs/hyg_v42.csv.gz')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--build-index', default='build-astrometry-index')
    parser.add_argument('--scales', type=int, nargs='+', default=list(range(10, 20)))
    args = parser.parse_args()
    if len(set(args.scales)) != len(args.scales) or any(s not in range(10, 20) for s in args.scales):
        parser.error('scales must be distinct values from 10 to 19')
    root = args.out_dir.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error('use a new or empty output directory')
    with gzip.open(args.catalog, 'rt') as stream:
        rows = [row for row in csv.DictReader(stream) if row['id'] != '0' and float(row['mag']) <= 6.5]
    rows.sort(key=lambda row: float(row['mag']))
    coords = np.array([[float(row['ra'])*15, float(row['dec']), float(row['mag'])] for row in rows])
    if (len(rows) < 4 or not np.isfinite(coords).all() or np.any((coords[:, 0] < 0) | (coords[:, 0] >= 360))
            or np.any(np.abs(coords[:, 1]) > 90)):
        parser.error('invalid catalog coordinates')
    root.mkdir(parents=True, exist_ok=True)
    catalog = root/'hyg-bright.fits'
    fits.BinTableHDU.from_columns([
        fits.Column(name='RA', format='D', array=coords[:, 0]),
        fits.Column(name='DEC', format='D', array=coords[:, 1]),
        fits.Column(name='MAG', format='E', array=coords[:, 2]),
        fits.Column(name='HYG_ID', format='J', array=[int(row['id']) for row in rows]),
    ]).writeto(catalog)
    provenance = {'source': str(args.catalog.resolve()), 'source_sha256': digest(args.catalog),
                  'filter': 'id != 0 and mag <= 6.5', 'n_stars': len(rows),
                  'license': 'CC BY-SA 4.0', 'attribution': 'HYG Database, Astronexus',
                  'purpose': 'independent algorithm check with shared engine catalog; not independent catalog accuracy',
                  'commands': [], 'indices': []}
    for scale in args.scales:
        index = root/f'index-hyg-{scale}.fits'
        command = [args.build_index, '-i', str(catalog), '-o', str(index), '-P', str(scale),
                   '-S', 'MAG', '-I', str(9900+scale), '-M']
        with (root/f'build-{scale}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        provenance['commands'].append(command)
        provenance['indices'].append({'file': index.name, 'sha256': digest(index)})
        (root/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
        print(scale, flush=True)
    # Explicit index list avoids trying to open the unindexed source FITS.
    config = root/'backend.cfg'
    config.write_text('cpulimit 30\nadd_path '+str(root)+'\n'+''.join(
        f'index index-hyg-{scale}.fits\n' for scale in args.scales))
    provenance['config_sha256'] = digest(config)
    (root/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')


if __name__ == '__main__':
    main()
