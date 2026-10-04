"""Geometry review must not alter observations or silently cross the split."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from collection_run import digest, write_json
from robustness_geometry import group_metrics, measure, native_centroid, prepare, validate_review
from robustness_geometry_wcs import check


class GeometryReviewTests(unittest.TestCase):
    def template(self):
        return dict(split='development', frames=[dict(id='example', width=100,
            points=[dict(id=0, hyg_id=1, ra_deg=20., dec_deg=10.,
                choices=[dict(row=3, x=12., y=15.)], label='pending', selected_row=None)])])

    def test_labels_can_change_but_measurements_and_identity_cannot(self):
        source = self.template()
        review = deepcopy(source)
        review['frames'][0]['points'][0].update(label='visible_source', selected_row=3, note='reviewed')
        validate_review(source, review)
        review['frames'][0]['points'][0]['choices'][0]['x'] += 1
        with self.assertRaisesRegex(ValueError, 'candidate changed'):
            validate_review(source, review)
        review = deepcopy(source)
        review['frames'][0]['points'][0]['hyg_id'] = 2
        with self.assertRaisesRegex(ValueError, 'candidate changed'):
            validate_review(source, review)

    def test_missing_points_or_changed_dimensions_are_rejected(self):
        for key, value in [('points', []), ('width', 200)]:
            source = self.template()
            review = deepcopy(source)
            review['frames'][0][key] = value
            with self.assertRaises(ValueError):
                validate_review(source, review)

    def test_prepare_rejects_holdout_before_creating_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            with self.assertRaisesRegex(ValueError, 'outside selected split'):
                prepare(out, ['wm_r_146925915'])
            self.assertFalse(out.exists())

    def test_measure_rejects_wrong_split_before_loading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            write_json(out / 'protocol.json', dict(split='holdout'))
            with self.assertRaisesRegex(ValueError, 'development'):
                measure(out, out / 'result.json')
            self.assertFalse((out / 'result.json').exists())

    def test_image_only_centroid_recovers_shifted_star(self):
        y, x = np.mgrid[:80, :80]
        true = np.array([40.3, 37.7])
        image = 20 + 150*np.exp(-((x-true[0])**2 + (y-true[1])**2)/4)
        for radius in (8, 12):
            found = native_centroid(image, [41., 37.], radius)
            np.testing.assert_allclose(found, true, atol=.001)

    def test_blank_aperture_is_not_a_star(self):
        with self.assertRaisesRegex(ValueError, 'no positive source flux'):
            native_centroid(np.ones((80, 80)), [40, 40], 8)

    def test_groups_keep_regressions_and_empty_groups(self):
        rows = [dict(control=dict(error_px=1), footprint=dict(error_px=3)),
                dict(control=dict(error_px=4), footprint=dict(error_px=2))]
        result = group_metrics(rows, [True, True])
        self.assertEqual((result['improved'], result['worsened']), (1, 1))
        self.assertEqual(result['median_paired_error_change_px'], 0)
        self.assertEqual(group_metrics(rows, [False, False]), dict(count=0))

    def test_external_check_rejects_wrong_split(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            write_json(out / 'protocol.json', dict(split='holdout'))
            write_json(out / 'review.json', dict(split='development'))
            with self.assertRaisesRegex(ValueError, 'development'):
                check(out, out / 'result.json')
            self.assertFalse((out / 'result.json').exists())

    def test_external_wcs_origin_and_low_weight_source_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            refdir = root / 'prototype/artifacts/robustness-baseline/wcs-run/example'
            refdir.mkdir(parents=True)
            wcs = WCS(naxis=2)
            wcs.wcs.crpix = [100, 100]
            wcs.wcs.crval = [25, 30]
            wcs.wcs.cdelt = [-.01, .01]
            wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
            fits.PrimaryHDU(np.zeros((200, 200)), header=wcs.to_header()).writeto(refdir / 'field.wcs')
            xy = np.array([[40., 60.], [90., 100.], [140., 150.]])
            world = wcs.all_pix2world(xy, 0)
            # Both high- and low-weight correspondence rows must be withheld.
            cols = [fits.Column(name=name, format='D', array=values) for name, values in
                    [('field_x', xy[:2, 0]+1), ('field_y', xy[:2, 1]+1), ('match_weight', [1., .01])]]
            fits.BinTableHDU.from_columns(cols).writeto(refdir / 'field.corr')
            points = [dict(id=i, hyg_id=i+1, ra_deg=ra, dec_deg=dec,
                          choices=[dict(row=i, x=x, y=y)], label='visible_source', selected_row=i)
                      for i, ((x, y), (ra, dec)) in enumerate(zip(xy, world))]
            review = dict(split='development', frames=[dict(id='example', source_sha256='image', points=points)])
            write_json(root / 'review.json', review)
            write_json(root / 'candidates.json', review)
            write_json(root / 'protocol.json', dict(split='development', ids=['example'],
                       candidates_sha256=digest(root / 'candidates.json')))
            reference = dict(source_sha256='image', status='solved_candidate', review_status='pending')
            for key, filename in [('wcs', 'field.wcs'), ('correspondences', 'field.corr')]:
                reference[key+'_file'] = str((refdir / filename).relative_to(root / 'prototype'))
                reference[key+'_sha256'] = digest(refdir / filename)
            write_json(refdir / 'reference.json', reference)
            with patch('robustness_geometry_wcs.ROOT', root), patch('robustness_geometry_wcs.selected'):
                check(root, root / 'result.json')
                result = json.loads((root / 'result.json').read_text())['frames'][0]
                self.assertLess(result['all_reviewed']['max_px'], 1e-7)
                self.assertEqual(result['high_weight_count'], 1)
                self.assertEqual(result['outside_all_corr_sources']['count'], 1)
                self.assertEqual([p['outside_all_corr_sources'] for p in result['sources']], [False, False, True])
                with (refdir / 'field.corr').open('ab') as stream:
                    stream.write(b'changed')
                with self.assertRaisesRegex(ValueError, 'external artifact changed'):
                    check(root, root / 'changed.json')


if __name__ == '__main__':
    unittest.main()
