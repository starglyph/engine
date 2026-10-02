import json
import shutil
import unittest

import test_smartphone_gate as fixtures
from sky_mask_experiment import compare, in_sky


class SkyMaskExperimentTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SmartphoneGateTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        for path in (f.run / 'solve-reports').glob('*.json'):
            value = json.loads(path.read_text())
            value['report']['timing_ms'] = {'detect': 1, 'solve': 2, 'total': 3}
            path.write_text(json.dumps(value))
        self.output = f.root / 'experiment'
        for mode in ('unmasked', 'masked'):
            shutil.copytree(f.run, self.output / mode)
        self.annotation = {'id': 'b', 'source_sha256': f.baseline['frames']['b']['sha256'],
                           'width': 2252, 'height': 4000, 'sky_polygon': [[0, 0], [1, 0], [1, 0.5], [0, 0.5]]}
        self.masks = f.root / 'masks.json'
        self.masks.write_text(json.dumps({'masks': [self.annotation]}))
        self.update('summary.json', lambda value: value['config'].update(sky_masks=str(self.masks)))
        self.update('solve-reports/b.json', lambda value: value.update(sky_mask=self.annotation))

    def update(self, file, change):
        path = self.output / 'masked' / file
        value = json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value))

    def compare(self):
        f = self.fixture
        return compare(f.manifest, f.baseline, self.masks, self.output)

    def test_pair_and_unchanged_frame_protection(self):
        self.assertEqual(self.compare()['solved'], {'unmasked': 1, 'masked': 1})
        self.update('solve-reports/a.json', lambda value: value['report']['pose'].update(ra_deg=42))
        with self.assertRaisesRegex(ValueError, 'unmasked result changed'):
            self.compare()

    def test_equal_flux_order_is_not_a_regression(self):
        points = [{'flux': 1., 'x': 10., 'y': 20., 'snr': 4., 'inlier': False},
                  {'flux': 1., 'x': 30., 'y': 40., 'snr': 5., 'inlier': False}]
        for mode in ('unmasked', 'masked'):
            path = self.output / mode / 'solve-reports' / 'a.json'
            value = json.loads(path.read_text())
            value['report']['detections'] = points if mode == 'unmasked' else list(reversed(points))
            path.write_text(json.dumps(value))
        self.compare()
        self.update('solve-reports/a.json', lambda value: value['report']['detections'][0].update(snr=99))
        with self.assertRaisesRegex(ValueError, 'unmasked result changed'):
            self.compare()

    def test_mask_identity_protection(self):
        self.update('solve-reports/b.json', lambda value: value['sky_mask'].update(width=4000))
        with self.assertRaisesRegex(ValueError, 'mask provenance mismatch'):
            self.compare()

    def test_pixel_center_selection(self):
        self.assertTrue(in_sky({'x': 100, 'y': 100}, self.annotation))
        self.assertFalse(in_sky({'x': 100, 'y': 3000}, self.annotation))
        self.assertFalse(in_sky({'x': 100, 'y': 1999.5}, self.annotation))


if __name__ == '__main__':
    unittest.main()
