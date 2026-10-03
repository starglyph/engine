"""Research controls must not silently alter verification or cross the split."""
import json
from pathlib import Path
import tempfile
import unittest

from collection_run import write_json
from robustness_catalog import RID, restore_key, run
from robustness_catalog_report import pattern_count


class CatalogDiagnosisTests(unittest.TestCase):
    def test_run_rejects_holdout_before_opening_binary_or_images(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            write_json(out / 'input.json', dict(protocol=dict(split='holdout', id=RID)))
            with self.assertRaisesRegex(ValueError, 'development'):
                run(out, Path('/nonexistent'))
            self.assertFalse((out / 'freeze-final.json').exists())

    def test_restoration_changes_search_only_and_preserves_pixel_centres(self):
        with tempfile.TemporaryDirectory() as folder:
            expanded = Path(folder) / 'expanded'
            out = Path(folder) / 'restore'
            expanded.mkdir(); out.mkdir()
            detections = [dict(x=10, y=20, flux=7), dict(x=40, y=50, flux=2)]
            source = dict(protocol=dict(split='development', id=RID), cases=[dict(
                name='4080-default-control', width=4080, height=3072,
                search=detections, verification=detections, search_ids=[100, 200])])
            write_json(expanded / 'input.json', source)
            before = (expanded / 'input.json').read_bytes()
            write_json(expanded / 'trace-input.json', dict(hyg_ids=[78165]))
            write_json(expanded / 'traces.json', dict(tiers=[dict(width=1600, height=1205,
                tier='default', quantile=False, blob_concentration=False,
                probes=[dict(component=dict(outcome='concentration', centroid=[100, 200]))])]))
            restore_key(expanded, out)
            cases = json.loads((out / 'input.json').read_text())['cases']
            self.assertEqual((expanded / 'input.json').read_bytes(), before)
            self.assertEqual(cases[0], source['cases'][0])
            self.assertEqual(cases[1]['verification'], detections)
            self.assertEqual(cases[1]['search'][1:], detections)
            self.assertEqual(cases[1]['search_ids'], [78165, 100, 200])
            restored = cases[1]['search'][0]
            self.assertAlmostEqual(restored['x'], 100.5*4080/1600-.5)
            self.assertAlmostEqual(restored['y'], 200.5*3072/1205-.5)
            self.assertGreater(restored['flux'], max(d['flux'] for d in detections))

    def test_unknown_ids_and_repetition_do_not_complete_pattern(self):
        patterns = [[1, 2, 3, 4], [1, 3, 5, 6]]
        self.assertEqual(pattern_count(patterns, [1, 2, 3, None, 3]), 0)
        self.assertEqual(pattern_count(patterns, [4, 3, 2, 1, None]), 1)


if __name__ == '__main__':
    unittest.main()
