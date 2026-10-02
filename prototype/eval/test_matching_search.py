import copy
import unittest

from matching_search_experiment import cases, reduced_deep


class MatchingSearchTests(unittest.TestCase):
    def test_negative_preserves_photometry_without_mutating_source(self):
        detections = [dict(x=20+i, y=30+i, flux=10-i, rank=i, area=5) for i in range(8)]
        original = copy.deepcopy(detections)
        first = cases(detections, 900, 1600, 20261002)
        self.assertEqual(first, cases(detections, 900, 1600, 20261002))
        self.assertEqual(detections, original)
        negative = first[1]['detections']
        self.assertNotEqual(negative, cases(detections, 900, 1600, 20261003)[1]['detections'])
        for source, randomized in zip(original, negative):
            self.assertEqual({k: v for k, v in source.items() if k not in ('x', 'y')},
                             {k: v for k, v in randomized.items() if k not in ('x', 'y')})
            self.assertTrue(8 <= randomized['x'] <= 892)
            self.assertTrue(8 <= randomized['y'] <= 1592)

    def test_portrait_and_landscape_use_long_edge(self):
        for width, height in [(1600, 900), (900, 1600)]:
            expected = dict(width=width, height=height, tier='deep')
            report = dict(detection_diagnostics=[
                dict(width=width, height=height, tier='default'),
                dict(width=4000, height=2252, tier='deep'), expected])
            self.assertIs(reduced_deep(report), expected)

    def test_missing_or_duplicate_scale_is_rejected(self):
        for diagnostics in [[], [dict(width=1600, height=900, tier='deep')]*2]:
            with self.assertRaises(ValueError):
                reduced_deep(dict(detection_diagnostics=diagnostics))
