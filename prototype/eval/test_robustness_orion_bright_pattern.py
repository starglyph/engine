"""Guard catalogue-only selection, held-out independence and interpolation claims."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from collection_run import write_json
from robustness_orion_bright_pattern import choose_pattern, evaluate, hull, inside, validate, RID, IDS


class BrightPatternTests(unittest.TestCase):
    def catalogue(self):
        return [dict(id=i, hyg_id=100+i, name=str(i), mag=4., radec=[ra % 360, dec])
                for i, (ra, dec) in enumerate([(1, 1), (-1, 1), (1, -1), (-1, -1),
                                               (.5, .5), (-.5, -.5), (25, 25)])]

    def fixture(self):
        xy = [[-1., -1.], [1., -1.], [1., 1.], [-1., 1.], [.3, .2], [-.4, .3]]
        def coords(p):
            return {'external_centroid': [100+40*p[0]+3*p[1], 200-2*p[0]+45*p[1]]}
        return dict(id=15, status='ready', coordinates=coords([0, 0]),
                    points=[dict(id=i, tangent_xy=p, role='anchor' if i < 4 else 'held_out',
                                 coordinates=coords(p)) for i, p in enumerate(xy)])

    def test_catalogue_selection_excludes_targets_and_ignores_pixels(self):
        cat = self.catalogue()
        for sid in IDS:
            cat.append(dict(id=sid, hyg_id=200+sid, name='excluded', mag=0., radec=[.01, .01]))
        target = dict(radec=[0, 0]); a = choose_pattern(target, cat)
        changed = copy.deepcopy(cat)
        for p in changed:
            p['coordinates'] = {'external_centroid': [1e6, -1e6]}
        b = choose_pattern(target, changed[::-1])
        self.assertEqual(a, b)
        self.assertEqual(a['status'], 'ready')
        self.assertEqual(len(a['points']), 6)
        self.assertEqual({p['quadrant'] for p in a['points'][:4]}, {0, 1, 2, 3})
        self.assertTrue(all(p['id'] not in (*IDS, 6) for p in a['eligible']))

    def test_missing_quadrant_is_not_filled_by_distant_star(self):
        cat = [p for p in self.catalogue() if p['id'] not in (0, 4)]
        result = choose_pattern(dict(radec=[0, 0]), cat)
        self.assertEqual(result['status'], 'insufficient_coverage')
        self.assertEqual(result['missing_quadrants'], [2])
        self.assertEqual(evaluate(result, 'external_centroid')['fit_attempts'], 0)

    def test_held_out_points_do_not_change_homography(self):
        t = self.fixture(); before = evaluate(t, 'external_centroid')
        self.assertTrue(before['criterion_passed'])
        t['coordinates']['external_centroid'] = [400., 500.]
        t['points'][-1]['coordinates']['external_centroid'] = [900., -500.]
        after = evaluate(t, 'external_centroid')
        np.testing.assert_allclose(before['homography'], after['homography'], atol=0)
        self.assertFalse(after['criterion_passed'])
        self.assertFalse(after['points'][-1]['inside_image_hull'])

    def test_hull_handles_interior_duplicates_collinearity_and_boundary(self):
        polygon = hull([[0, 0], [2, 0], [0, 2], [.1, .1], [0, 0]])
        self.assertEqual(len(polygon), 3)
        self.assertTrue(inside([.5, .5], polygon))
        self.assertFalse(inside([1.5, 1.5], polygon))
        self.assertFalse(inside([1, 1], polygon))
        self.assertFalse(inside([1, 0], hull([[0, 0], [1, 0], [2, 0]])))

    def test_degenerate_anchors_are_a_recorded_failure(self):
        t = self.fixture()
        for i, p in enumerate(t['points'][:4]):
            p['tangent_xy'] = [float(i), 0.]
        self.assertEqual(evaluate(t, 'external_centroid')['status'], 'degenerate_registration')

    def test_holdout_rejected_before_file_access(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            write_json(out/'protocol.json', dict(id=RID, split='holdout', holdout=True, source_ids=list(IDS)))
            with self.assertRaisesRegex(ValueError, 'development'):
                validate(out)


if __name__ == '__main__':
    unittest.main()
