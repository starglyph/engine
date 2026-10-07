"""Guards for reusing the observer on a new development frame."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from collection_run import digest, write_json
from robustness_orion_refinement import RID, validate
from robustness_refinement_report import lift
from robustness_refinement_crossover_report import close


class OrionRefinementTests(unittest.TestCase):
    def test_scope_guard_precedes_source_access(self):
        for rid, split, holdout in [(RID, 'holdout', True), ('wm_r_146925915', 'development', False)]:
            with tempfile.TemporaryDirectory() as folder:
                out = Path(folder)
                write_json(out/'protocol.json', dict(id=rid, split=split, holdout=holdout))
                with patch('robustness_orion_refinement.selected') as select:
                    with self.assertRaisesRegex(ValueError, 'development only'):
                        validate(out)
                    select.assert_not_called()

    def test_observer_hash_is_enforced(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            binary = out/'observer'
            binary.write_bytes(b'original')
            write_json(out/'protocol.json', dict(id=RID, split='development', holdout=False,
                hashes={str(binary): digest(binary)}))
            binary.write_bytes(b'changed')
            with patch('robustness_orion_refinement.selected'):
                with self.assertRaisesRegex(ValueError, 'frozen input changed'):
                    validate(out)

    def test_nonuniform_resize_keeps_pixel_centers(self):
        points = np.array([[0., 0.], [1599., 1041.]])
        native = lift(points, dict(width=1600, height=1042), 4485, 2920)
        np.testing.assert_allclose((native+.5)/[4485/1600, 2920/1042]-.5, points, atol=1e-12)
        self.assertGreater(float(np.max(np.abs(native-points*(4485/1600)))), .5)

    def test_exact_reproduction_rejects_a_changed_inlier_flag(self):
        a = dict(camera=dict(focal_px=1000.), detections=[dict(x=12., inlier=True)])
        b = dict(camera=dict(focal_px=1000.), detections=[dict(x=12., inlier=False)])
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            close(a, b, tolerance=0)
