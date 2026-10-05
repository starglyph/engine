"""Guard residual cross-validation and the frozen single-frame scope."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from robustness_pair_audit import linear_diagnostics, validate, FRAME


class PairAuditTests(unittest.TestCase):
    def test_translation_predicts_unseen_pairs(self):
        xy=np.array([[20,30],[180,30],[30,150],[170,155],[100,60],[80,125]],dtype=float)
        residual=np.tile([2.,-1.],(len(xy),1))
        rows=linear_diagnostics(xy,residual,150,[200,180])
        for row in rows.values():
            self.assertLess(row['training_rms_px'],1e-10)
            self.assertLess(row['leave_one_out_rms_px'],1e-10)

    def test_degenerate_design_rejected(self):
        with self.assertRaisesRegex(ValueError,'rank-deficient'):
            linear_diagnostics(np.ones((6,2)),np.zeros((6,2)),100,[200,200])

    def test_scope_guard_runs_before_hash_access(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            for data in [dict(id='another',holdout=False),dict(id=FRAME,holdout=True)]:
                (out/'protocol.json').write_text(json.dumps(data))
                with patch('robustness_pair_audit.digest',side_effect=AssertionError('must not access')):
                    with self.assertRaisesRegex(ValueError,'fixed legacy'):
                        validate(out)


if __name__=='__main__':
    unittest.main()
