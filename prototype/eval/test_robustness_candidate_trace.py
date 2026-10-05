"""Guard isolated observation and explicit single-frame selection."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import robustness_candidate_trace as trace
from robustness_candidate_trace_report import project
import numpy as np


class CandidateTraceTests(unittest.TestCase):
    def test_observer_patch_is_reversible(self):
        original = (trace.ROOT/'prototype/crates/starglyph-core/src/solve.rs').read_text()
        control = trace.instrument(original, 'control')
        restored = control.removesuffix('\nmod candidate_trace;\n').replace(trace.HOOK_AFTER, trace.HOOK_BEFORE)
        self.assertEqual(restored, original)
        self.assertEqual(trace.instrument(original, 'centered'), control.replace(trace.BEFORE, trace.AFTER))
        with self.assertRaises(ValueError):
            trace.instrument(control, 'control')

    def test_other_frame_rejected_before_file_access(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            for plan in (dict(id='another',holdout=False),dict(id=trace.FRAME,holdout=True)):
                (out/'protocol.json').write_text(json.dumps(plan))
                with patch.object(trace,'digest',side_effect=AssertionError('must not read')):
                    with self.assertRaisesRegex(ValueError,'only frozen legacy'):
                        trace.validate(out)

    def test_projection_axes_and_front_guard(self):
        camera=dict(ra_deg=0.,dec_deg=0.,roll_deg=0.,focal_px=1000.,width=1600,height=900)
        np.testing.assert_allclose(project(camera,[[1,0,0],[1,.1,0],[1,0,.1]]),
                                   [[800,450],[700,450],[800,350]])
        with self.assertRaisesRegex(ValueError,'behind camera'):
            project(camera,[[-1,0,0]])

    def test_extract_only_complete_trace_records(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'log'
            p.write_text('irrelevant\n'+trace.PREFIX+'{"pairs": []}\n  [dense] k=16\n')
            self.assertEqual(trace.traces(p),[dict(pairs=[])])
            p.write_text(trace.PREFIX+'broken\n')
            with self.assertRaises(json.JSONDecodeError):
                trace.traces(p)


if __name__=='__main__':
    unittest.main()
