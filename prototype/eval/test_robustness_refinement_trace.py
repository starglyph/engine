"""Protect diagnostic scope, production block extraction, and equivalence checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import robustness_refinement_trace as trace
from robustness_refinement_crossover_report import close


class RefinementTraceTests(unittest.TestCase):
    def test_observer_preserves_acceptance_and_post_refinement(self):
        source = (trace.ROOT/'prototype/crates/starglyph-core/src/solve.rs').read_text()
        patched = trace.instrument(source)
        before = source[:source.index(trace.START)]
        self.assertEqual(patched[:patched.index('    refinement_trace::input(')],before)
        tail = source[source.index(trace.END):].replace(trace.HOOK_BEFORE,trace.HOOK_AFTER)
        self.assertEqual(patched[patched.index(trace.END):],tail+'\nmod candidate_trace;\nmod refinement_trace;\n')
        generated = trace.module(source)
        self.assertIn('for radius in WIDE_REMATCH_RADII_PX',generated)
        self.assertIn('if rematch.matches.len() >= REMATCH_MIN_MATCHES',generated)
        self.assertIn('match_predictions(&refined, verify, detections, FINAL_RADIUS_PX)',generated)
        self.assertNotIn('GENERATED_REFINEMENT_PATH',generated)

    def test_changed_boundaries_rejected(self):
        with self.assertRaises(ValueError):
            trace.extract('wrong source')

    def test_holdout_or_missing_solver_cannot_be_selected(self):
        for rows in [[],[dict(id=trace.FRAME,track='stress')]]:
            with patch.object(trace,'records',return_value=rows):
                with self.assertRaises(ValueError):
                    trace.selected()

    def test_frozen_selection_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            plan = dict(id=trace.FRAME,split='holdout',holdout=False,hashes={})
            (out/'protocol.json').write_text(json.dumps(plan))
            with patch.object(trace,'selected'):
                with self.assertRaisesRegex(ValueError,'selection'):
                    trace.validate(out)
                plan.update(split='development',hashes={'binary':'old'})
                (out/'protocol.json').write_text(json.dumps(plan))
                with patch.object(trace,'digest',return_value='new'):
                    with self.assertRaisesRegex(ValueError,'frozen file'):
                        trace.validate(out)

    def test_equivalence_rejects_discrete_changes_and_numeric_drift(self):
        close({'indices':[1,2],'x':1.},{'indices':[1,2],'x':1.+1e-9})
        for a,b in [([1,2],[2,1]),(1.,1.+1e-7),(float('nan'),0.),({'x':1},{'y':1})]:
            with self.assertRaises(ValueError):
                close(a,b)


if __name__ == '__main__':
    unittest.main()
