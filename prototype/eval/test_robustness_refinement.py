"""Research replay safeguards and scale-correct geometry attribution."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from robustness_refinement import run
from robustness_refinement_report import compact_report, compare_saved, lift, summarize_stage


class RefinementReplayTests(unittest.TestCase):
    def camera(self):
        return dict(width=100,height=80,focal_px=60.,k1=0.,world_to_camera=np.eye(3).tolist())

    def test_lift_preserves_pixel_centers_and_identity_scale(self):
        camera=self.camera()
        np.testing.assert_allclose(lift([[0,0],[99,79]],camera,200,160),[[.5,.5],[198.5,158.5]])
        np.testing.assert_allclose(lift([[12,25]],camera,100,80),[[12,25]])

    def test_stage_exposes_incorrect_identity_near_reviewed_source(self):
        camera=self.camera()
        source=dict(id=4,hyg_id=20,ra_deg=0.,dec_deg=90.,xy=[100.5,80.5],outside_both_inputs=True)
        raw=dict(stage='test',camera=camera,pairs=[dict(hyg_ids=[30],xy=[50,40],projected_xy=[50,40])])
        result=summarize_stage(raw,[source],200,160)
        self.assertLess(result['source_errors_px'][0],1e-12)
        self.assertFalse(result['fit_pairs'][0]['reviewed_identity_agrees'])
        self.assertEqual(result['groups']['outside_both_inputs']['count'],1)
        self.assertEqual(result['groups']['left_10pct'],dict(count=0))

    def test_reproduction_rejects_wrong_variant_even_if_successful(self):
        camera=self.camera()
        tier=dict(width=100,height=80,detections=[dict(x=30.,y=20.)])
        candidate=dict(stages=[dict(camera=camera,pairs=[{}])])
        saved=dict(camera=camera.copy(),width=100,height=80,
                   report=dict(detections=[dict(x=30.,y=20.)],quality=dict(n_inliers=1)))
        self.assertEqual(compare_saved(tier,candidate,saved)['status'],'passed')
        saved['camera']['k1']=.01
        with self.assertRaisesRegex(ValueError,'camera differs'):
            compare_saved(tier,candidate,saved)

    def test_run_rejects_holdout_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            (out/'protocol.json').write_text(json.dumps(dict(split='holdout',id='wm_r_146925915')))
            with self.assertRaisesRegex(ValueError,'development'):
                run(out)

    def test_compact_report_retains_all_fit_pairs_and_errors(self):
        stage=dict(projected_xy=[[1,2]],source_errors_px=[1.23456789],
                   fit_pairs=[dict(hyg_ids=[15,16],xy=[2.3,4.5],residual_px=1.1)])
        original=dict(candidates=[dict(stages=[stage])])
        result=compact_report(original)['candidates'][0]['stages'][0]
        self.assertEqual(result['fit_pairs'],[[[15,16],2.3,4.5,1.1]])
        self.assertEqual(result['source_errors_px'],[1.234568])
        self.assertIn('projected_xy',stage)


if __name__ == '__main__':
    unittest.main()
