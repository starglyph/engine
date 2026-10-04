"""Fixed-pair experiments must keep objectives and observations comparable."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from collection_run import digest, write_json
from robustness_pair_refit import build_cases, run
from robustness_pair_report import sensitivity
from robustness_pairs import RID, reviewed


class PairFitTests(unittest.TestCase):
    def inputs(self):
        camera=dict(width=1200,height=1600,focal_px=1190.,k1=0.,ra_deg=270.,dec_deg=0.,roll_deg=0.)
        traces={}
        for arm,n,tier in [('control',22,'deep'),('footprint',11,'default')]:
            initial=dict(camera=camera.copy(),pairs=[dict(world=[1.,0.,0.],xy=[float(i),10.]) for i in range(n)])
            traces[arm]=dict(tiers=[dict(tier=tier,candidates=[dict(chosen=True,stages=[initial,dict(camera=camera.copy())])])])
        points=[dict(id=i,label='not_visible' if i==9 else 'blend' if i==8 else 'visible_source',
                     native_r8_xy=[float(i)+1.,5.],native_r12_xy=[float(i)+2.,5.]) for i in range(11)]
        return traces,points

    def test_matrix_has_identical_prior_and_fixed_pairs_for_both_starts(self):
        traces,points=self.inputs();original=deepcopy(traces)
        cases={c['name']:c for c in build_cases(traces,points)}
        self.assertEqual(len(cases),23)
        for pairs in ['control','footprint']:
            a,b=[cases[f'matrix_{start}_{pairs}'] for start in ['control','footprint']]
            self.assertEqual(a['matches'],b['matches'])
            self.assertEqual(a['prior_weight'],b['prior_weight'])
        self.assertEqual(cases['leave_one_out_9']['matches'],cases['stellar_including_blend_footprint']['matches'])
        self.assertEqual(len(cases['reviewed_single_sources_footprint']['matches']),9)
        self.assertEqual(traces,original)

    def test_too_few_reviewed_pairs_cannot_change_parameter_count(self):
        traces,points=self.inputs()
        for p in points[:3]:p['label']='ambiguous'
        with self.assertRaisesRegex(ValueError,'five-parameter'):
            build_cases(traces,points)

    def test_linear_sensitivity_known_orthogonal_system(self):
        jac=np.vstack((np.eye(5),np.zeros((1,5))))
        probes=np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,0,0,0]],dtype=float)
        result=sensitivity(jac,probes)
        self.assertAlmostEqual(result['column_normalized_data_condition'],1)
        self.assertAlmostEqual(result['probe_amplification'][0],np.sqrt(2))
        self.assertAlmostEqual(result['focal_k1_response_correlation'],0)
        jac[:,4]=0
        with self.assertRaisesRegex(ValueError,'degenerate'):
            sensitivity(jac,probes)

    def test_review_rejects_pending_or_reused_centroids(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            template=dict(split='development',frames=[dict(id=RID,points=[
                dict(id=i,choices=[dict(row=3,x=1.,y=2.)],label='pending',selected_row=None) for i in range(2)])])
            write_json(out/'candidates.json',template);write_json(out/'review.json',template)
            write_json(out/'review-protocol.json',dict(id=RID,split='development',hashes={},candidates_sha256=digest(out/'candidates.json')))
            with self.assertRaisesRegex(ValueError,'complete visual'):
                reviewed(out)
            for p in template['frames'][0]['points']:p.update(label='visible_source',selected_row=3)
            write_json(out/'review.json',template)
            with self.assertRaisesRegex(ValueError,'reused'):
                reviewed(out)

    def test_runner_rejects_holdout_before_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);(out/'protocol.json').write_text(json.dumps(dict(id=RID,split='holdout')))
            with self.assertRaisesRegex(ValueError,'development'):
                run(out)


if __name__=='__main__':
    unittest.main()
