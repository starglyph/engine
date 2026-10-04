"""Lock the diagnostic's split, training membership, nesting and radial guard."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from collection_run import ROOT
from compare_local_wcs import project
from robustness_radial_fit import IDS, build_frames, optimizer_source, project_extended, radial_sensor_coverage, validate


class RadialFitTests(unittest.TestCase):
    def fixture(self):
        geometry=dict(split='development',frames=[])
        reviews=dict(frames=[])
        reports={}
        for rid,ntrain,ntest in zip(IDS,[17,26,19],[18,14,23]):
            geometry['frames'].append(dict(id=rid,sources=[dict(source_id=i,hyg_id=i+1,name=str(i),
                xy=[i+.2,i+1.],outside_both_inputs=i>=ntrain) for i in range(ntrain+ntest)]))
            reviews['frames'].append(dict(id=rid,points=[dict(id=i,hyg_id=i+1,ra_deg=240.+i*.1,dec_deg=-5.)
                for i in range(ntrain+ntest)]))
            reports[rid]=dict(camera=dict(k1=.1))
        return geometry,reviews,reports

    def test_existing_membership_keeps_training_and_scoring_disjoint(self):
        args=self.fixture(); before=copy.deepcopy(args)
        frames=build_frames(*args)
        self.assertEqual(args,before)
        self.assertEqual([sum(not s['outside_both_inputs'] for s in f['sources']) for f in frames],[17,26,19])
        self.assertEqual([sum(s['outside_both_inputs'] for s in f['sources']) for f in frames],[18,14,23])
        for f,g in zip(frames,args[0]['frames']):
            self.assertEqual([s['xy'] for s in f['sources']],[s['xy'] for s in g['sources']])
            np.testing.assert_allclose(np.linalg.norm([s['world'] for s in f['sources']],axis=1),1.)

    def test_changed_membership_or_identity_rejected(self):
        args=self.fixture(); args[0]['frames'][0]['sources'][0]['outside_both_inputs']=True
        with self.assertRaisesRegex(ValueError,'membership'):build_frames(*args)
        args=self.fixture();args[1]['frames'][0]['points'][0]['hyg_id']=999
        with self.assertRaisesRegex(ValueError,'identity'):build_frames(*args)

    def test_holdout_rejected_before_hash_access(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            (p/'protocol.json').write_text(json.dumps(dict(split='holdout',holdout=True,ids=['wm_r_146925915'])))
            with patch('robustness_radial_fit.digest',side_effect=AssertionError('must not read')):
                with self.assertRaisesRegex(ValueError,'development'):validate(p)

    def test_optimizer_preserves_loop_and_rejects_source_drift(self):
        src=(ROOT/'prototype/crates/starglyph-core/src/solve.rs').read_text()
        generated=optimizer_source(src)
        self.assertIn('for _ in 0..30',generated)
        self.assertIn('for _ in 0..10',generated)
        self.assertIn('improvement < 1e-9',generated)
        self.assertIn('trial[5] = trial[5].clamp(-0.5, 0.5)',generated)
        with self.assertRaisesRegex(ValueError,'optimizer changed'):
            optimizer_source(src.replace('for _ in 0..30','for _ in 0..40'))

    def test_nested_projection_and_radial_sign(self):
        camera=dict(world_to_camera=[[0,1,0],[0,0,1],[1,0,0]],width=100,height=80,focal_px=90.,k1=.03,k2=0.)
        points=[[1.,2.],[15.,-7.]]
        np.testing.assert_array_equal(project(camera,points),project_extended(camera,points))
        a=project(camera,points)-[50,40]
        camera['k2']=.1
        b=project_extended(camera,points)-[50,40]
        self.assertTrue(np.all(np.linalg.norm(b,axis=1)>np.linalg.norm(a,axis=1)))
        np.testing.assert_allclose(a[:,0]*b[:,1]-a[:,1]*b[:,0],0,atol=1e-10)

    def test_sensor_coverage_uses_first_turn_not_later_branch(self):
        camera=dict(k1=-1.,k2=.1,focal_px=100.,width=100,height=100)
        result=radial_sensor_coverage(camera)
        self.assertFalse(result['covers_sensor'])
        self.assertLess(result['first_turning_radius'],1.)
        camera.update(k1=0.,k2=.1)
        self.assertTrue(radial_sensor_coverage(camera)['covers_sensor'])
        camera.update(k1=-.001,k2=0.)
        self.assertTrue(radial_sensor_coverage(camera)['covers_sensor'])


if __name__=='__main__':unittest.main()
