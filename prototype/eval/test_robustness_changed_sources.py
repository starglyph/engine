"""Protect the observational stage mapping and the single manual intervention."""
import copy
import unittest
from unittest.mock import patch

import robustness_changed_sources as review
from robustness_foreground_crossover import drop_foreground


class ChangedSourcesTests(unittest.TestCase):
    def plan(self):
        inp=dict(detections=[dict(x=i,y=i+1) for i in range(40)],
                 verification=dict(detection_indices=[2,19,37,39],matches=[{'world':[1,0,0]}]),
                 camera={'roll_deg':1.},catalog=[[[1,0,0],1.]])
        return dict(id=review.RID,holdout=False,inputs=dict(control=copy.deepcopy(inp),huber=copy.deepcopy(inp)))

    def test_drop_only_one_detection_preserves_fit_and_catalogue(self):
        before=self.plan();saved=copy.deepcopy(before);after=drop_foreground(before)
        self.assertEqual(before,saved)
        for arm in ['control','huber']:
            self.assertEqual(after['inputs'][arm]['detections'],before['inputs'][arm]['detections'][:5]+before['inputs'][arm]['detections'][6:])
            self.assertEqual(after['inputs'][arm]['verification']['detection_indices'],[2,18,36,38])
            for key in ['camera','catalog']:
                self.assertEqual(after['inputs'][arm][key],before['inputs'][arm][key])
            self.assertEqual(after['inputs'][arm]['verification']['matches'],before['inputs'][arm]['verification']['matches'])

    def test_intervention_rejects_holdout_or_initial_fit_change(self):
        for modification in ['holdout','id','initial_match','different_detections']:
            p=self.plan()
            if modification=='holdout':p['holdout']=True
            elif modification=='id':p['id']='wm_r_146925915'
            elif modification=='initial_match':p['inputs']['control']['verification']['detection_indices'].append(5)
            else:p['inputs']['huber']['detections'][0]['x']=999
            with self.assertRaises(ValueError):drop_foreground(p)

    def test_split_guard_precedes_artifact_reads(self):
        with patch.object(review,'selected',side_effect=ValueError('split guard')):
            with self.assertRaisesRegex(ValueError,'split guard'):review.load_inputs()

    def test_rematch_residual_uses_camera_before_that_fit(self):
        inp=dict(camera='initial',verification='v0')
        path=[dict(stage='first_lm',camera='after_first'),
              dict(stage='rematch',radius_px=10.,camera='after10',input='v10'),
              dict(stage='rematch',radius_px=5.,camera='after5',input='v5'),
              dict(stage='final',camera='after5')]
        self.assertEqual(list(review.stages(inp,path)),[
            ('initial',3.,'initial','v0'),('rematch_10.0',10.,'after_first','v10'),
            ('rematch_5.0',5.,'after10','v5')])


if __name__=='__main__':
    unittest.main()
