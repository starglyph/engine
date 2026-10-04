"""Separate an initial-pair edit from persistent physical-source removal."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from robustness_rematch import build_cases, validate
from robustness_rematch_report import assert_same_stages, reused_pairs


class RematchTests(unittest.TestCase):
    def inputs(self):
        pairs=[dict(hyg_ids=[5],world=[1,0,0],xy=[1.,2.]),dict(hyg_ids=[80761],world=[0,1,0],xy=[3.,4.])]
        detections=[dict(x=1.,y=2.,rank=0),dict(x=3.,y=4.,rank=1),dict(x=5.,y=6.,rank=2)]
        leaf=dict(hyg_id=80761,label='not_visible',detector_xy_working=[3.,4.])
        return dict(stages=[dict(pairs=pairs)]),detections,leaf

    def test_cases_preserve_other_sources_and_distinguish_reentry(self):
        c,d,leaf=self.inputs();original=deepcopy((c,d,leaf))
        cases,index=build_cases(c,d,leaf)
        self.assertEqual(index,1)
        self.assertEqual(cases[0]['matches'],c['stages'][0]['pairs'])
        self.assertEqual(cases[0]['detections'],cases[1]['detections'])
        self.assertEqual(cases[1]['matches'],cases[2]['matches'])
        self.assertEqual(cases[2]['detections'],[d[0],d[2]])
        self.assertEqual((c,d,leaf),original)
        cases[0]['detections'][0]['x']=20
        self.assertEqual(d[0]['x'],1.)

    def test_wrong_or_duplicate_detection_is_rejected(self):
        c,d,leaf=self.inputs()
        for bad in [[d[0]],d+[d[1]]]:
            with self.assertRaisesRegex(ValueError,'identity mismatch'):build_cases(c,bad,leaf)
        leaf['label']='visible_source'
        with self.assertRaisesRegex(ValueError,'visually rejected'):build_cases(c,d,leaf)

    def test_reuse_tracks_physical_source_even_if_catalogue_id_changes(self):
        stage=dict(camera=dict(width=1200,height=1600),pairs=[dict(hyg_ids=[95652],xy=[10.,20.])])
        self.assertEqual(reused_pairs(stage,[25.75,50.75]),[[95652]])
        self.assertEqual(reused_pairs(stage,[26.,51.]),[])

    def test_baseline_verifies_intermediate_stage_not_only_final(self):
        camera=dict(width=1200,height=1600,focal_px=1000.,k1=0.,world_to_camera=[[1,0,0],[0,1,0],[0,0,1]])
        stages=[dict(stage=n,camera=deepcopy(camera),pairs=[]) for n in ['initial_refine','original_final']]
        changed=deepcopy(stages);assert_same_stages(changed,stages)
        changed[0]['camera']['k1']=.01
        with self.assertRaisesRegex(ValueError,'initial_refine'):assert_same_stages(changed,stages)

    def test_validate_rejects_wrong_split_before_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);(out/'protocol.json').write_text(json.dumps(dict(id='wm_r_143159342',split='holdout')))
            with self.assertRaisesRegex(ValueError,'development'):validate(out)

    def test_baseline_rejects_lost_identity_and_moved_source(self):
        camera=dict(width=1200,height=1600,focal_px=1000.,k1=0.,world_to_camera=[[1,0,0],[0,1,0],[0,0,1]])
        stages=[dict(stage='candidate',camera=camera,
                     pairs=[dict(hyg_ids=[80761],world=[1.,0.,0.],xy=[10.,20.])])]
        for key,value in [('hyg_ids',[]),('xy',[10.,20.01]),('world',[0.,1.,0.])]:
            changed=deepcopy(stages);changed[0]['pairs'][0][key]=value
            with self.assertRaisesRegex(ValueError,'baseline pairs differ'):assert_same_stages(changed,stages)


if __name__=='__main__':unittest.main()
