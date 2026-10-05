"""Freeze the complete shared rematch rather than selecting easy residuals."""
import copy
import unittest
from unittest.mock import patch

import robustness_remaining_sources as audit


class RemainingSourcesTests(unittest.TestCase):
    def cases(self):
        ids=[i for i in range(32) if i!=5]
        v=dict(original_detection_indices=ids,matches=[dict(world=[float(i),0.,1.],xy=[float(i),2.]) for i in ids])
        return [dict(pose_arm=a,match_arm=b,after=dict(steps=[{},dict(stage='rematch',radius_px=10.,input=copy.deepcopy(v))])) for a in ['control','huber'] for b in ['control','huber']]

    def test_all_four_sets_required_but_pair_order_may_differ(self):
        cases=self.cases();v=cases[3]['after']['steps'][1]['input']
        v['matches'].reverse();v['original_detection_indices'].reverse()
        self.assertIs(audit.common_stage(cases),cases[0]['after'])

    def test_rejects_changed_identity_selection_or_stage(self):
        for change in ['world','foreground','radius','length']:
            cases=self.cases();step=cases[2]['after']['steps'][1];v=step['input']
            if change=='world':v['matches'][0]['world'][0]=999.
            elif change=='foreground':v['original_detection_indices'][0]=5
            elif change=='radius':step['radius_px']=5.
            else:v['matches'].pop()
            with self.assertRaises(ValueError):audit.common_stage(cases)

    def test_split_guard_precedes_prior_trace_reads(self):
        with patch.object(audit,'selected',side_effect=ValueError('split guard')):
            with self.assertRaisesRegex(ValueError,'split guard'):audit.load_stage()

    def test_review_cannot_move_source_or_change_catalogue_id(self):
        original=dict(split='development',frames=[dict(id=audit.RID,points=[dict(id=0,hyg_id=123,review_center=[1.,2.],label='pending',selected_row=None)])])
        for key,value in [('hyg_id',124),('review_center',[2.,3.])]:
            changed=copy.deepcopy(original);changed['frames'][0]['points'][0][key]=value
            with self.assertRaises(ValueError):audit.validate_review(original,changed)


if __name__=='__main__':
    unittest.main()
