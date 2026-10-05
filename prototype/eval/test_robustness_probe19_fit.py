"""Protect the common nonlinear objective and exact prior controls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import robustness_probe19_fit as fit


class Probe19FitTests(unittest.TestCase):
    def fixture(self):
        _,probes,alternate=fit.inputs()
        saved=fit.read(fit.prior('30-results'));old=fit.read(fit.prior('30-input'))
        ids=fit.read(fit.prior('30-protocol'))['group_probe_ids']['common_unused']
        return saved,old,probes,alternate,ids

    def test_identical19_targets_and_prior_across_both_starts(self):
        args=self.fixture();before=copy.deepcopy(args)
        result=fit.build_cases(*args)
        self.assertEqual(args,before)
        self.assertEqual(len(result['cases']),12)
        for i in (0,4,8):
            a,b=result['cases'][i+1],result['cases'][i+3]
            self.assertEqual(a['matches'],b['matches'])
            self.assertEqual(a['prior_weight'],b['prior_weight'])
            self.assertEqual(len(a['matches']),19)
            self.assertEqual([m['source_id'] for m in a['matches']],args[-1])
            self.assertNotEqual(a['initial'],b['initial'])

    def test_control_retains_old_inputs_and_expects_saved_final_camera(self):
        args=self.fixture();result=fit.build_cases(*args)
        saved={c['name']:c for c in args[0]['cases']};old={c['name']:c for c in args[1]['cases']}
        for control in result['cases'][::2]:
            name=control['name'].removeprefix('control_')
            self.assertEqual(control['initial'],old[name]['initial'])
            self.assertEqual(control['matches'],old[name]['matches'])
            self.assertEqual(control['expected'],saved[name]['camera'])
            self.assertEqual(control['prior_weight'],saved[name]['prior_weight'])

    def test_fit_input_overlap_or_different_prior_rejected(self):
        saved,old,probes,alt,ids=self.fixture()
        used={m['hyg_id'] for c in old['cases'] for m in c['matches']}
        overlap=next(p['source_id'] for p in probes if p['hyg_id'] in used)
        with self.assertRaisesRegex(ValueError,'overlaps'):
            fit.build_cases(saved,old,probes,alt,[overlap]+ids[1:])
        next(c for c in saved['cases'] if c['name']=='coverage22_native_r8')['prior_weight']+=1
        with self.assertRaisesRegex(ValueError,'objectives'):
            fit.build_cases(saved,old,probes,alt,ids)

    def test_holdout_guard_precedes_prior_artifact_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            (out/'protocol.json').write_text(json.dumps(dict(id=fit.RID,split='holdout',holdout=True)))
            with patch.object(fit,'validate30',side_effect=AssertionError('must not read')):
                with self.assertRaisesRegex(ValueError,'development'):fit.validate(out)


if __name__=='__main__':unittest.main()
