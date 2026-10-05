"""Local-field coordinate and frozen development-scope guards."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from collection_run import digest,write_json
from robustness_subfield_identity import RID,subfield,validate
from robustness_subfield_wcs import environment,validate as validate_external


class SubfieldTests(unittest.TestCase):
    def test_half_open_selection_preserves_photometry_and_order(self):
        ds=[dict(x=x,y=y,rank=i,flux=10-i)for i,(x,y)in enumerate([(2,2),(5,5),(8,3),(1,4),(4,8)])]
        old=copy.deepcopy(ds);rows,ranks=subfield(ds,[2,2,8,8],limit=40)
        self.assertEqual(ranks,[0,1]);self.assertEqual([r['x']for r in rows],[0,3]);self.assertEqual([r['y']for r in rows],[0,3])
        self.assertEqual([r['flux']for r in rows],[10,9]);self.assertEqual(ds,old)
        self.assertEqual(subfield(ds,[2,2,8,8],limit=1)[1],[0])

    def test_rejects_empty_rectangle(self):
        for box in [[0,0,0,4],[0,4,4,0]]:
            with self.assertRaisesRegex(ValueError,'positive rectangle'):subfield([],box)

    def test_holdout_rejected_before_selection_or_image_access(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);write_json(out/'protocol.json',dict(id=RID,split='holdout',holdout=True))
            with patch('robustness_subfield_identity.selected') as select:
                with self.assertRaisesRegex(ValueError,'development only'):validate(out)
                select.assert_not_called()

    def test_changed_input_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);src=out/'input';src.write_text('before')
            write_json(out/'protocol.json',dict(id=RID,split='development',holdout=False,hashes={str(src):digest(src)}))
            src.write_text('after')
            with patch('robustness_subfield_identity.selected'):
                with self.assertRaisesRegex(ValueError,'frozen input changed'):validate(out)

    def test_external_holdout_rejected_before_hash_access(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            write_json(out/'external-protocol.json',dict(id=RID,split='holdout',holdout=True,regions=[]))
            with patch('robustness_subfield_wcs.validate_local'):
                with self.assertRaisesRegex(ValueError,'scope changed'):validate_external(out)

    def test_external_subprocess_gets_explicit_python_runtime(self):
        with patch.dict('os.environ',{'PATH':'/usr/bin','PYTHONPATH':'/unrelated'},clear=True):
            env=environment(Path('/tools'))
        self.assertTrue(env['PATH'].startswith('/tools/usr/bin:'))
        self.assertTrue(env['PYTHONPATH'].startswith('/tools/usr/lib/python3/dist-packages:'))
        self.assertNotIn('/unrelated',env['PYTHONPATH'])
        self.assertEqual(env['PYTHONDONTWRITEBYTECODE'],'1')
