import json
from pathlib import Path
import subprocess
import sys
import unittest

import test_smartphone_gate
from smartphone_gate import check_reports, sha256


class SkyFillGateTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_smartphone_gate.SmartphoneGateTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.default_baseline = json.loads(json.dumps(f.baseline))
        self.mask = {'id': 'a', 'source_sha256': f.baseline['frames']['a']['sha256'],
                     'width': 2252, 'height': 4000, 'sky_polygon': [[0, 0], [1, 0], [1, 1], [0, 1]]}
        self.masks = f.root/'masks.json'
        self.masks.write_text(json.dumps({'schema_version': 1,
            'coordinates': 'exif_oriented_normalized_image_edges', 'masks': [self.mask]}))
        f.baseline.update(mode='sky_fill', masks_sha256=sha256(self.masks), mask_ids=['a'])
        f.baseline['config'].update(sky_fill=True, sky_statistics=True)
        f.summary['config'].update(sky_fill=True, sky_statistics=True, sky_masks=str(self.masks))
        f.save_summary()
        self.decorate()

    def decorate(self):
        for frame_id in ('a', 'b'):
            self.edit(frame_id, {'sky_mask': self.mask if frame_id == 'a' else None,
                                 'sky_fill': frame_id == 'a', 'sky_statistics': frame_id == 'a'})

    def edit(self, frame_id, changes):
        path = self.fixture.run/'solve-reports'/f'{frame_id}.json'
        data = json.loads(path.read_text()); data.update(changes)
        path.write_text(json.dumps(data))

    def check(self):
        f = self.fixture
        return check_reports(f.manifest, f.baseline, f.run, sky_fill_masks=self.masks)

    def test_modes_cannot_satisfy_each_others_baselines(self):
        f = self.fixture
        self.assertEqual(self.check(), ['a'])
        with self.assertRaisesRegex(ValueError, 'experimental baseline'):
            check_reports(f.manifest, f.baseline, f.run)
        with self.assertRaisesRegex(ValueError, 'requires its experimental baseline'):
            check_reports(f.manifest, self.default_baseline, f.run, sky_fill_masks=self.masks)
        with self.assertRaisesRegex(ValueError, 'experimental sky fill'):
            check_reports(f.manifest, self.default_baseline, f.run)
        f.summary['config'].pop('sky_fill'); f.save_summary()
        with self.assertRaisesRegex(ValueError, 'both experimental flags'):
            self.check()

    def test_swapped_success_cannot_hide_lost_required_frame(self):
        f = self.fixture
        f.write_status('a', 'failed'); f.write_status('b', 'solved'); self.decorate()
        with self.assertRaisesRegex(ValueError, 'previously solved frames lost'):
            self.check()

    def test_new_success_allowed(self):
        f = self.fixture
        f.write_status('b', 'solved'); self.decorate()
        f.summary['solver_track']['solved'] = 2; f.save_summary()
        self.assertEqual(self.check(), ['a', 'b'])

    def test_pinned_masks_and_per_frame_metadata(self):
        f = self.fixture
        original = self.masks.read_bytes()
        self.masks.write_bytes(original+b' ')
        with self.assertRaisesRegex(ValueError, 'mask file SHA-256'):
            self.check()
        self.masks.write_bytes(original)
        self.edit('a', {'sky_mask': None})
        with self.assertRaisesRegex(ValueError, 'mask provenance'):
            self.check()
        self.decorate(); self.edit('a', {'sky_statistics': False})
        with self.assertRaisesRegex(ValueError, 'per-frame experimental flags'):
            self.check()
        self.decorate(); self.edit('b', {'sky_fill': True})
        with self.assertRaisesRegex(ValueError, 'per-frame experimental flags'):
            self.check()
        self.decorate(); self.check()

    def test_corrupt_and_missing_artifacts_fail(self):
        f = self.fixture
        self.edit('a', {'source_sha256': 'wrong'})
        with self.assertRaisesRegex(ValueError, 'report SHA-256'):
            self.check()
        self.edit('a', {'source_sha256': self.mask['source_sha256'], 'camera': None})
        with self.assertRaisesRegex(ValueError, 'missing camera'):
            self.check()
        (f.run/'solve-reports'/'a.json').unlink()
        with self.assertRaises(OSError):
            self.check()

    def test_runner_explicit_flags_fresh_directories_and_provenance(self):
        f = self.fixture
        baseline = f.root/'baseline.json'; baseline.write_text(json.dumps(f.baseline))
        fake = f.root/'solver'
        fake.write_text(f'#!{sys.executable}\n' +
            'import sys, shutil\nfrom pathlib import Path\n' +
            "assert sys.argv[1] == 'eval'\n" +
            "assert '--sky-fill' in sys.argv and '--sky-statistics' in sys.argv\n" +
            f"assert Path(sys.argv[sys.argv.index('--sky-masks')+1]) == Path({str(self.masks)!r})\n" +
            "dest = Path(sys.argv[sys.argv.index('--out-dir')+1])\n" +
            f'source = Path({str(f.run)!r})\n' +
            'for item in source.iterdir():\n' +
            ' if item.is_dir(): shutil.copytree(item, dest/item.name)\n' +
            ' else: shutil.copy2(item, dest/item.name)\n')
        fake.chmod(0o700)
        outputs = f.root/'outputs'
        command = [sys.executable, str(Path(__file__).with_name('sky_fill_gate.py').resolve()),
                   '--manifest', str(f.manifest), '--baseline', str(baseline), '--masks', str(self.masks),
                   '--binary', str(fake), '--out-dir', str(outputs)]
        for _ in range(2):
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn('SKY-FILL GATE PASS', result.stdout)
        runs = list(outputs.iterdir()); self.assertEqual(len(runs), 2)
        for run in runs:
            provenance = json.loads((run/'provenance.json').read_text())
            self.assertEqual(provenance['mode'], 'sky_fill')
            self.assertEqual(provenance['masks_sha256'], sha256(self.masks))
            self.assertEqual(provenance['exit_code'], 0)


if __name__ == '__main__':
    unittest.main()
