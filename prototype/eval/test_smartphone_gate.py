import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from smartphone_gate import check_reports, sha256


class SmartphoneGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / 'manifest.json'
        self.catalog = self.root / 'catalog.gz'
        self.catalog.write_bytes(b'catalog')
        self.run = self.root / 'run'
        for subdir in ('per-frame', 'solve-reports'):
            (self.run / subdir).mkdir(parents=True)
        self.baseline = {'schema_version': 1, 'config': {'blind': True, 'exif_hints': True, 'fov_hint_deg': None},
                         'catalog_sha256': sha256(self.catalog), 'frames': {}, 'required_solved': ['a']}
        entries = []
        for frame_id, status in [('a', 'solved'), ('b', 'failed')]:
            source = self.root / f'{frame_id}.jpg'
            source.write_bytes(frame_id.encode())
            digest = sha256(source)
            entries.append({'id': frame_id, 'file': source.name, 'track': 'solver', 'width': 4000,
                            'height': 2252, 'orientation': 6, 'sha256': digest})
            self.baseline['frames'][frame_id] = {'sha256': digest, 'dimensions': [2252, 4000]}
            self.write_status(frame_id, status, digest)
        self.manifest.write_text(json.dumps(entries))
        self.summary = {'generated_by': 'test solver', 'dataset': {'n_selected': 2, 'tracks': ['solver']},
                        'config': {**self.baseline['config'], 'catalog': str(self.catalog)},
                        'solver_track': {'solved': 1}}
        self.save_summary()

    def save_summary(self):
        (self.run / 'summary.json').write_text(json.dumps(self.summary))

    def write_status(self, frame_id, status, digest=None):
        digest = digest or self.baseline['frames'][frame_id]['sha256']
        (self.run / 'per-frame' / f'{frame_id}.json').write_text(json.dumps({'id': frame_id, 'status': status}))
        artifact = {'id': frame_id, 'source_sha256': digest, 'width': 2252, 'height': 4000,
                    'pixel_convention': 'top_left_zero_based', 'report': {'status': status,
                               'pose': {'ra_deg': 0, 'dec_deg': 0, 'roll_deg': 0},
                               'fov': {'fov_x_deg': 50, 'fov_y_deg': 60, 'focal_px': 1200},
                               'quality': {'n_detections': 12, 'n_inliers': 8, 'rms_px': 1, 'log_odds': 30, 'confidence': 1}},
                    'camera': {'width': 2252, 'height': 4000, 'focal_px': 1200, 'k1': 0,
                               'world_to_camera': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}}
        (self.run / 'solve-reports' / f'{frame_id}.json').write_text(json.dumps(artifact))

    def check(self):
        return check_reports(self.manifest, self.baseline, self.run)

    def test_runner_uses_eval_and_fresh_output_directory(self):
        baseline = self.root / 'baseline.json'
        baseline.write_text(json.dumps(self.baseline))
        fake = self.root / 'solver'
        fake.write_text(f"#!{sys.executable}\n" +
                        "import sys, shutil\nfrom pathlib import Path\n" +
                        "assert sys.argv[1] == 'eval', sys.argv\n" +
                        "dest = Path(sys.argv[sys.argv.index('--out-dir') + 1])\n" +
                        f"source = Path({str(self.run)!r})\n" +
                        "for item in source.iterdir():\n" +
                        " if item.is_dir(): shutil.copytree(item, dest / item.name)\n" +
                        " else: shutil.copy2(item, dest / item.name)\n")
        fake.chmod(0o700)
        outputs = self.root / 'outputs'
        command = [sys.executable, str(Path(__file__).with_name('smartphone_gate.py').resolve()),
                   '--manifest', str(self.manifest), '--baseline', str(baseline),
                   '--binary', str(fake), '--out-dir', str(outputs)]
        for _ in range(2):
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        runs = list(outputs.iterdir())
        self.assertEqual(len(runs), 2)
        for run in runs:
            provenance = json.loads((run / 'provenance.json').read_text())
            self.assertEqual(provenance['exit_code'], 0)
            self.assertEqual(provenance['generated_by'], 'test solver')

    def test_experimental_fill_cannot_satisfy_default_gate(self):
        self.summary['config']['sky_fill'] = True
        self.save_summary()
        with self.assertRaisesRegex(ValueError, 'experimental sky fill'):
            self.check()

    def test_experimental_quantile_cannot_satisfy_default_gate(self):
        self.summary['config']['quantile_threshold'] = True
        self.save_summary()
        with self.assertRaisesRegex(ValueError, 'experimental quantile'):
            self.check()
        self.summary['config'].pop('quantile_threshold')
        self.save_summary()
        path = self.run / 'solve-reports' / 'a.json'
        artifact = json.loads(path.read_text())
        artifact['quantile_threshold'] = True
        path.write_text(json.dumps(artifact))
        with self.assertRaisesRegex(ValueError, 'experimental quantile'):
            self.check()

    def test_experimental_statistics_cannot_satisfy_default_gate(self):
        self.summary['config']['sky_statistics'] = True
        self.save_summary()
        with self.assertRaisesRegex(ValueError, 'experimental sky statistics'):
            self.check()

    def test_experimental_mask_cannot_satisfy_default_gate(self):
        self.summary['config']['sky_masks'] = 'masks.json'
        self.save_summary()
        with self.assertRaisesRegex(ValueError, 'experimental sky masks'):
            self.check()
        self.summary['config'].pop('sky_masks')
        self.save_summary()
        path = self.run / 'solve-reports' / 'a.json'
        artifact = json.loads(path.read_text())
        artifact['sky_mask'] = {'id': 'a'}
        path.write_text(json.dumps(artifact))
        with self.assertRaisesRegex(ValueError, 'experimental mask'):
            self.check()

    def test_pass_and_additional_success(self):
        self.assertEqual(self.check(), ['a'])
        self.write_status('b', 'solved')
        self.summary['solver_track']['solved'] = 2
        self.save_summary()
        self.assertEqual(self.check(), ['a', 'b'])

    def test_replacement_success_cannot_hide_lost_frame(self):
        self.write_status('a', 'failed')
        self.write_status('b', 'solved')
        with self.assertRaisesRegex(ValueError, 'previously solved'):
            self.check()

    def test_missing_input_and_stale_reports(self):
        (self.root / 'a.jpg').unlink()
        with self.assertRaises(OSError):
            self.check()

    def test_corrupt_input(self):
        (self.root / 'a.jpg').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'input SHA-256'):
            self.check()

    def test_manifest_membership(self):
        entries = json.loads(self.manifest.read_text())
        self.manifest.write_text(json.dumps(entries[:1]))
        with self.assertRaisesRegex(ValueError, 'membership'):
            self.check()

    def test_wrong_orientation_hash_camera_and_nonfinite(self):
        path = self.run / 'solve-reports/a.json'
        original = json.loads(path.read_text())
        for changes, message in [({'width': 4000, 'height': 2252}, 'dimensions'),
                                 ({'source_sha256': 'wrong'}, 'SHA-256'),
                                 ({'camera': None}, 'missing camera'),
                                 ({'camera': {**original['camera'], 'focal_px': None}}, 'focal'),
                                 ({'camera': {**original['camera'], 'k1': float('nan')}}, 'non-finite')]:
            with self.subTest(message=message):
                path.write_text(json.dumps({**original, **changes}))
                with self.assertRaisesRegex(ValueError, message):
                    self.check()
        path.write_text(json.dumps(original))
        self.check()

    def test_configuration_change(self):
        self.summary['config']['exif_hints'] = False
        self.save_summary()
        with self.assertRaisesRegex(ValueError, 'configuration'):
            self.check()

    def test_missing_report(self):
        (self.run / 'solve-reports/a.json').unlink()
        with self.assertRaises(OSError):
            self.check()


if __name__ == '__main__':
    unittest.main()
