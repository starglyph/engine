"""Guard provenance, pixel-centre geometry and development-only audit inputs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import robustness_coordinate_audit as audit


class CoordinateAuditTests(unittest.TestCase):
    def test_holdout_rejected_before_photograph_or_result_access(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(audit, 'digest', side_effect=AssertionError('must not read')):
                with self.assertRaisesRegex(ValueError, 'outside selected split'):
                    audit.prepare(Path(temp) / 'new', ['wm_r_146925915'])
            self.assertFalse((Path(temp) / 'new').exists())

    def test_tampered_protocol_split_rejected_before_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / 'protocol.json').write_text(json.dumps(dict(split='holdout', holdout=True)))
            with patch.object(audit, 'digest', side_effect=AssertionError('must not read')):
                with self.assertRaisesRegex(ValueError, 'development'):
                    audit.validate(out)

    def test_pixel_area_boundaries_and_centre_survive_rounded_resize(self):
        for size in [(3648, 5472), (3000, 4000), (1989, 1498), (1600, 999), (1, 1)]:
            r = audit.resize_audit(*size)
            self.assertLess(r['geometric_centre_roundtrip_max_abs_px'], 1e-10)
            self.assertLess(r['pixel_area_boundary_max_abs_px'], 1e-10)
        r = audit.resize_audit(3000, 4000)
        self.assertEqual(r['working_size'], [1200, 1600])
        self.assertEqual(r['fixed_centre_seed_offset_px'], [.75, .75])
        self.assertEqual(audit.resize_audit(100, 200)['fixed_centre_seed_offset_px'], [0., 0.])
        with self.assertRaises(ValueError):
            audit.resize_audit(0, 200)

    def test_fixed_shift_does_not_fit_or_drop_empty_edge_groups(self):
        sources = [dict(xy=[20., 30.], outside_both_inputs=True),
                   dict(xy=[40., 50.], outside_both_inputs=False)]
        case = dict(name='k1', projected_xy=[[20., 30.], [40., 50.]])
        r = audit.shift_metrics(sources, case, 100, 100)
        self.assertFalse(r['fit_or_rematch'])
        self.assertAlmostEqual(r['groups']['outside_both_inputs']['shifted_rms_px'], np.sqrt(.5))
        self.assertEqual(r['groups']['left_10pct'], {'count': 0})
        with self.assertRaisesRegex(ValueError, 'invalid saved'):
            audit.shift_metrics(sources, dict(name='k1', projected_xy=[[0, float('nan')]]), 100, 100)

    def test_metadata_export_excludes_location_and_serials(self):
        xmp = b'<x xmlns:c="urn:crop" c:CropLeft="0.1" c:SerialNumber="secret"><c:CropRight>0.9</c:CropRight><c:GPSLatitude>private</c:GPSLatitude></x>'
        result = audit.xmp_geometry(xmp)
        self.assertEqual(result, dict(status='parsed', fields={
            '{urn:crop}CropLeft': ['0.1'], '{urn:crop}CropRight': ['0.9']}))
        self.assertEqual(audit.xmp_geometry(None)['status'], 'absent')
        self.assertEqual(audit.xmp_geometry(b'broken')['status'], 'unparsed')

    def test_normalization_checks_pixels_and_rejects_replaced_original(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, clean = root / '1', root / 'clean.png'
            image = Image.new('RGB', (6, 4), (10, 20, 30))
            exif = Image.Exif()
            exif[274] = 6
            image.save(raw, format='JPEG', exif=exif)
            encoded, pixels, size = audit.normalize(raw.read_bytes())
            clean.write_bytes(encoded)
            record = dict(id='wm_r_1', file=clean.name,
                          orig_sha256=audit.digest(raw), clean_sha256=audit.digest(clean),
                          orig_width=6, orig_height=4, width=size[0], height=size[1],
                          processing=dict(pixel_sha256=pixels, orientation_applied=6))
            with patch.object(audit, 'RAW', root), patch.object(audit, 'SAMPLES', root):
                result = audit.normalization_audit(record)
                self.assertTrue(all(result['checks'].values()))
                self.assertEqual(result['metadata']['normalized_size'], [4, 6])
                self.assertIsNone(result['physical_principal_point'])
                raw.write_bytes(raw.read_bytes() + b'changed')
                with self.assertRaisesRegex(ValueError, 'differs from manifest'):
                    audit.normalization_audit(record)


if __name__ == '__main__':
    unittest.main()
