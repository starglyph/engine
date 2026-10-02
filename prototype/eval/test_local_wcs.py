import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image

from compare_local_wcs import compare, project
from solve_local_wcs import sha256, solve_one
from verify_masked_wcs import verify, reviewed_selection, restore_full_frame


class LocalWcsTests(unittest.TestCase):
    def test_crop_restoration_preserves_sip_world_and_correspondence_residuals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            header = fits.Header({'CTYPE1': 'RA---TAN-SIP', 'CTYPE2': 'DEC--TAN-SIP',
                                  'CRVAL1': 300., 'CRVAL2': 10., 'CRPIX1': 800., 'CRPIX2': 900.,
                                  'CD1_1': -.02, 'CD1_2': .001, 'CD2_1': .001, 'CD2_2': -.02,
                                  'A_ORDER': 2, 'B_ORDER': 2, 'A_2_0': 1e-5, 'B_0_2': -2e-5})
            native = root/'field.wcs'
            fits.PrimaryHDU(header=header).writeto(native)
            corr = root/'field.corr'
            fits.BinTableHDU.from_columns([
                fits.Column(name='field_x', format='D', array=[10.]),
                fits.Column(name='field_y', format='D', array=[20.]),
                fits.Column(name='index_x', format='D', array=[12.]),
                fits.Column(name='index_y', format='D', array=[23.])]).writeto(corr)
            path, translated = restore_full_frame(native, corr, (500, 200, 2100, 2500), 2252, 4000)
            local = np.array([[0., 0.], [799., 899.], [1599., 2299.]])
            np.testing.assert_allclose(WCS(fits.getheader(path)).all_pix2world(local+[500, 200], 0),
                                       WCS(header).all_pix2world(local, 0), atol=1e-10)
            data = fits.getdata(translated)
            self.assertEqual(data['field_x'][0], 510.)
            self.assertEqual(data['field_y'][0], 220.)
            self.assertEqual(data['index_x'][0]-data['field_x'][0], 2.)
            self.assertEqual(data['index_y'][0]-data['field_y'][0], 3.)
            self.assertEqual(fits.getdata(corr)['field_x'][0], 10.)

    def test_reviewed_sources_are_bound_to_external_rows_and_pixels(self):
        data = np.array([(1., 1.), (4., 3.), (8., 2.)], dtype=[('X', float), ('Y', float)])
        reference = {'source_sha256': 'source', 'width': 10, 'height': 5}
        review = {**reference, 'extraction_sha256': 'extraction', 'points': [
            {'axy_row': 0, 'x': 0., 'y': 0., 'label': 'visible_source'},
            {'axy_row': 1, 'x': 3., 'y': 2., 'label': 'ambiguous_texture'}]}
        sky = np.array([True, True, False])
        np.testing.assert_array_equal(reviewed_selection(data, review, reference, 'extraction', sky),
                                      [True, False, False])
        with self.assertRaisesRegex(ValueError, 'extraction SHA-256'):
            reviewed_selection(data, review, reference, 'changed', sky)
        review['points'][0]['x'] = 1.
        with self.assertRaisesRegex(ValueError, 'coordinates mismatch'):
            reviewed_selection(data, review, reference, 'extraction', sky)
        review['points'][0]['x'] = 0.
        with self.assertRaisesRegex(ValueError, 'outside sky'):
            reviewed_selection(data, review, reference, 'extraction', np.zeros(3, dtype=bool))
        review['points'].append(dict(review['points'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            reviewed_selection(data, review, reference, 'extraction', sky)

    def test_masked_external_extraction_preserves_pixels_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'photo.jpg'
            source.write_bytes(b'original source')
            raw = root / 'raw'
            attempt = raw / 'photo/downsample-4'
            attempt.mkdir(parents=True)
            normalized = raw / 'photo/input.fits'
            fits.writeto(normalized, np.ones((4, 4)))
            primary = fits.PrimaryHDU()
            primary.header['IMAGEW'] = primary.header['IMAGEH'] = 4
            table = fits.BinTableHDU.from_columns([
                fits.Column(name='X', format='E', array=[1, 1, 4, 2, 2]),
                fits.Column(name='Y', format='E', array=[1, 3, 1, 2, 2.5]),
                fits.Column(name='FLUX', format='E', array=[5, 4, 3, 2, 1])])
            fits.HDUList([primary, table]).writeto(attempt / 'field.axy')
            reference = {'source': str(source), 'source_sha256': sha256(source), 'width': 4, 'height': 4,
                         'pixel_convention': 'top_left_zero_based', 'input_fits_sha256': sha256(normalized)}
            mask = {'source_sha256': sha256(source), 'width': 4, 'height': 4,
                    'sky_polygon': [[0,0],[1,0],[1,0.5],[0,0.5]]}
            executable = root / 'no-solution'
            executable.write_text('#!/bin/sh\nexit 0\n'); executable.chmod(0o700)
            args = SimpleNamespace(solve_field=str(executable), config=root/'config', cpu_limit=1)
            output = root/'output'
            stale = output/'photo/downsample-4'
            stale.mkdir(parents=True)
            (stale/'field.solved').write_bytes(b'\x01')
            (stale/'field.wcs').write_text('stale')
            result = verify(reference, mask, raw, output, args)
            self.assertEqual(result['status'], 'unresolved')
            self.assertEqual(result['attempts'][0]['n_in_sky'], 3)
            np.testing.assert_array_equal(fits.getdata(stale/'sky.xyls')['Y'], [1,1,2])
            self.assertFalse((stale/'field.wcs').exists())
            mask['sky_polygon'] = [[0,0],[0.01,0],[0,0.01]]
            self.assertEqual(verify(reference, mask, raw, output, args)['attempts'][0]['n_in_sky'], 0)
            normalized.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'normalized FITS SHA-256'):
                verify(reference, mask, raw, output, args)

    def test_tan_reference_uses_top_down_zero_based_pixels_and_ra_wrap(self):
        width, height, focal = 1000, 600, 900
        camera = {"width": width, "height": height, "focal_px": focal, "k1": 0,
                  "world_to_camera": [[0, -1, 0], [0, 0, 1], [1, 0, 0]]}
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [width / 2 + 1, height / 2 + 1]
        wcs.wcs.crval = [0, 0]
        wcs.wcs.cd = np.diag([-180 / np.pi / focal] * 2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        pixels = np.array([[0, 0], [width - 1, height - 1], [100, 500], [500, 300]])
        world = wcs.all_pix2world(pixels, 0)
        self.assertGreater(world[1, 0], 300)  # RA wraps through zero.
        np.testing.assert_allclose(project(camera, world), pixels, atol=1e-9)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.wcs"
            fits.PrimaryHDU(header=wcs.to_header()).writeto(path)
            reference = {"source_sha256": "source", "width": width, "height": height,
                         "pixel_convention": "top_left_zero_based", "status": "solved_candidate",
                         "review_status": "pending", "wcs_file": str(path), "wcs_sha256": sha256(path),
                         "reference_points": [{"x": x, "y": y} for x, y in pixels]}
            artifact = {"source_sha256": "source", "width": width, "height": height,
                        "pixel_convention": "top_left_zero_based", "report": {"status": "solved"},
                        "camera": camera}
            result = compare(reference, artifact)
            self.assertEqual(result["review_status"], "pending")
            self.assertLess(result["full_field_wcs_difference"]["rms_px"], 1e-9)
            artifact["source_sha256"] = "another image"
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                compare(reference, artifact)
            artifact["source_sha256"] = "source"
            artifact["width"] += 1
            with self.assertRaisesRegex(ValueError, "width mismatch"):
                compare(reference, artifact)

    def test_rerun_cannot_accept_stale_wcs_and_preserves_sensor_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sensor.tiff"
            pixels = np.full((32, 48), 1000, dtype=np.uint16)
            pixels[10, 20] = 4000
            Image.fromarray(pixels).save(source)
            executable = root / "no-solution"
            executable.write_text("#!/bin/sh\nexit 0\n")
            executable.chmod(0o700)
            attempt = root / "output/sensor/downsample-4"
            attempt.mkdir(parents=True)
            (attempt / "field.solved").write_bytes(b"\x01")
            (attempt / "field.wcs").write_text("old result")
            args = SimpleNamespace(solve_field=str(executable), config=None, cpu_limit=1)
            result = solve_one(source, root / "output", args)
            self.assertEqual(result["status"], "unresolved")
            self.assertFalse((attempt / "field.wcs").exists())
            np.testing.assert_array_equal(fits.getdata(root / "output/sensor/input.fits"), pixels)
            json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
