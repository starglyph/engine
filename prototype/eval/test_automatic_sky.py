import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw

from automatic_sky_mask import estimate, generate, skyline_polygon
from automatic_sky_compare import compare, compare_reference, mask_overlap, rasterize, reference_inlier_recovery, same_annotation, transitions, validate_artifact
from automatic_sky_experiment import write_json
from smartphone_gate import sha256
import test_smartphone_gate


class SkylineTests(unittest.TestCase):
    def test_flat_horizon_excludes_bright_land_without_losing_upper_sky(self):
        image = Image.new('RGB', (160, 100), (10, 20, 30))
        ImageDraw.Draw(image).rectangle((0, 60, 159, 99), fill=(150, 80, 50))
        polygon, diagnostics = estimate(image)
        sky = rasterize(polygon, 160, 100)
        self.assertTrue(sky[:55].all())
        self.assertFalse(sky[60:].any())
        self.assertEqual(diagnostics['decision'], 'mask')

    def test_isolated_stars_do_not_cut_columns_and_no_horizon_keeps_whole_frame(self):
        image = Image.new('RGB', (160, 100), (10, 20, 30))
        draw = ImageDraw.Draw(image)
        for x in (15, 40, 80, 130):
            draw.point((x, 40), fill='white')
        polygon, _ = estimate(image)
        self.assertTrue(rasterize(polygon, 160, 100).all())

    def test_tree_extending_above_horizon_is_excluded(self):
        image = Image.new('RGB', (160, 100), (30, 40, 50))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 85, 159, 99), fill=(150, 80, 50))
        draw.rectangle((60, 40, 90, 99), fill=(2, 2, 2))
        polygon, _ = estimate(image)
        sky = rasterize(polygon, 160, 100)
        self.assertTrue(sky[25, 75])
        self.assertFalse(sky[45, 75])
        self.assertTrue(sky[70, 20])

    def test_tiny_selected_area_abstains(self):
        image = Image.new('RGB', (160, 100), (140, 80, 50))
        ImageDraw.Draw(image).rectangle((0, 0, 159, 8), fill=(10, 20, 30))
        polygon, diagnostics = estimate(image)
        self.assertIsNone(polygon)
        self.assertEqual(diagnostics['decision'], 'abstain_small_region')

    def test_heterogeneous_top_abstains(self):
        image = Image.new('RGB', (160, 100), (10, 20, 30))
        ImageDraw.Draw(image).rectangle((80, 0, 159, 99), fill=(180, 180, 180))
        polygon, diagnostics = estimate(image)
        self.assertIsNone(polygon)
        self.assertEqual(diagnostics['decision'], 'abstain_heterogeneous_seed')

    def test_zero_columns_have_no_selected_pixel_centers(self):
        polygon = skyline_polygon([0, 0, 50, 100, 0], 100)
        self.assertFalse(rasterize(polygon, 5, 100)[:, [0, 1, 4]].any())

    def test_repeated_estimation_is_identical(self):
        image = Image.new('RGB', (200, 100), (12, 18, 25))
        self.assertEqual(estimate(image), estimate(image))

    def test_exif_orientation_hash_and_no_frame_id_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = Image.new('RGB', (100, 60), (10, 20, 30))
            exif = image.getexif()
            exif[274] = 6
            image.save(root/'photo.jpg', exif=exif)
            entry = {'id': 'a', 'file': 'photo.jpg', 'sha256': sha256(root/'photo.jpg'),
                     'width': 100, 'height': 60, 'orientation': 6}
            masks, _ = generate(root/'manifest.json', [entry])
            annotation = masks['masks'][0]
            self.assertEqual((annotation['width'], annotation['height']), (60, 100))
            other, _ = generate(root/'manifest.json', [{**entry, 'id': 'unseen'}])
            self.assertEqual(annotation['sky_polygon'], other['masks'][0]['sky_polygon'])
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                generate(root/'manifest.json', [{**entry, 'sha256': 'wrong'}])
            with self.assertRaisesRegex(ValueError, 'dimensions mismatch'):
                generate(root/'manifest.json', [{**entry, 'orientation': 1}])


class ComparisonTests(unittest.TestCase):
    def test_reference_inlier_recovery_scales_matches_once_and_keeps_top_k_separate(self):
        reference = {'source_sha256': 'hash', 'width': 400, 'height': 200,
                     'report': {'detections': [{'x': 101.5, 'y': 81.5, 'inlier': True},
                                               {'x': 102.5, 'y': 81.5, 'inlier': True},
                                               {'x': 201.5, 'y': 81.5, 'inlier': False}]}}
        candidate = {**reference, 'detection_diagnostics': [{'width': 100, 'height': 50, 'tier': 'deep',
            'max_detections': 1, 'result': {'detections': [{'x': 25, 'y': 20, 'rank': 2}, {'x': 50, 'y': 20, 'rank': 0}]}}]}
        row = reference_inlier_recovery(reference, candidate)['tiers'][0]
        self.assertEqual((row['reference_inliers'], row['matched_before_top_k'], row['retained']), (2, 1, 0))
        self.assertEqual(row['ranks_by_reference_inlier'], [2, None])
        with self.assertRaisesRegex(ValueError, 'recovery image mismatch'):
            reference_inlier_recovery(reference, {**candidate, 'source_sha256': 'other'})

    def test_json_float_roundtrip_is_allowed_but_polygon_changes_are_not(self):
        original = {'id': 'a', 'sky_polygon': [[.7953125, .23333333333333334]]}
        roundtripped = {'id': 'a', 'sky_polygon': [[.7953125, .23333333333333336]]}
        self.assertTrue(same_annotation(original, roundtripped))
        self.assertFalse(same_annotation(original, {**roundtripped, 'id': 'b'}))
        self.assertFalse(same_annotation(original, {**original, 'sky_polygon': [[.7953125, .23333334]]}))
        self.assertFalse(same_annotation(original, {**original, 'sky_polygon': [[float('nan'), .23]]}))

    def test_overlap_and_abstention_are_not_counted_as_perfect_segmentation(self):
        manual = {'width': 100, 'height': 100, 'sky_polygon': [[0,0], [1,0], [1,.5], [0,.5]]}
        self.assertEqual(mask_overlap(manual, manual)['iou_with_manual'], 1)
        fallback = mask_overlap(manual, None)
        self.assertAlmostEqual(fallback['iou_with_manual'], .5)
        self.assertAlmostEqual(fallback['selected_outside_manual_fraction'], .5)

    def test_per_id_losses_cannot_be_hidden_by_equal_total(self):
        self.assertEqual(transitions(['a', 'b'], ['a', 'c']), {'gained': ['c'], 'lost': ['b']})

    def test_raster_matches_pixel_center_boundary_convention(self):
        mask = rasterize([[.125,.125], [.875,.125], [.875,.875], [.125,.875]], 4, 4)
        self.assertEqual(int(mask.sum()), 4)

    def test_artifact_rejects_wrong_identity_mask_flags_nonfinite_and_false_success(self):
        entry, pinned = {'id': 'a', 'sha256': 'hash'}, {'dimensions': [100, 60]}
        record = {'id': 'a', 'status': 'failed'}
        original = {'id': 'a', 'source_sha256': 'hash', 'width': 100, 'height': 60,
                    'pixel_convention': 'top_left_zero_based', 'sky_mask': None,
                    'sky_statistics': False, 'sky_fill': False, 'report': {'status': 'failed'}}
        validate_artifact(original, record, entry, pinned, None)
        for key, value in [('source_sha256', 'bad'), ('width', 60), ('id', 'b'),
                           ('sky_fill', True), ('sky_mask', {}), ('bad', float('nan')),
                           ('report', {'status': 'solved'})]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_artifact({**copy.deepcopy(original), key: value}, record, entry, pinned, None)
        with self.assertRaisesRegex(ValueError, 'missing pose'):
            validate_artifact({**original, 'report': {'status': 'solved'}},
                              {**record, 'status': 'solved'}, entry, pinned, None)


class ComparisonIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_smartphone_gate.SmartphoneGateTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.output = f.root/'experiment'
        self.output.mkdir()
        baseline_path, manual_path, manual_baseline_path, probes_path = [f.root/name for name in
            ('baseline.json', 'masks.json', 'manual-baseline.json', 'probes.json')]
        masks = [{'id': frame_id, 'width': 2252, 'height': 4000,
                  'source_sha256': f.baseline['frames'][frame_id]['sha256'],
                  'sky_polygon': [[0,0], [1,0], [1,1], [0,1]]} for frame_id in ('a', 'b')]
        document = {'schema_version': 1, 'coordinates': 'exif_oriented_normalized_image_edges', 'masks': masks[:1]}
        write_json(manual_path, document)
        write_json(baseline_path, f.baseline)
        manual_baseline = copy.deepcopy(f.baseline)
        manual_baseline.update(mode='sky_fill', masks_sha256=sha256(manual_path), mask_ids=['a'])
        manual_baseline['config'].update(sky_fill=True, sky_statistics=True)
        write_json(manual_baseline_path, manual_baseline)
        write_json(probes_path, {'frames': [], 'match_radius_original_px': 12})
        write_json(self.output/'automatic-masks.json', {**document, 'masks': masks})
        write_json(self.output/'generation.json', {'frames': {'a': {}, 'b': {}}})
        paths = {'manifest': f.manifest, 'baseline': baseline_path, 'manual_masks': manual_path,
                 'manual_baseline': manual_baseline_path, 'probes': probes_path,
                 'binary': f.catalog, 'catalog': f.catalog}
        plan = {key: str(path) for key, path in paths.items()}
        plan.update(inputs={str(path): sha256(path) for path in paths.values()},
                    split={'annotated': ['a'], 'validation': ['b']})
        write_json(self.output/'plan.json', plan)
        for mode in ('unmasked', 'manual', 'automatic'):
            shutil.copytree(f.run, self.output/mode)
            summary = copy.deepcopy(f.summary)
            if mode != 'unmasked':
                summary['config'].update(sky_fill=True, sky_statistics=True,
                    sky_masks=str(manual_path if mode == 'manual' else self.output/'automatic-masks.json'))
            write_json(self.output/mode/'summary.json', summary)
            for index, frame_id in enumerate(('a', 'b')):
                path = self.output/mode/'solve-reports'/f'{frame_id}.json'
                artifact = json.loads(path.read_text())
                applied = mode == 'automatic' or (mode == 'manual' and frame_id == 'a')
                artifact.update(sky_mask=masks[index] if applied else None, sky_fill=applied, sky_statistics=applied)
                artifact['report']['timing_ms'] = {'total': 1}
                write_json(path, artifact)
        self.provenance = {'plan_sha256': sha256(self.output/'plan.json'),
                           'runs': {mode: {'exit_code': 0} for mode in ('unmasked', 'manual', 'automatic')}}
        self.rehash()

    def rehash(self):
        self.provenance['artifacts'] = {str(p.relative_to(self.output)): sha256(p)
            for p in self.output.rglob('*.json') if p.name not in ('plan.json', 'provenance.json', 'comparison.json')}
        write_json(self.output/'provenance.json', self.provenance)

    def test_valid_comparison_and_negative_candidate_are_both_reported(self):
        self.assertEqual(compare(self.output)['solved'], {'unmasked': 1, 'manual': 1, 'automatic': 1})
        path = self.output/'automatic'/'solve-reports'/'a.json'
        artifact = json.loads(path.read_text())
        artifact['report']['status'] = 'failed'
        write_json(path, artifact)
        write_json(self.output/'automatic'/'per-frame'/'a.json', {'id': 'a', 'status': 'failed'})
        summary_path = self.output/'automatic'/'summary.json'
        summary = json.loads(summary_path.read_text())
        summary['solver_track']['solved'] = 0
        write_json(summary_path, summary)
        self.rehash()
        self.assertEqual(compare(self.output)['automatic_vs_unmasked'], {'gained': [], 'lost': ['a']})

    def test_changed_masks_and_reports_are_rejected(self):
        for path in (self.output/'automatic-masks.json', self.output/'automatic'/'solve-reports'/'a.json'):
            original = path.read_bytes()
            path.write_bytes(original+b' ')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                compare(self.output)
            path.write_bytes(original)

    def test_missing_inventory_and_config_drift_are_rejected(self):
        self.provenance['artifacts'].pop('automatic/solve-reports/a.json')
        write_json(self.output/'provenance.json', self.provenance)
        with self.assertRaisesRegex(ValueError, 'incomplete artifact inventory'):
            compare(self.output)
        path = self.output/'automatic'/'summary.json'
        summary = json.loads(path.read_text())
        summary['config']['fov_hint_deg'] = 45
        write_json(path, summary)
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'solve configurations differ'):
            compare(self.output)

    def test_generation_algorithm_and_config_must_match_frozen_plan(self):
        path = self.output/'plan.json'
        plan = json.loads(path.read_text())
        plan.update(algorithm='rgb_gradient_v2', config={'history_rows': 12})
        write_json(path, plan)
        self.provenance['plan_sha256'] = sha256(path)
        write_json(self.output/'provenance.json', self.provenance)
        with self.assertRaisesRegex(ValueError, 'generation algorithm/config mismatch'):
            compare(self.output)

    def test_reference_comparison_requires_same_inputs_and_control_reports(self):
        reference = ComparisonIntegrityTests()
        reference.setUp()
        self.addCleanup(reference.doCleanups)
        result = compare(self.output)
        self.assertEqual(compare_reference(self.output, reference.output, result)['lost'], [])
        path = reference.output/'unmasked'/'solve-reports'/'a.json'
        artifact = json.loads(path.read_text())
        artifact['report']['quality']['rms_px'] = 2
        write_json(path, artifact)
        reference.rehash()
        with self.assertRaisesRegex(ValueError, 'reference unmasked control changed'):
            compare_reference(self.output, reference.output, result)
        plan_path = reference.output/'plan.json'
        plan = json.loads(plan_path.read_text())
        plan['inputs'][plan['binary']] = 'different'
        write_json(plan_path, plan)
        with self.assertRaisesRegex(ValueError, 'reference binary differs'):
            compare_reference(self.output, reference.output, result)


if __name__ == '__main__':
    unittest.main()
