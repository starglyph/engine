import unittest
from sky_statistics_experiment import probe_metrics, check_pair_configs


class ProbeMetricTests(unittest.TestCase):
    def test_sky_fill_pair_changes_only_occupancy(self):
        a = {'sky_masks': 'm.json', 'sky_statistics': True}
        b = {**a, 'sky_fill': True}
        check_pair_configs([a,b], 'm.json', True)
        with self.assertRaisesRegex(ValueError, 'statistics mode mismatch'):
            check_pair_configs([{**a,'sky_statistics':False},b], 'm.json', True)
        with self.assertRaisesRegex(ValueError, 'sky-fill mode mismatch'):
            check_pair_configs([a,a], 'm.json', True)
        with self.assertRaisesRegex(ValueError, 'configuration mismatch'):
            check_pair_configs([a,{**b,'fov_hint':40}], 'm.json', True)

    def test_scaling_one_to_one_top_k_and_uncertainty(self):
        annotation = {'source_sha256': 'hash', 'width': 400, 'height': 200,
                      'points': [{'id': 'a', 'x': 101.5, 'y': 81.5, 'label': 'visible_compact_source'},
                                 {'id': 'b', 'x': 102.5, 'y': 81.5, 'label': 'visible_compact_source'},
                                 {'id': 'c', 'x': 201.5, 'y': 81.5, 'label': 'uncertain_diffuse_source'}],
                      'cloud_regions': [{'id': 'cloud', 'polygon': [[0,0],[150,0],[150,150],[0,150]]}]}
        artifact = {'source_sha256': 'hash', 'width': 400, 'height': 200, 'detection_diagnostics': [
            {'width': 100, 'height': 50, 'tier': 'deep', 'max_detections': 1,
             'result': {'stats': {}, 'detections': [{'x': 25., 'y': 20., 'rank': 0},
                                                   {'x': 50., 'y': 20., 'rank': 1}]}}]}
        row = probe_metrics(artifact, annotation, 12)[0]
        self.assertEqual((row['primary_total'], row['primary_before_top_k'], row['primary_retained']), (2, 1, 1))
        self.assertEqual(row['probes'][0]['distance_original_px'], 0)
        self.assertFalse(row['probes'][1]['detected_before_top_k'])
        self.assertTrue(row['probes'][2]['detected_before_top_k'])
        self.assertFalse(row['probes'][2]['retained'])
        self.assertEqual(row['cloud_region_retained'], {'cloud': 1})
        artifact['source_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            probe_metrics(artifact, annotation, 12)


if __name__ == '__main__':
    unittest.main()
