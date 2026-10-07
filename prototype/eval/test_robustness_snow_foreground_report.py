"""A timeout cannot silently count as reproduced NoMatch."""
import unittest

from robustness_snow_foreground_report import control_changes


class SnowReportTests(unittest.TestCase):
    def case(self, status):
        return [dict(name='control', attempts=[dict(db=1, k=20, fov=23., result=dict(error=status))])]

    def test_timeout_is_a_reproduction_failure(self):
        changes = control_changes(self.case('Timeout'), ['NoMatch'])
        self.assertEqual(changes, [dict(case='control', db=1, k=20, fov=23., before='NoMatch', after='Timeout')])

    def test_equal_status_has_no_change(self):
        self.assertEqual(control_changes(self.case('NoMatch'), ['NoMatch']), [])

    def test_short_output_does_not_silently_zip(self):
        with self.assertRaisesRegex(ValueError, 'count'):
            control_changes(self.case('NoMatch'), ['NoMatch', 'NoMatch'])


if __name__ == '__main__':
    unittest.main()
