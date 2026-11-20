import unittest

from imagent.pipeline.progress import format_duration


class FormatDurationTests(unittest.TestCase):
    def test_minutes_hours_and_days_join_once_a_minute_an_hour_or_a_day_is_crossed(self):
        for seconds, shown in ((0, "0s"), (12, "12s"), (59.9, "59s"), (60, "1m 00s"), (150, "2m 30s"), (3599, "59m 59s"),
                               (3600, "1h 00m 00s"), (3930, "1h 05m 30s"), (86399, "23h 59m 59s"),
                               (86400, "1d 00h 00m 00s"), (183930, "2d 03h 05m 30s")):
            self.assertEqual(format_duration(seconds), shown)

    def test_decimals_show_only_below_a_minute(self):
        self.assertEqual(format_duration(12.345, decimals=1), "12.3s")
        self.assertEqual(format_duration(12.345, decimals=2), "12.35s")
        self.assertEqual(format_duration(150.7, decimals=2), "2m 30s")
        self.assertEqual(format_duration(90000, decimals=1), "1d 01h 00m 00s")


if __name__ == "__main__":
    unittest.main()
