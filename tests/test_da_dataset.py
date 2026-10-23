import unittest

import numpy as np
import pandas as pd

from data.da_dataset import observation_offsets, project_station_values


class ObservationSamplingTest(unittest.TestCase):
    def test_six_hours_every_twenty_minutes_is_inclusive(self):
        offsets = observation_offsets(6, 20)
        self.assertEqual(len(offsets), 19)
        self.assertEqual(offsets[0], pd.Timedelta(hours=-6))
        self.assertEqual(offsets[-1], pd.Timedelta(0))
        self.assertTrue(
            all(delta == pd.Timedelta(minutes=20) for delta in offsets[1:] - offsets[:-1])
        )

    def test_non_divisible_window_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            observation_offsets(1, 17)

    def test_station_projection_averages_collisions_and_preserves_missing(self):
        raw = np.full((2, 2, 2), np.nan, dtype=np.float32)
        raw[:, 0, 0] = [2.0, np.nan]
        raw[:, 0, 1] = [4.0, 6.0]
        data, mask = project_station_values(
            raw,
            np.array([0, 0]), np.array([0, 1]),
            np.array([2, 2]), (2, 2),
        )
        np.testing.assert_array_equal(data[:, 1, 0], [3.0, 6.0])
        np.testing.assert_array_equal(mask[:, 1, 0], [True, True])
        self.assertFalse(mask[:, 0, :].any())


if __name__ == "__main__":
    unittest.main()
