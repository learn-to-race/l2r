import unittest

import numpy as np

from l2r.utils.space import convert_ll_to_enu
from l2r.utils.space import get_vehicle_corner_coordinates


class UtilsTest(unittest.TestCase):
    """
    Tests associated l2r env utilities
    """

    def test_get_vehicle_corner_coordinates(self):
        actual = get_vehicle_corner_coordinates(
            vehicle_center=(55.0, 40.0),
            vehicle_length=2.0,
            vehicle_width=1.5,
            heading=0.2,
        )

        expected = np.array(
            [
                (55.5363806, 41.12906858),
                (54.06628074, 40.83106458),
                (55.93371926, 39.16893542),
                (54.4636194, 38.8709314),
            ]
        )
        self.assertTrue(np.allclose(actual, expected))

    def test_convert_ll_to_enu(self):
        enu_x, enu_y, enu_z = convert_ll_to_enu(
            center=[53.189876, -4.501157, 0.0], ref_point=(53.1906219, -4.4966478, 0.0)
        )
        expected = np.array([-301.4054, -83.0015848, -0.00764652789])
        self.assertTrue(np.allclose(np.array([enu_x, enu_y, enu_z]), expected))
