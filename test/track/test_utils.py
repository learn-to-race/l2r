import unittest

from l2r.track import get_supported_levels
from l2r.track import level_2_simlevel
from l2r.track import level_2_trackmap
from l2r.track import LevelNotFoundError
from l2r.track import SimVersionNotSupported
from l2r.track.data import SUPPORTED_RACETRACKS


class TrackUtilsTest(unittest.TestCase):
    """
    Tests associated l2r track utilities
    """

    valid_sim_version = "ArrivalSim-linux-0.7.1.188691"
    invalid_sim_version = "sim12345"
    valid_level = "Thruxton"
    invalid_level = "this-track-doesnt-exist"

    def test_get_supported_levels_valid_version(self):
        levels = get_supported_levels(sim_version=self.valid_sim_version)
        expected_levels = [
            "VegasNorthRoad",
            "AngleseyNational",
            "Thruxton",
            "Banbury",
            "UPP_UDRV",
            "RaceTrack",
            "Ternois",
            "WinterTrack",
        ]
        self.assertTrue(levels, expected_levels)

    def test_get_supported_levels_invalid_version(self):
        with self.assertRaises(SimVersionNotSupported):
            get_supported_levels(self.invalid_sim_version)

    def test_level_2_track_map_all_levels(self):
        for level in SUPPORTED_RACETRACKS:
            trackmap, rand_pos, segments = level_2_trackmap(level=self.valid_level)
            self.assertTrue(isinstance(trackmap, str))
            self.assertTrue(isinstance(rand_pos, list))
            self.assertTrue(isinstance(segments, list))

    def test_level_2_track_map_invalid_level(self):
        with self.assertRaises(LevelNotFoundError):
            level_2_trackmap(self.invalid_level)

    def test_level_2_simlevel_valid_level(self):
        filepath = level_2_simlevel(
            level=self.valid_level, sim_version=self.valid_sim_version
        )
        expected_fp = "../../../ArrivalSim/Content/Ribbon/thruxton/thruxton.umap"
        self.assertEqual(filepath, expected_fp)

    def test_level_2_simlevel_invalid_level(self):
        with self.assertRaises(LevelNotFoundError):
            level_2_simlevel(
                level=self.invalid_level, sim_version=self.valid_sim_version
            )
