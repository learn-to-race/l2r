import unittest

from l2r.build import build_camera_interfaces
from l2r.build import load_camera_config_file


class TestBuildUtilites(unittest.TestCase):
    """
    Tests associated with l2r.build utilities
    """

    valid_camera_fp = "./test/resources/configs/valid-camera.yml"
    missing_key_fp = "./test/resources/configs/camera-missing-camera-key.yml"
    invalid_camera_name_fp = "./test/resources/configs/camera-invalid-camera-name.yml"

    def test_valid_camera_load(self):
        """Test valid configuration loads correctly"""
        cfg = load_camera_config_file(filepath=self.valid_camera_fp)
        expected = [
            {
                "Name": "CameraFrontRGB",
                "Addr": "tcp://0.0.0.0:8008",
                "Format": "ColorBGR8",
                "bAutoAdvertise": True,
                "FOVAngle": 90,
                "Width": 256,
                "Height": 192,
            }
        ]
        self.assertEqual(cfg, expected)

    def test_missing_camera_key(self):
        """Test invalid camera configuration raises error"""
        with self.assertRaises(KeyError):
            load_camera_config_file(filepath=self.missing_key_fp)

    def test_invalid_camera_name(self):
        """Test invalid camera configuration raises error"""
        with self.assertRaises(TypeError):
            build_camera_interfaces(filepath=self.invalid_camera_name_fp)
