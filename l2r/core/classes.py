import dataclasses
import logging
from typing import Dict
from typing import Set


class InvalidConfigError(Exception):
    pass


@dataclasses.dataclass
class CameraConfig:
    """Defines a simulator camera configuration"""

    name: str = "CameraFrontRGB"
    Addr: str = "0.0.0.0"
    Format: str = "ColorBGR8"
    bAutoAdvertise: bool = True
    FOVAngle: int = 90
    Width: int = 512
    Height: int = 384
    sim_addr: str = "tcp://0.0.0.0:8008"

    def __post_init__(self) -> None:
        """Configuration checks"""
        if self.name not in self.valid_cameras:
            logging.error(f"Found camera {self.name}. Must be in {self.valid_cameras}")
            raise InvalidConfigError

    @property
    def valid_cameras(self) -> Set[str]:
        return {
            "CameraFrontRGB",
            "CameraLeftRGB",
            "CameraRightRGB",
            "CameraFrontSegm",
            "CameraRightSegm",
            "CameraBirdsEye",
            "CameraBirdsEyeSegm",
        }

    def get_sim_param_dict(self) -> Dict[str, str]:
        """Return configuration as a dictionary,"""
        return {
            "ColorPublisher : Addr": self.Addr,
            "Format": self.Format,
            "bAutoAdvertise": self.bAutoAdvertise,
            "FOVAngle": self.FOVAngle,
            "Width": self.Width,
            "Height": self.Height,
        }
