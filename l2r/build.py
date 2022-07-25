import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import yaml

from l2r.core import ActionInterface
from l2r.core import CameraConfig
from l2r.core import CameraInterface
from l2r.core import PoseInterface
from l2r.env import SimulatorController
from l2r.env import RacingEnv


def build_env(
    levels: List[str] = ["Thruxton"],
    controller_kwargs: Optional[Dict[str, Union[str, bool]]] = dict(),
    camera_cfg: Optional[Union[str, List[Dict[str, Any]]]] = dict(),
    action_cfg: Optional[Dict[str, Any]] = dict(),
    pose_cfg: Optional[Dict[str, Any]] = dict(),
    env_ip: Optional[str] = "0.0.0.0",
) -> RacingEnv:
    """Build a l2r environment

    :param list levels: a list of racetracks to use

    :return: a gym environment for autonomous racing
    :rtype: l2r.env.RacingEnv
    """

    # attempt to create an interface with the simulator
    controller = SimulatorController(**controller_kwargs)

    # create camera interfaces
    camera_interfaces = build_camera_interfaces(camera_cfg=camera_cfg)

    # create pose interface
    pose_interface = PoseInterface(**pose_cfg)

    # create action interface
    action_interface = ActionInterface(**action_cfg)

    # create environment
    env = RacingEnv(
        controller=controller,
        action_interface=action_interface,
        camera_interfaces=camera_interfaces,
        pose_interface=pose_interface,
        env_ip=env_ip,
    )

    return env.make(levels=levels)


def build_camera_interfaces(
    camera_cfg: Union[str, List[Dict[str, Any]]]
) -> List[CameraInterface]:
    """Build camera interfaces from a configuration file or list of camera
    config dictionaries"""
    if not camera_cfg:
        return [CameraInterface(cfg=CameraConfig())]

    cfgs = (
        load_camera_config_file(filepath=camera_cfg)
        if isinstance(camera_cfg, str)
        else camera_cfg
    )

    # Load and validate camera configurations
    camera_configs = [CameraConfig(**cfg) for cfg in cfgs]

    # Build and return camera interfaces
    return [CameraInterface(cfg=cfg) for cfg in camera_configs]


def load_camera_config_file(filepath: str) -> List[Dict[str, Any]]:
    """Load camera configuration file"""
    cfg = _load_config(filepath=filepath)

    if "cameras" not in cfg:
        logging.error("Camera configuration must have top-level key: `cameras`")
        raise KeyError

    if not isinstance(cfg["cameras"], list):
        logging.error("Expected camera configuration to be a list")
        raise TypeError

    return cfg["cameras"]


def _load_config(filepath: str) -> Dict[str, Any]:
    """Load a json or yaml file"""
    with open(filepath, "r") as fp:
        return yaml.safe_load(fp)
