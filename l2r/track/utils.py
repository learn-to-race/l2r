import os
from typing import List
from typing import Tuple

from .data import SUPPORTED_RACETRACKS
from .data import SUPPORTED_SIMULATORS


class LevelNotFoundError(Exception):
    pass


class SimVersionNotSupported(Exception):
    pass


def get_supported_levels(sim_version: str) -> List[str]:
    """Get a list of supported levels for a specific simulator version

    :param str sim_version: the simulator version
    :returns: a list levels
    :rtype: a list of strings
    """
    for sim in SUPPORTED_SIMULATORS:
        if sim["version"] == sim_version:
            return list(sim["levels"].keys())

    raise SimVersionNotSupported


def level_2_trackmap(level: str) -> Tuple[str, List[List[float]], List[List[float]]]:
    """Utility to convert a human readable track name to the filepath of
    the racetrack's map.

    :param str level: name of the racetrack
    :returns: the filepath of the racetrack's map, random start positions in
      the form [x,y,z,yaw], segment boundary points (enu)
    :rtype: string, list of lists of floats, list of list of floats
    """
    for rt in SUPPORTED_RACETRACKS:
        if level == rt["level"]:
            _path, _ = os.path.split(__file__)
            return os.path.join(_path, rt["trackmap"]), rt["random_pos"], rt["segments"]

    raise LevelNotFoundError(f"Map of track not found for level: {level}")


def level_2_simlevel(level: str, sim_version: str) -> str:
    """Utility to convert a human readable track name to the name of the track
    used in the simulator (typically a filepath)

    :param str level: name of the racetrack
    :param str sim_version: version of the simulator being used
    :returns: the filepath of the racetrack used by the simulator
    :rtype: string
    """
    for simulator in SUPPORTED_SIMULATORS:
        if simulator["version"] == sim_version:
            try:
                return simulator["levels"][level]
            except KeyError:
                raise LevelNotFoundError

    raise SimVersionNotSupported(
        f"Could not find level: {level} \
                               for simulator verision: {sim_version}"
    )
