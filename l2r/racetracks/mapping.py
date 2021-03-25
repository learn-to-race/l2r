# =========================================================================== #
# Filename:                                                                   #
#    mapping.py                                                               #
#                                                                             #
# Description:                                                                #
#    This file contains utility functions to map to appropriate directories   #
# =========================================================================== #

import json


class LevelNotFoundError(Exception):
    pass


def level_2_trackmap(level):
    """Utility to convert a human readable track name to the filepath of
    the racetrack's map.

    :param str level: name of the racetrack
    :returns: the filepath of the racetrack's map, random start positions in
      the form [x,y,z,yaw]
    :rtype: string, list of lists
    """
    with open('racetracks/racetracks.json', 'r') as f:
        data = json.load(f)
        racetracks = data['racetracks']
        for track in racetracks:
            if level == track['level']:
                return track['trackmap'], track['random_pos']

    raise LevelNotFoundError(f'Map of track not found for level: {level}')


def level_2_simlevel(level, sim_version):
    """Utility to convert a human readable track name to the name of the track
    used in the simulator (typically a filepath)

    :param str level: name of the racetrack
    :param str sim_version: version of the simulator being used
    :returns: the filepath of the racetrack used by the simulator
    :rtype: string
    """
    with open('racetracks/racetracks.json', 'r') as f:
        data = json.load(f)
        for simulator in data['simulators']:
            if simulator['version'] == sim_version:
                return simulator['levels'][level]

    raise LevelNotFoundError(f'Could not find level: {level} \
                               for simulator verision: {sim_version}')
