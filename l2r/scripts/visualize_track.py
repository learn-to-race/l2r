# ========================================================================= #
# Filename:                                                                 #
#    visualize_track.py                                                     #
#                                                                           #
# Description:                                                              #
#    Simple racetrack plotter                                               #
# ========================================================================= #

import json
import os
import pathlib
import sys

import numpy
from matplotlib import pyplot as plt

from racetracks.mapping import level_2_trackmap

if __name__ == "__main__":

    # command line argument
    track_name = sys.argv[1]
    assert track_name in ['VegasNorthRoad', 'Thruxton']

    # load track
    track_fn = level_2_trackmap(track_name)

    with open(os.path.join(pathlib.Path().absolute(), track_fn), 'r') as f:
        racetrack = json.load(f)
        inside_coords = numpy.asarray(racetrack['Inside'])
        outside_coords = numpy.asarray(racetrack['Outside'])
        center_coords = numpy.asarray(racetrack['Centre'])

    if track_name == 'VegasNorthRoad':
        raceline = numpy.asarray(racetrack['Racing'])
        inside_coords = inside_coords[:, :-1]
        outside_coords = outside_coords[:, :-1]

    in_arr = numpy.transpose(inside_coords)
    out_arr = numpy.transpose(outside_coords)
    center_arr = numpy.transpose(center_coords)

    # plot track
    plt.plot(in_arr[0], in_arr[1], color='black', linewidth=0.5)
    plt.plot(out_arr[0], out_arr[1], color='black', linewidth=0.5)
    plt.plot(center_arr[0], center_arr[1], color='grey', linewidth=0.5)

    tn = 'Thruxton Circuit'
    if track_name == 'VegasNorthRoad':
        race_arr = numpy.transpose(raceline)
        plt.plot(race_arr[0], race_arr[1], linewidth=0.5, label='Expert Line')
        tn = 'Las Vegas North'

    plt.title(tn + ' Racetrack')
    # plt.legend()
    plt.show()
