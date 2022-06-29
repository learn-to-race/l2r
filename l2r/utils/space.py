import math
from typing import List
from typing import Tuple

import numpy as np

from l2r.constants import EARTH_ECCEN2, DEG_2_RAD, EARTH_SEMI_MAJOR_RADIUS_M


def get_vehicle_corner_coordinates(
    vehicle_center: Tuple[float, float],
    vehicle_length: float,
    vehicle_width: float,
    heading: float,
) -> np.array:
    """Get the corner's of the vehicle. Assumes the vehicle is perfectly
    rectangular.

    :param tuple vehicle_center: the (x,y) coordinates of the center of the vehicle
    :param float vehicle_length: the length of the vehicle in meters
    :param float vehicle_width: the width of the vehicle in meters
    :param float heading: heading of the vehicle in radians

    :return: array of (x,y) coordinates in the following order:
      Top_right, Top_left, Bottom_right, Bottom_left
    :rtype: numpy.array
    """

    cos_val = math.cos(heading)
    sin_val = math.sin(heading)
    w_cos = (vehicle_width / 2) * cos_val
    w_sin = (vehicle_width / 2) * sin_val

    h_cos = (vehicle_length / 2) * cos_val
    h_sin = (vehicle_length / 2) * sin_val

    top_Right_x = vehicle_center[0] + w_cos - h_sin
    top_Right_y = vehicle_center[1] + w_sin + h_cos

    top_Left_x = vehicle_center[0] - w_cos - h_sin
    top_Left_y = vehicle_center[1] - w_sin + h_cos

    bot_Left_x = vehicle_center[0] - w_cos + h_sin
    bot_Left_y = vehicle_center[1] - w_sin - h_cos

    bot_Right_x = vehicle_center[0] + w_cos + h_sin
    bot_Right_y = vehicle_center[1] + w_sin - h_cos

    return np.array(
        [
            (top_Right_x, top_Right_y),
            (top_Left_x, top_Left_y),
            (bot_Right_x, bot_Right_y),
            (bot_Left_x, bot_Left_y),
        ]
    )


def convert_ll_to_enu(
    center: List[float], ref_point: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    """Convert latitude/longitude coordinates to ENU coordinates.

    :param list center: latitude/longitude coordinates of vehicle
    :param tuple ref_point: local reference point which serves as the local
      origin in the form (east, north, up)

    :return: ENU coordinates of the center of the vehicle in the form:
      [East, North, Up]
    :rtype: Tuple
    """

    ref_x, ref_y, ref_z = ref_point
    x, y, z = center

    # Initialize sines and cosines
    clatRef = math.cos(ref_x * DEG_2_RAD)
    clonRef = math.cos(ref_y * DEG_2_RAD)
    slatRef = math.sin(ref_x * DEG_2_RAD)
    slonRef = math.sin(ref_y * DEG_2_RAD)
    clat = math.cos(x * DEG_2_RAD)
    clon = math.cos(y * DEG_2_RAD)
    slat = math.sin(x * DEG_2_RAD)
    slon = math.sin(y * DEG_2_RAD)

    # Compute reference position vector in ECEF coordinates
    r0Ref = EARTH_SEMI_MAJOR_RADIUS_M / (
        math.sqrt(1.0 - EARTH_ECCEN2 * slatRef * slatRef)
    )
    ecefRef = [0.0] * 3
    ecefRef[0] = (ref_z + r0Ref) * clatRef * clonRef
    ecefRef[1] = (ref_z + r0Ref) * clatRef * slonRef
    ecefRef[2] = (ref_z + r0Ref * (1.0 - EARTH_ECCEN2)) * slatRef

    # Compute data position vectors relative to reference point in ECEF co-ordinates
    r0 = EARTH_SEMI_MAJOR_RADIUS_M / (math.sqrt(1.0 - EARTH_ECCEN2 * slat * slat))
    dECEF = [0.0] * 3
    dECEF[0] = (z + r0) * clat * clon - ecefRef[0]
    dECEF[1] = (z + r0) * clat * slon - ecefRef[1]
    dECEF[2] = (z + r0 * (1.0 - EARTH_ECCEN2)) * slat - ecefRef[2]

    # Define rotation from ECEF to ENU
    R = [
        [-slonRef, clonRef, 0],
        [-slatRef * clonRef, -slatRef * slonRef, clatRef],
        [clatRef * clonRef, clatRef * slonRef, slatRef],
    ]

    enu = [0.0] * 3

    for i in range(3):
        enu[i] = 0
        for j in range(3):
            enu[i] += R[i][j] * dECEF[j]

    # Assign matrix multiplication outputs to custom message for publishing
    enu_east = enu[0]
    enu_north = enu[1]
    enu_up = enu[2]

    return enu_east, enu_north, enu_up


def smooth_yaw(yaw: List[float]) -> List[float]:
    """Smoothing function"""
    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]
    return yaw
