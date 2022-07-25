import struct

##############################################################################
# Simulator Lag Delay
##############################################################################
OBS_DELAY = 0.1
MEDIUM_DELAY = 3
TIMEOUT_DELAY = 30
LAUNCHING_DELAY = 15

##############################################################################
# Vehicle dimensions in meters
##############################################################################
CAR_DIMS = [3.0, 1.68]

##############################################################################
# Raw action space boundaries
##############################################################################
MIN_STEER_REQ = -1.0
MAX_STEER_REQ = 1.0
STEER_REQ_RANGE = MAX_STEER_REQ - MIN_STEER_REQ

MIN_ACC_REQ = -16.0
MAX_ACC_REQ = 6.0
ACC_REQ_RANGE = MAX_ACC_REQ - MIN_ACC_REQ

##############################################################################
# Pose observation space boundaries
##############################################################################
MIN_OBS_ARR = [
    -1.0, -1.0, -1.0,                    # steering, gear, mode
    -200.0, -200.0, -10.0,               # velocity
    -100.0, -100.0, -100.0,              # acceleration
    -1.0, -1.0, -5.0,                    # angular velocity
    -6.2832, -6.2832, -6.2832,           # yaw, pitch, roll
    -2000.0, 2000.0, 2000.0,             # location coordinates in the format (y, x, z)
    -2000.0, -2000.0, -2000.0, -2000.0,  # rpm (per wheel)
    -1.0, -1.0, -1.0, -1.0,              # brake (per wheel)
    -1.0, -1.0, -1300.0, -1300.0,        # torq (per wheel)
]

MAX_OBS_ARR = [
    1.0, 4.0, 1.0,                   # steering, gear, mode
    200.0, 200.0, 10.0,              # velocity
    100.0, 100.0, 100.0,             # acceleration
    1.0, 1.0, 5.0,                   # angular velocity
    6.2832, 6.2832, 6.2832,          # yaw, pitch, roll
    2000.0, 2000.0, 2000.0,          # location coordinates in the format (y, x, z)
    2500.0, 2500.0, 2500.0, 2500.0,  # rpm (per wheel)
    1.0, 1.0, 2.0, 2.0,              # brake (per wheel)
    1.0, 1.0, 1300.0, 1300.0,        # torq (per wheel)
]

VELOCITY_IDX_LOW = 3
VELOCITY_IDX_HIGH = 6

##############################################################################
# Environment specific
##############################################################################
N_EPISODE_LAPS = 1
N_SEGMENTS = 10

##############################################################################
# Socket receive size
##############################################################################
BUFFER_SIZE = 1024

##############################################################################
# Valid gear actions
##############################################################################
NEUTRAL_GEAR = 0
DRIVE_GEAR = 1
REVERSE_GEAR = 2
PARK_GEAR = 3
GEAR_REQ_RANGE = 4

##############################################################################
# Acceleration request boundaries
##############################################################################
MIN_ACC_REQ = -16.
MAX_ACC_REQ = 6.

##############################################################################
# Acceleration request boundaries
##############################################################################
MIN_STEER_REQ = -1.
MAX_STEER_REQ = 1.

##############################################################################
# Image Type Declarations
##############################################################################
CV_8U = 0
CV_8S = 1
CV_16U = 2
CV_16S = 3
CV_32S = 4
CV_32F = 5
CV_64F = 6

##############################################################################
# Message byte formats
##############################################################################
OUT_MSG_HEADER_FMT = '=ffb'
OUT_MSG_HEADER_LENGTH = struct.calcsize(OUT_MSG_HEADER_FMT)
IN_MSG_HEADER_FMT = '=fbbffffffffffffdddffffffffffff'
IN_MSG_HEADER_LENGTH = struct.calcsize(IN_MSG_HEADER_FMT)
IMG_MSG_HEADER_FMT = 'iiiiiqq'
HEADER_LENGTH = struct.calcsize(IMG_MSG_HEADER_FMT)

##############################################################################
# Geolocation constants
##############################################################################
EARTH_SEMI_MAJOR_RADIUS_M = 6_378_137
EARTH_ECCEN2 = 0.00669438
DEG_2_RAD = 3.141592 / 180
