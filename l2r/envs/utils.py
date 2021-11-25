# ========================================================================= #
# Filename:                                                                 #
#    utils.py                                                               #
#                                                                           #
# Description:                                                              #
#    environment utilities                                                  #
# ========================================================================= #

import copy
import socket
import struct
import threading
from math import sqrt, cos, sin

import numpy as np
import zmq

from core.templates import AbstractInterface

# Message byte formats
OUT_MSG_HEADER_FMT = '=ffb'
OUT_MSG_HEADER_LENGTH = struct.calcsize(OUT_MSG_HEADER_FMT)
IN_MSG_HEADER_FMT = '=fbbffffffffffffdddffffffffffff'
IN_MSG_HEADER_LENGTH = struct.calcsize(IN_MSG_HEADER_FMT)
IMG_MSG_HEADER_FMT = 'iiiiiqq'
HEADER_LENGTH = struct.calcsize(IMG_MSG_HEADER_FMT)

# Image Type Declarations
CV_8U = 0
CV_8S = 1
CV_16U = 2
CV_16S = 3
CV_32S = 4
CV_32F = 5
CV_64F = 6

# Socket receive size
BUFFER = 1024

# Valid gear actions
NEUTRAL_GEAR = 0
DRIVE_GEAR = 1
REVERSE_GEAR = 2
PARK_GEAR = 3
GEAR_REQ_RANGE = 4

# Acceleration request boundaries
MIN_ACC_REQ = -16.
MAX_ACC_REQ = 6.

# Steering request boundaries
MIN_STEER_REQ = -1.
MAX_STEER_REQ = 1.


class InvalidActionException(Exception):
    pass


class ActionInterface(object):
    """Action send interface. This class communicates with the simulator and
    sends action requests from the agent.

    :param str ip: ip address
    :param int port: port to bind to
    :param float max_steer: maximum steering request, bounded by 1.
    :param float min_steer: minimum steering request, bounded by -1.
    :param flaot max_accel: maximum acceleration request, bounded by 6.
    :param float min_accel: minimum acceleration request, bounded by -16.
    """

    def __init__(self, ip='', port=7077, max_steer=1., min_steer=-1.,
                 max_accel=MAX_ACC_REQ, min_accel=MIN_ACC_REQ):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.max_steer = max_steer
        self.min_steer = min_steer
        self.max_accel = max_accel
        self.min_accel = min_accel
        if max_steer > MAX_STEER_REQ or min_steer < MIN_ACC_REQ:
            raise InvalidActionException('Invalid steering boundaries')
        if max_accel > MAX_ACC_REQ or min_accel < MIN_ACC_REQ:
            raise InvalidActionException('Invalid acceleration boundaries')

    def act(self, action):
        """Send action request to the simulator.

        :param array-like action: action to send to the simulator in the form:
          [steering, acceleration], expected to be in the range (-1., 1.)
        """
        self._check_action(action)
        steer, acc = self._scale_action(action)
        bytes = struct.pack(OUT_MSG_HEADER_FMT, steer, acc, DRIVE_GEAR)
        self.sock.sendto(bytes, self.addr)

    def _scale_action(self, action):
        """Scale the action
        """
        steer, acc = action[0], action[1]
        steer *= self.max_steer if steer > 0 else (-1. * self.min_steer)
        acc *= self.max_accel if acc > 0 else (-1. * self.min_accel)
        return steer, acc

    def _check_action(self, action):
        """Check that the action is valid with reference to the action space.

        :param array-like action: action to send to the simulator in the form:
          [steering, acceleration], expected to be in the range (-1., 1.)
        """
        if action[0] < -1.0 or action[0] > 1.0:
            raise InvalidActionException('Invalid steering request')

        if action[1] < -1.0 or action[1] > 1.0:
            raise InvalidActionException('Invalid acceleration request')


class PoseInterface(AbstractInterface):
    """Receives sensor data from the simulator. The data received is in the
    following format:

    [0,1,2] steering, gear, mode \n
    [3,4,5] \n
    [6,7,8] acceleration \n
    [9,10,11] angular velocity \n
    [12,13,14] yaw, pitch, roll \n
    [15,16,17] location coordinates in the format (y, x, z) \n
    [18,19,20 21] rpm (by wheel) \n
    [22,23,24,25] brake (by wheel) \n
    [26,27,28,29] torq (by wheel)

    :param str ip: ip address
    :param int port: port to bind to
    :param int data_elems: number of elements to listen for, elements are
      assumed to be of type float
    """

    def __init__(self, ip='', port=7078, data_elems=30):
        addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(addr)
        self.data_elems = data_elems

    def start(self):
        """Starts a thread to listen for data from the simulator.
        """
        self.reset()
        self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()

    def get_data(self):
        """Return the most recent data received from the simulator.

        :return: data from the simulator
        :rtype: array of length self.data_elems
        """
        return copy.deepcopy(self.data)

    def reset(self):
        """Allocates memory for data receive.
        """
        self.data = np.zeros(shape=(self.data_elems,), dtype=float)

    def _receive(self):
        """Indefinitely wait for data from the simulator and unpack into an
        array.
        """
        while True:
            bytes, addr = self.sock.recvfrom(BUFFER)
            assert len(bytes) == IN_MSG_HEADER_LENGTH
            self.data = np.asarray(struct.unpack(IN_MSG_HEADER_FMT, bytes))


class CameraInterface(AbstractInterface):
    """Receives images from the simulator.

    :param str ip: ip address to listen on
    :param int port: system port
    """

    def __init__(self, ip='tcp://127.0.0.1', port=8008):
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, b'')
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.connect(f'{ip}:{port}')
        self.addr = f'{ip}:{port}'

    def start(self, img_dims):
        """Starts a thread to listen for images on.

        :param tuple img_dims: dimensions of the image to listen for in the
          form: (width, height, depth)
        """
        self.img_dims = (img_dims[1], img_dims[0], img_dims[2])
        self.reset()
        self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()

    def get_data(self):
        """Return the most recent image(s) received from the simulator.

        :return: RGB image of shape (height, width, 3)
        :rtype: numpy.array
        """
        return copy.deepcopy(self.img)

    def reset(self):
        """Allocates memory for data receive.
        """
        self.img = np.zeros(shape=self.img_dims, dtype=float)

    def reconnect(self):
        """Reconnect to the socket
        """
        self.sock.connect(self.addr)

    def _receive(self):
        """Indefinitely wait for data from the simulator and unpack into an
        array. Supports RGB images only.
        """
        while True:
            try:
                rawbuf = self.sock.recv(0)
            except zmq.ZMQError as e:
                if e.errno != zmq.EAGAIN:
                    raise

            head = struct.unpack(IMG_MSG_HEADER_FMT, rawbuf[:HEADER_LENGTH])
            data = rawbuf[HEADER_LENGTH:]
            im = np.frombuffer(data, self._ocv2np_type(head[4]))
            self.img = im.reshape(head[1:4])[:, :, ::-1]  # BGR to RGB

    def _ocv2np_type(self, ocv_type):
        """Utility to determine dtype of the image.
        """
        if ocv_type == CV_8U:
            return np.uint8
        if ocv_type == CV_8S:
            return np.int8
        if ocv_type == CV_16U:
            return np.uint16
        if ocv_type == CV_16S:
            return np.int16
        if ocv_type == CV_32S:
            return np.int32
        if ocv_type == CV_32F:
            return np.float32
        if ocv_type == CV_64F:
            return np.float64


class GeoLocation(object):
    """Global to local coordinate conversion class.

    :param tuple ref_point: local reference point which serves as the local
      origin in the form (east, north, up)
    """

    def __init__(self, ref_point):
        self.ref_point = ref_point
        self.D2R = 3.141592 / 180
        self.EARTHSEMIMAJOR = 6378137
        self.EARTHECCEN2 = 0.00669438

    @staticmethod
    def get_corners(center, angle, dimensions):
        """Get the corner's of the vehicle. Assumes the vehicle is perfectly
        rectangular.

        :param tuple center: the (x,y) coordinates of the center of the vehicle
        :param float angle: heading of the vehicle in radians
        :param list dimensions: [height, width] of the vehicle in meters

        :return: array of (x,y) coordinates in the following order:
          Top_right, Top_left, Bottom_right, Bottom_left
        :rtype: numpy.array
        """
        length, width = dimensions

        cos_val = cos(angle)
        sin_val = sin(angle)
        w_cos = (width / 2) * cos_val
        w_sin = (width / 2) * sin_val

        h_cos = (length / 2) * cos_val
        h_sin = (length / 2) * sin_val

        Top_Right_x = center[0] + w_cos - h_sin
        Top_Right_y = center[1] + w_sin + h_cos

        Top_Left_x = center[0] - w_cos - h_sin
        Top_Left_y = center[1] - w_sin + h_cos

        Bot_Left_x = center[0] - w_cos + h_sin
        Bot_Left_y = center[1] - w_sin - h_cos

        Bot_Right_x = center[0] + w_cos + h_sin
        Bot_Right_y = center[1] + w_sin - h_cos

        return np.array([(Top_Right_x, Top_Right_y), (Top_Left_x, Top_Left_y),
                         (Bot_Right_x, Bot_Right_y), (Bot_Left_x, Bot_Left_y)])

    def convert_to_ENU(self, center):
        """Convert latitude/longitude coordinates to ENU coordinates.

        :param list center: latitude/longitude coordinates of vehicle
        :return: ENU coordinates of the center of the vehicle in the form:
          [East, North, Up]
        :rtype: numpy.array
        """
        ref_x, ref_y, ref_z = self.ref_point
        x, y, z = center

        # Initialize sines and cosines
        clatRef = cos(ref_x * self.D2R)
        clonRef = cos(ref_y * self.D2R)
        slatRef = sin(ref_x * self.D2R)
        slonRef = sin(ref_y * self.D2R)
        clat = cos(x * self.D2R)
        clon = cos(y * self.D2R)
        slat = sin(x * self.D2R)
        slon = sin(y * self.D2R)

        # Compute reference position vector in ECEF coordinates
        r0Ref = self.EARTHSEMIMAJOR / (sqrt(1.0 - self.EARTHECCEN2 * slatRef * slatRef))
        ecefRef = [0.0] * 3
        ecefRef[0] = (ref_z + r0Ref) * clatRef * clonRef
        ecefRef[1] = (ref_z + r0Ref) * clatRef * slonRef
        ecefRef[2] = (ref_z + r0Ref * (1.0 - self.EARTHECCEN2)) * slatRef

        # Compute data position vectors relative to reference point in ECEF co-ordinates
        r0 = self.EARTHSEMIMAJOR / (sqrt(1.0 - self.EARTHECCEN2 * slat * slat))
        dECEF = [0.0] * 3
        dECEF[0] = (z + r0) * clat * clon - ecefRef[0]
        dECEF[1] = (z + r0) * clat * slon - ecefRef[1]
        dECEF[2] = (z + r0 * (1.0 - self.EARTHECCEN2)) * slat - ecefRef[2]

        # Define rotation from ECEF to ENU
        R = [[-slonRef, clonRef, 0], [-slatRef * clonRef, -slatRef * slonRef, clatRef],
             [clatRef * clonRef, clatRef * slonRef, slatRef]]

        enu = [0.0] * 3

        for i in range(3):
            enu[i] = 0
            for j in range(3):
                enu[i] += R[i][j] * dECEF[j]

        # Assign matrix multiplication outputs to custom message for publishing
        enu_east = enu[0]
        enu_north = enu[1]
        enu_up = enu[2]

        return np.array([enu_east, enu_north, enu_up])
