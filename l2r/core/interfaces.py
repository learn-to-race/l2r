import copy
import logging
import socket
import struct
import threading
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import zmq

from .classes import CameraConfig
from .templates import AbstractInterface
from l2r.constants import (
    BUFFER_SIZE,
    CV_8U,
    CV_8S,
    CV_16U,
    CV_16S,
    CV_32S,
    CV_32F,
    CV_64F,
    DRIVE_GEAR,
    HEADER_LENGTH,
    IMG_MSG_HEADER_FMT,
    IN_MSG_HEADER_FMT,
    IN_MSG_HEADER_LENGTH,
    MAX_ACC_REQ,
    MAX_STEER_REQ,
    MIN_ACC_REQ,
    MIN_STEER_REQ,
    OUT_MSG_HEADER_FMT,
)


class InvalidActionError(Exception):
    pass


class ActionInterface(object):
    """Action send interface. This class communicates with the simulator and
    sends action requests from the agent.

    :param str ip: ip address
    :param int port: port to bind to
    :param float max_steer: maximum steering request, bounded by 1.
    :param float min_steer: minimum steering request, bounded by -1.
    :param float max_accel: maximum acceleration request, bounded by 6.
    :param float min_accel: minimum acceleration request, bounded by -16.
    """

    def __init__(
        self,
        ip: str = "0.0.0.0",
        port: int = 7077,
        max_steer: float = 1.0,
        min_steer: float = -1.0,
        max_accel: float = MAX_ACC_REQ,
        min_accel: float = MIN_ACC_REQ,
    ) -> None:
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.max_steer = max_steer
        self.min_steer = min_steer
        self.max_accel = max_accel
        self.min_accel = min_accel
        self._validate_action_boundaries()

    def _validate_action_boundaries(self):
        if self.max_steer > MAX_STEER_REQ:
            logging.error(f"Steer boundary {self.max_steer} > {MAX_STEER_REQ}")
            raise InvalidActionError

        if self.min_steer < MIN_STEER_REQ:
            logging.error(f"Steer boundary {self.min_steer} < {MIN_STEER_REQ}")
            raise InvalidActionError

        if self.max_accel > MAX_ACC_REQ:
            logging.error(f"Acceleration boundary {self.max_accel} > {MAX_ACC_REQ}")
            raise InvalidActionError

        if self.min_accel < MIN_ACC_REQ:
            logging.error(f"Acceleration boundary {self.min_accel} < {MIN_ACC_REQ}")
            raise InvalidActionError

    def act(self, action: List[float]) -> None:
        """Send action request to the simulator.

        :param array-like action: action to send to the simulator in the form:
          [steering, acceleration], expected to be in the range (-1., 1.)
        """
        self._check_action(action=action)
        steer, acc = self._scale_action(action=action)
        bytes = struct.pack(OUT_MSG_HEADER_FMT, steer, acc, DRIVE_GEAR)
        self.sock.sendto(bytes, self.addr)

    def _scale_action(self, action: List[float]) -> Tuple[float, float]:
        """Scale the action"""
        steer, acc = action[0], action[1]
        steer *= self.max_steer if steer > 0 else (-1.0 * self.min_steer)
        acc *= self.max_accel if acc > 0 else (-1.0 * self.min_accel)
        return steer, acc

    def _check_action(self, action: List[float]) -> None:
        """Check that the action is valid with reference to the action space.

        :param array-like action: action to send to the simulator in the form:
          [steering, acceleration], expected to be in the range (-1., 1.)
        """
        if len(action) != 2:
            logging.error("Action should have length 2. Got: " + str(action))
            raise InvalidActionError

        if action[0] < -1.0 or action[0] > 1.0:
            logging.error("Invalid steering request: " + str(action[0]))
            raise InvalidActionError

        if action[1] < -1.0 or action[1] > 1.0:
            logging.error("Invalid acceleration request: " + str(action[1]))
            raise InvalidActionError


class PoseInterface(AbstractInterface):
    """Receives sensor data from the simulator. The data received is in the
    following format:

    [0,1,2] steering, gear, mode \n
    [3,4,5] velocity \n
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

    def __init__(self, ip: str = "0.0.0.0", port: int = 7078, data_elems: int = 30):
        addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(addr)
        self.data_elems = data_elems

    def start(self) -> None:
        """Starts a thread to listen for data from the simulator."""
        self.reset()
        self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()

    def get_data(self) -> np.array:
        """Return the most recent data received from the simulator.

        :return: data from the simulator
        :rtype: array of length self.data_elems
        """
        return copy.deepcopy(self.data)

    def reset(self) -> None:
        """Allocates memory for data receive."""
        self.data = np.zeros(shape=(self.data_elems,), dtype=float)

    def _receive(self) -> None:
        """Indefinitely wait for data from the simulator and unpack into an
        array.
        """
        while True:
            bytes, addr = self.sock.recvfrom(BUFFER_SIZE)
            assert len(bytes) == IN_MSG_HEADER_LENGTH
            self.data = np.asarray(struct.unpack(IN_MSG_HEADER_FMT, bytes))


class CameraInterface(AbstractInterface):
    """Receives images from the simulator.

    :param l2r.core.CameraConfig cfg: socket connection and interface parameters
    """

    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg

        # Build socket and connect
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.connect(cfg.sim_addr)

    @property
    def camera_name(self) -> str:
        """Return the name of the camera"""
        return self.cfg.name

    @property
    def camera_param_dict(self) -> Dict[str, str]:
        """Return camera configuration as a dictionary"""
        return self.cfg.get_sim_param_dict()

    def start(self) -> None:
        """Starts a thread to listen for images on."""
        self.img_dims = (self.cfg.Width, self.cfg.Height, 3)
        self.reset()
        self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()

    def get_data(self) -> np.array:
        """Return the most recent image(s) received from the simulator.

        :return: RGB image of shape (height, width, 3)
        :rtype: numpy.array
        """
        return copy.deepcopy(self.img)

    def reset(self) -> None:
        """Allocates memory for data receive."""
        self.img = np.zeros(shape=self.img_dims, dtype=float)

    def reconnect(self) -> None:
        """Reconnect to the socket"""
        try:
            _ = self.sock.recv(16, socket.MSG_DONTWAIT | socket.MSG_PEEK)
        except BlockingIOError:
            return

        self.sock.connect(self.cfg.sim_addr)

    def _receive(self) -> None:
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
        """Utility to determine dtype of the image."""
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
