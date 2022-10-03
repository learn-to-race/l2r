import zmq
import socket
import struct

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

class Connection:
    def __init__(self):
        pass
    def send(self, data):
        pass
    def recv(self):
        pass

class L2RCamConnection(Connection):
    def __init__(self):
        host = ""
        port = 18008
        self.ip_port = "tcp://127.0.0.1:{}".format(port)
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.PUB)
        self.sock.bind(self.ip_port)
    def send(self, data):
        self.sock.send(data)
        
    def recv(self):
        pass

class L2RPoseConnection(Connection):
    def __init__(self):
        host = ""
        port = 17078
        self.ip_port = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
       
    def send(self, data):
        self.sock.sendto(data, self.ip_port)

    def recv(self):
        print("not implemented")
        pass

class L2RActionConnection(Connection):
    def __init__(self):
        host = ""
        port = 17077
        self.ip_port = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(self.ip_port)

    def send(self, data):
        pass
    def recv(self):
        bytes, _ = self.sock.recvfrom(BUFFER)
        return bytes

class ArrivalPoseConnection(Connection):
    def __init__(self):
        host = ""
        port = 7078
        self.ip_port = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(self.ip_port)

    def send(self, data):
        pass
    def recv(self):
        bytes, _ = self.sock.recvfrom(BUFFER)
        return bytes

class ArrivalCamConnection(Connection):
    def __init__(self):
        host = ""
        port = 8008
        self.ip_port = "tcp://127.0.0.1:{}".format(port)
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, b'')
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.connect(self.ip_port)

    def send(self, data):
        pass
    def recv(self):
        while True:
            try:
                rawbuf = self.sock.recv(0)
                
                return rawbuf
            except zmq.ZMQError as e:
                if e.errno != zmq.EAGAIN:
                    raise

class ArrivalActionConnection(Connection):
    def __init__(self):
        host = ""
        port = 7077
        self.ip_port = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, data):
        self.sock.sendto(data, self.ip_port)
    def recv(self):
        pass