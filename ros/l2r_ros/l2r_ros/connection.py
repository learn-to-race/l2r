import time
import zmq
import socket
from rclpy.node import Node

# Socket receive size
BUFFER = 1024


class Connection:
    def __init__(self):
        pass

    def send(self, data):
        print("send not implemented for " + self.__name__)

    def recv(self):
        print("send not implemented for " + self.__name__)


class L2RCamConnection(Connection):
    def __init__(self, ip_port="tcp://127.0.0.1:18008"):
        self.ip_port = ip_port
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.PUB)
        self.sock.bind(self.ip_port)

    def send(self, data):
        self.sock.send(data)


class L2RPoseConnection(Connection):
    def __init__(self, host="", port=17078):
        self.ip_port = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, data):
        self.sock.sendto(data, self.ip_port)


class L2RActionConnection(Connection):
    def __init__(self, host="", port=17077):
        self.ip_port = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(self.ip_port)

    def recv(self):
        bytes, _ = self.sock.recvfrom(BUFFER)
        return bytes


class ArrivalPoseConnection(Connection):
    def __init__(self, host="", port=7078):
        self.ip_port = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(self.ip_port)

    def recv(self):
        bytes, _ = self.sock.recvfrom(BUFFER)
        return bytes


class ArrivalCamConnection(Connection):
    def __init__(self, ip_port="tcp://127.0.0.1:8008"):
        self.ip_port = ip_port
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.connect(self.ip_port)

    def recv(self):
        while True:
            try:
                rawbuf = self.sock.recv(0)

                return rawbuf
            except zmq.ZMQError as e:
                if e.errno != zmq.EAGAIN:
                    raise


class ArrivalActionConnection(Connection):
    def __init__(self, host="", port=7077):
        self.ip_port = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, data):
        self.sock.sendto(data, self.ip_port)


class ROS2PubConnection(Connection, Node):
    def __init__(self, node_name, topic, data_type):
        Node.__init__(self, node_name)
        self.publisher_ = self.create_publisher(data_type, topic, 10)
        self.i = 0
        self.data_type = data_type
        self.topic = topic

    def send(self, data):
        msg = self.data_type()
        msg.data = data
        self.get_logger().info(f"Pub topic={self.topic}, len={len(msg.data)}")
        self.publisher_.publish(msg)


class ROS2SubConnection(Connection, Node):
    def __init__(self, node_name, topic, data_type):
        Node.__init__(self, node_name)
        self.subscription = self.create_subscription(
            data_type, topic, self.listener_callback, 10
        )
        self.subscription
        self.updated = False
        self.data = None
        self.topic = topic

    def listener_callback(self, msg):
        self.get_logger().info(f"Sub topic={self.topic}, len={len(msg.data)}")
        self.data = msg.data
        self.updated = True

    def recv(self):
        while not self.updated:
            time.sleep(0.01)
        return self.data
