import rclpy
from .middleware import MockROS2Middleware


def main(args=None):
    rclpy.init(args=args)
    mid = MockROS2Middleware()

    mid.start_threads()

    mid.join()
