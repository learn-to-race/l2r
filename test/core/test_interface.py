import unittest
from unittest import mock

from l2r.core import ActionInterface
from l2r.core.interfaces import InvalidActionError


class ActionInterfaceTest(unittest.TestCase):
    """
    Tests associated with the l2r action interface
    """

    _ip = "0.0.0.0"
    _port = "0.0.0.0"
    _addr = (_ip, _port)
    _max_acc = 3
    _min_acc = -2

    @mock.patch("socket.socket")
    def setUp(self, mock_socket: mock.MagicMock):
        self.action_interface = ActionInterface(
            ip=self._ip,
            port=self._port,
            max_accel=self._max_acc,
            min_accel=self._min_acc,
        )

    @mock.patch("struct.pack")
    @mock.patch("l2r.core.interfaces.ActionInterface._check_action")
    @mock.patch("l2r.core.interfaces.ActionInterface._scale_action")
    def test_act_sends_action(
        self,
        mock_scale_action: mock.MagicMock,
        mock_check_action: mock.MagicMock,
        mock_struct_pack: mock.MagicMock,
    ):
        """Validate an action is sent"""
        mock_struct_pack.return_value = b"bytes"
        mock_scale_action.return_value = 0.5, 0.5
        self.action_interface.act(action=[0.5, 0.5])
        self.action_interface.sock.sendto.assert_called_once_with(b"bytes", self._addr)

    def test_invalid_steering_fails(self):
        with self.assertRaises(InvalidActionError):
            self.action_interface._check_action(action=[5, 0.0])

    def test_invalid_acceleration_fails(self):
        with self.assertRaises(InvalidActionError):
            self.action_interface._check_action(action=[0.0, -6])

    def test_valid_steering_succeeds(self):
        self.action_interface._check_action(action=[0.5, -0.5])

    def test_scale_action(self):
        scaled_steer, scaled_accel = self.action_interface._scale_action(
            action=[0.6, 0.3]
        )
        self.assertAlmostEqual(scaled_steer, 0.6)
        self.assertAlmostEqual(scaled_accel, 0.9)


class PoseInterface(unittest.TestCase):
    """
    Tests associated with the l2r position interface
    """

    pass


class CameraInterface(unittest.TestCase):
    """
    Tests associated with the l2r camera interface
    """

    pass
