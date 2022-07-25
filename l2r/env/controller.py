import json
import logging
import time
from typing import Any, Dict, List, Tuple

from websocket import create_connection

from l2r.constants import MEDIUM_DELAY
from l2r.track import level_2_simlevel


class SimulatorConnectionError(Exception):
    pass


class SimulatorApiError(Exception):
    pass


class SimulatorController(object):
    """This class is used to communicate with the simulator which is
    accessible via Websocket. The methods in this class allow one to change
    the position of the vehicle, set the map, reset the map, change the
    settings (such as turning the camera on), etc.

    :param str ip: IP address of the simulator
    :param str port: port to connect to
    :param boolean quiet: flag to show print statements about changes being
      made to the simulator
    :param str sim_version: version of the simulator
    :param boolean evaluation: if evaluation mode, restrict certain sensors
    :param boolean autoconnect: if true, attempt to connect in constructor
    """

    def __init__(
        self,
        ip="0.0.0.0",
        port="16000",
        quiet=False,
        sim_version="ArrivalSim-linux-0.7.1.188691",
        evaluation=False,
        drive_mode_param="bADModeInput",
        auto_connect=True,
    ):
        """Constructor method"""
        self.addr = f"{ip}:{port}"
        self.evaluation = evaluation
        self.sim_version = sim_version
        self.drive_mode_param = drive_mode_param
        self.id = 0
        self.quiet = quiet

        if auto_connect:
            self.connect_to_simulator()

    def heartbeat(self) -> None:
        # TODO
        pass

    def connect_to_simulator(self) -> None:
        """Attempt to connect to the simulator"""
        logging.info("Attempting to connect to simulator...")

        try:
            self.conn = create_connection(f"ws://{self.addr}")
        except ConnectionRefusedError:
            logging.warning(f"Failed to connect to simulator at addr: {self.addr}")
            logging.warning("Retrying in 10 seconds...")
            time.sleep(10)
            try:
                self.conn = create_connection(f"ws://{self.addr}")
            except ConnectionRefusedError:
                logging.error(f"Failed to connect to simulator at addr: {self.addr}")
                raise SimulatorConnectionError

        logging.info("Created connection.")

    def _send_msg(
        self, method: str, params: Dict[str, Any] = {}, delay: float = 0.1
    ) -> Any:
        """Helper routine to send jsonprc messages.

        :param str method: specified method
        :param dict params: parameters to send
        :param int delay: time delay after sending
        :return: response from the simulator
        :rtype: varies
        """
        self.id += 1
        msg = {"jsonrpc": "2.0", "params": params, "id": self.id, "method": method}

        # Send message
        self.conn.send(json.dumps(msg))

        # Receive response
        response = json.loads(self.conn.recv())

        if "result" not in response:
            raise SimulatorApiError

        if delay:
            time.sleep(delay)

        return response["result"]

    def _print(self, msg: str) -> None:
        """Helper print routine"""
        if not self.quiet:
            logging.info(f"[Controller] {msg}")

    def set_level(self, level: str) -> Any:
        """Sets the simulator to a specified level (map).

        :param string level: name of the racetrack
        """
        level_name = level_2_simlevel(level, self.sim_version)
        self._print(f"Setting level to {level}")
        return self._send_msg(
            method="open_level", params={"level_name": level_name}, delay=MEDIUM_DELAY
        )

    def set_location(
        self, coords: Dict[str, float], rot: Dict[str, float], veh_id: int = 0
    ) -> None:
        """Sets a vehicle to a specific location on track.

        :param dict coords: desired ENU coordinates with keys [x, y, z]
        :param dict rot: rotation of the vehicle in degrees with keys
          [yaw, pitch, roll]
        :param int veh_id: identifier of the vehicle, defaults to 0
        """
        _ = self._send_msg(
            method="set_vehicle_position",
            params={"veh_id": veh_id, "location": coords, "rotation": rot},
            delay=MEDIUM_DELAY,
        )

    def reset_level(self) -> None:
        """Resets the map to its default configuration. This does not reset
        the vehicle parameters, but it does set the vehicle to it's default
        pose.
        """
        self._print("Resetting level")
        _ = self._send_msg(method="reset_level")

    def get_level(self) -> str:
        """Get the active level of the simulator.

        :return: name of the simulator's active racetrack
        :rtype: str
        """
        return self._send_msg(method="get_level")

    def get_levels(self) -> List[str]:
        """Gets the levels available in the simulator.

        :return: list of the levels available in the simulator
        :rtype: list of str
        """
        return self._send_msg(method="get_levels")

    def get_position(self, veh_id: int = 0) -> Tuple[List[float], float]:
        """Get vehicle's current position.

        :param int veh_id: identifier of the vehicle, defaults to 0
        :return: ENU coordinates of the vehicle [East, North, Up], vehicle
          heading
        :rtype: list of floats, floatx
        """
        result = self._send_msg(method="get_vehicle_state", params={"veh_id": veh_id})
        return result["pos_xyz"], result["yaw"]

    def get_vehicle_params(self, veh_id: int = 0) -> Dict[str, Any]:
        """Get the active parameters of a vehicle.

        :param int veh_id: identifier of the vehicle, defaults to 0
        :return: a dictionary of the vehicle's parameters
        :rtype: dict
        """
        return self._send_msg(method="get_vehicle_params", params={"veh_id": veh_id})

    def set_vehicle_params(self, parameters, veh_id: int = 0) -> None:
        """Set parameters of a vehicle.

        :param list parameters: name value pairs, list of dictionaries
        :param int veh_id: identifier of the vehicle, defaults to 0
        """
        self._print(f"Setting vehicle parameters for vehicle: {veh_id}")
        _ = self._send_msg(
            method="set_vehicle_params",
            params={"veh_id": veh_id, "parameters": parameters},
        )

    def reset_vehicle_params(self, veh_id: int = 0) -> None:
        """Reset a vehicle's parameters to default.

        :param int veh_id: identifier of the vehicle, defaults to 0
        """
        self._print(f"Resetting vehicle parameters for vehicle: {veh_id}")
        _ = self._send_msg(
            method="reset_default_vehicle_params", params={"veh_id": veh_id}
        )

    def get_active_vehicles(self) -> List[Dict[str, str]]:
        """Get a list of the active vehicles in the simulator.

        :return: list of active vehicles in the following format:
          {"veh_id": <veh_id>, "vehicle_class_name": <vehicle_class_name>}
        :rtype: list of dictionaries
        """
        return self._send_msg(method="get_level_vehicles")

    def get_vehicle_classes(self) -> List[str]:
        """Get a list of the vehicle classes.

        :return: list of the vehicle classes
        :rtype: list
        """
        return self._send_msg(method="get_vehicle_classes")

    def get_sensors_params(self, veh_id: int = 0):
        """Get the parameters of each sensor for a specified vehicle.

        :param int veh_id: identifier of the vehicle, defaults to 0
        :return: parameters for each sensor
        :rtype: list
        """
        return self._send_msg(method="get_sensors_params", params={"veh_id": veh_id})

    def get_sensor_params(self, sensor: str, veh_id: int = 0) -> Dict[str, Any]:
        """Get the parameters of a specified sensor of a specified vehicle.

        :param str sensor: name of sensor to get the params for
        :param int veh_id: identifier of the vehicle, defaults to 0
        :return: the sensor's parameters
        :rtype: dict
        """
        return self._send_msg(
            method="get_sensor_params", params={"veh_id": veh_id, "sensor_name": sensor}
        )

    def set_sensor_param(
        self, sensor: str, name: str, value: str, veh_id: int = 0
    ) -> int:
        """Set a specified parameter to a specified value of a specified
        sensor on a specified vehicle.

        :param str sensor: name of sensor to set
        :param str name: name of the parameter to set
        :param str value: value of the parameter to set
        :param int veh_id: identifier of the vehicle, defaults to 0
        :return: 0, if successful
        :rtype: int
        """
        self._print(f"Setting: {name} to: {value} for vehicle: {veh_id}")

        if self.evaluation:
            if name == "Format" and value.startswith("Segm"):
                raise ValueError("Illegal parameters detected")

        return self._send_msg(
            method="set_sensor_param",
            params={
                "veh_id": veh_id,
                "sensor_name": sensor,
                "parameter_name": name,
                "value": value,
            },
            delay=MEDIUM_DELAY,
        )

    def set_sensor_params(self, sensor: str, params: Dict[str, Any], veh_id: int = 0):
        """Set specified parameters of a specified sensor on a specified
        vehicle.

        :param str sensor: name of sensor to set
        :param dict parameters: name value pairs of the parameters to set
        :param int veh_id: identifier of the vehicle, defaults to 0
        """
        self._print(f"Setting: {sensor} parameters for vehicle: {veh_id}")
        parameters = [{"name": k, "value": v} for k, v in params.items()]

        if self.evaluation:
            for p in parameters:
                if p["name"] == "Format" and p["value"].startswith("Segm"):
                    raise ValueError("Illegal parameters detected")

        _ = self._send_msg(
            method="set_sensor_params",
            params={"veh_id": veh_id, "sensor_name": sensor, "parameters": parameters},
            delay=MEDIUM_DELAY,
        )

    def get_vehicle_driver_params(self, veh_id: int = 0):
        """Get the parameters of the sensor 'ArrivalVehicleDriver'.

        :param int veh_id: identifier of the vehicle, defaults to 0
        :return: parameters of the vehicle driver
        :rtype: dict
        """
        return self._send_msg(
            method="get_sensor_params",
            params={"veh_id": veh_id, "sensor_name": "ArrivalVehicleDriver"},
        )

    def get_driver_mode(self, veh_id: int = 0):
        """Get the mode of a specified vehicle.

        :param int veh_id: identifier of the vehicle, defaults to 0
        :return: true if manual mode, false if AI mode
        :rtype: boolean
        """
        return self._send_msg(
            method="get_sensor_param",
            params={
                "veh_id": veh_id,
                "sensor_name": "ArrivalVehicleDriver",
                "parameter_name": self.drive_mode_param,
            },
        )

    def set_mode_ai(self, veh_id: int = 0):
        """Sets a specified vehicle mode to "AI".

        :param int veh_id: identifier of the vehicle, defaults to 0
        """
        self._print(f"Setting to AI mode for vehicle: {veh_id}")
        _ = self._send_msg(
            method="set_sensor_param",
            params={
                "veh_id": veh_id,
                "sensor_name": "ArrivalVehicleDriver",
                "parameter_name": self.drive_mode_param,
                "value": True,
            },
        )

    def set_mode_manual(self, veh_id: int = 0):
        """Sets a specified vehicle mode to "manual".

        :param int veh_id: identifier of the vehicle, defaults to 0
        """
        self._print(f"Setting to manual mode for vehicle: {veh_id}")
        _ = self._send_msg(
            method="set_sensor_param",
            params={
                "veh_id": veh_id,
                "sensor_name": "ArrivalVehicleDriver",
                "parameter_name": self.drive_mode_param,
                "value": False,
            },
        )

    def set_api_udp(self, veh_id: int = 0):
        """Set a specified vehicle API class to VApiUdp to allow the
        agent to control it.

        :param int veh_id: identifier of the vehicle, defaults to 0
        """
        self._print(f"Setting to RL mode for vehicle: {veh_id}")
        _ = self._send_msg(
            method="set_sensor_param",
            params={
                "veh_id": veh_id,
                "sensor_name": "ArrivalVehicleDriver",
                "parameter_name": "DriverAPIClass",
                "value": "VApiUdp",
            },
            delay=MEDIUM_DELAY,
        )

    def enable_sensor(self, sensor_name: str, veh_id: int = 0):
        """Activates a specified sensor on a specified vehicle.

        :param str sensor_name: sensor to be activated
        :param int veh_id: identifier of the vehicle, defaults to 0
        :return: 0, if successful
        :rtype: int
        """
        self._print(f"Enabling {sensor_name}")
        return self._send_msg(
            method="enable_sensor",
            params={"veh_id": veh_id, "sensor_name": sensor_name},
        )

    def disable_sensor(self, sensor_name: str, veh_id: int = 0):
        """Disables a specified sensor on a specified vehicle.

        :param str sensor_name: sensor to be deactivated
        :param int veh_id: identifier of the vehicle, defaults to 0
        :return: 0, if successful
        :rtype: int
        """
        self._print(f"Disabling {sensor_name}")
        return self._send_msg(
            method="disable_sensor",
            params={"veh_id": veh_id, "sensor_name": sensor_name},
        )
