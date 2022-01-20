# ========================================================================= #
# Filename:                                                                 #
#    controller.py                                                          #
#                                                                           #
# Description:                                                              #
#    Contains the SimulatorController class to interface with the           #
#    simulator's via the API.                                               #
# ========================================================================= #
import json
import os
import subprocess
import time

from websocket import create_connection

from racetracks.mapping import level_2_simlevel

SHORT_DELAY = 0.05
POLL_DELAY = 0.5
MEDIUM_DELAY = 12

# Shell commands to launch and kill container
START_CMD = (
    "docker run -t --rm "
    "--user=ubuntu --gpus all "
    '--entrypoint="./ArrivalSim.sh" '
    "--net=host "
)

KILL_CMD = "docker kill "


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
    :param str sim_path: path of the simulator, if launching as a subprocess
    :param boolean start_container: start simulator container. Assumes
      unix-based os
    :param str image_name: name of the simulator's docker image
    :param str container_name: name of the container to create
    :param str user: username is required because simulator cannot be started
      with root privileges
    :param boolean evaluation: if evaluation mode, restrict certain sensors
    """

    def __init__(
        self,
        ip="0.0.0.0",
        port="16000",
        quiet=False,
        sim_version="ArrivalSim-linux-0.7.0.182276",
        sim_running=False,
        start_container=False,
        image_name="arrival-sim",
        container_name="racing-sim",
        sim_path=False,
        user="ubuntu",
        evaluation=False,
    ):
        """Constructor method"""
        self.sim_running = sim_running
        self.start_container = start_container
        self.sim_path = sim_path
        self.start = START_CMD + f"--name {container_name} {image_name}"
        self.kill = KILL_CMD + container_name
        self.addr = f"{ip}:{port}"
        self.user = user
        self.evaluation = evaluation

        if start_container or sim_path:
            _ = self.start_simulator()

        try:
            self.ws = create_connection(f"ws://{self.addr}")
        except ConnectionRefusedError:
            if start_container or sim_path:
                print("[Controller] Failed to connect to simulator. Trying again.")
                time.sleep(MEDIUM_DELAY // 2)
                self.ws = create_connection(f"ws://{self.addr}")
            else:
                print("[Controller] Simulator is not running. Aborting.")
                exit(1)

        self.sim_version = sim_version
        self.drive_mode_param = (
            "bIsManulMode" if sim_version.startswith("RoboraceMaps") else "bADModeInput"
        )
        self.id = 0
        self.quiet = quiet

    def __del__(self):
        """Stop the simulator on exit"""
        if self.start_container:
            print("[Controller] Controller destroyed. Tearing down simulator.")
            self.kill_simulator()

    def start_simulator(self):
        """Starts the simulator container, and briefly waits. If a connection
        error occurs, you may need to increase this delay.
        """
        assert not (
            self.start_container and self.sim_path
        ), "[Controller] Let L2R start EITHER a docker container OR a native simulator -- not both"

        if self.sim_running:
            print("[Controller] Assuming sim is running as a separate process")
            pass
        else:

            if self.start_container:
                print("[Controller] Starting simulator container")
                with open("/tmp/sim_log.txt", "w") as out:
                    subprocess.Popen(self.start, shell=True, stdout=out, stderr=out)

            elif self.sim_path:
                print("[Controller] Starting simulator")
                pth = os.path.join(self.sim_path, "ArrivalSim.sh")
                cmd = ["sudo", "-u", self.user, pth, "-openGL"]
                with open("/tmp/sim_log.txt", "w") as out:
                    self.simproc = subprocess.Popen(cmd, stdout=out)

            time.sleep(MEDIUM_DELAY)

        return

    def poll_sim(self):
        """Periodically check the simulator process"""
        if not self.sim_path:
            return

        while self.simproc.poll() is None:
            time.sleep(POLL_DELAY)

        raise Exception("Simulator process terminated.")

    def kill_simulator(self):
        """Kill the simulator"""
        if self.start_container:
            subprocess.Popen(self.kill, shell=True)
        elif self.sim_path:
            self.simproc.kill()

        time.sleep(MEDIUM_DELAY)

    def restart_simulator(self):
        """Restarts the simulator container"""
        self.kill_simulator()
        self.start_simulator()
        self._print("Reconnecting", force=True)
        self.ws = create_connection(f"ws://{self.addr}")

    def set_level(self, level):
        """Sets the simulator to a specified level (map).

        :param string level: name of the racetrack
        """
        level_name = level_2_simlevel(level, self.sim_version)
        self._print(f"Setting level to {level}")
        return self._send_msg(method="open_level", params={"level_name": level_name})

    def set_location(self, coords, rot, veh_id=0):
        """Sets a vehicle to a specific location on track.

        :param dict coords: desired ENU coordinates with keys [x, y, z]
        :param dict rot: rotation of the vehicle in degrees with keys
          [yaw, pitch, roll]
        :param int veh_id: identifier of the vehicle, defaults to 0
        """
        _ = self._send_msg(
            method="set_vehicle_position",
            params={"veh_id": veh_id, "location": coords, "rotation": rot},
        )

    def reset_level(self):
        """Resets the map to its default configuration. This does not reset
        the vehicle parameters, but it does set the vehicle to it's default
        pose.
        """
        self._print("Resetting level")
        _ = self._send_msg(method="reset_level")

    def get_level(self):
        """Get the active level of the simulator.

        :return: name of the simulator's active racetrack
        :rtype: str
        """
        return self._send_msg(method="get_level")

    def get_levels(self):
        """Gets the levels available in the simulator.

        :return: list of the levels available in the simulator
        :rtype: list of str
        """
        return self._send_msg(method="get_levels")

    def get_position(self, veh_id=0):
        """Get vehicle's current position.

        :param int veh_id: identifier of the vehicle, defaults to 0
        :return: ENU coordinates of the vehicle [East, North, Up], vehicle
          heading
        :rtype: list of floats, float
        """
        result = self._send_msg(method="get_vehicle_state", params={"veh_id": veh_id})
        return result["pos_xyz"], result["yaw"]

    def get_vehicle_params(self, veh_id=0):
        """Get the active parameters of a vehicle.

        :param int veh_id: identifier of the vehicle, defaults to 0
        :return: a dictionary of the vehicle's parameters
        :rtype: dict
        """
        return self._send_msg(method="get_vehicle_params", params={"veh_id": veh_id})

    def set_vehicle_params(self, parameters, veh_id=0):
        """Set parameters of a vehicle.

        :param list parameters: name value pairs, list of dictionaries
        :param int veh_id: identifier of the vehicle, defaults to 0
        """
        self._print(f"Setting vehicle parameters for vehicle: {veh_id}")
        _ = self._send_msg(
            method="set_vehicle_params",
            params={"veh_id": veh_id, "parameters": parameters},
        )

    def reset_vehicle_params(self, veh_id=0):
        """Reset a vehicle's parameters to default.

        :param int veh_id: identifier of the vehicle, defaults to 0
        """
        self._print(f"Resetting vehicle parameters for vehicle: {veh_id}")
        _ = self._send_msg(
            method="reset_default_vehicle_params", params={"veh_id": veh_id}
        )

    def get_active_vehicles(self):
        """Get a list of the active vehicles in the simulator.

        :return: list of active vehicles in the following format:
          {"veh_id": <veh_id>, "vehicle_class_name": <vehicle_class_name>}
        :rtype: list of dictionaries
        """
        return self._send_msg(method="get_level_vehicles")

    def get_vehicle_classes(self):
        """Get a list of the vehicle classes.

        :return: list of the vehicle classes
        :rtype: list
        """
        return self._send_msg(method="get_vehicle_classes")

    def get_sensors_params(self, veh_id=0):
        """Get the parameters of each sensor for a specified vehicle.

        :param int veh_id: identifier of the vehicle, defaults to 0
        :return: parameters for each sensor
        :rtype: list
        """
        return self._send_msg(method="get_sensors_params", params={"veh_id": veh_id})

    def get_sensor_params(self, sensor, veh_id=0):
        """Get the parameters of a specified sensor of a specified vehicle.

        :param str sensor: name of sensor to get the params for
        :param int veh_id: identifier of the vehicle, defaults to 0
        :return: the sensor's parameters
        :rtype: dict
        """
        return self._send_msg(
            method="get_sensor_params", params={"veh_id": veh_id, "sensor_name": sensor}
        )

    def set_sensor_param(self, sensor, name, value, veh_id=0):
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
        )

    def set_sensor_params(self, sensor, params, veh_id=0):
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
        )

    def get_vehicle_driver_params(self, veh_id=0):
        """Get the parameters of the sensor 'ArrivalVehicleDriver'.

        :param int veh_id: identifier of the vehicle, defaults to 0
        :return: parameters of the vehicle driver
        :rtype: dict
        """
        return self._send_msg(
            method="get_sensor_params",
            params={"veh_id": veh_id, "sensor_name": "ArrivalVehicleDriver"},
        )

    def get_driver_mode(self, veh_id=0):
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

    def set_mode_ai(self, veh_id=0):
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

    def set_mode_manual(self, veh_id=0):
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

    def set_api_udp(self, veh_id=0):
        """Set a specified vehicle API class to VApiUdp to allow the RL
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
        )

    def enable_sensor(self, sensor_name, veh_id=0):
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

    def disable_sensor(self, sensor_name, veh_id=0):
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

    def _send_msg(self, method, params=None):
        """Helper routine to send jsonprc messages.

        :param str method: specified method
        :param dict params: parameters to send
        :return: response from the simulator
        :rtype: varies
        """
        self.id += 1
        msg = {"jsonrpc": "2.0", "params": {}, "id": self.id}
        msg["method"] = method
        if params:
            msg["params"] = params
        try:
            self.ws.send(json.dumps(msg))
            msg = json.loads(self.ws.recv())
            time.sleep(SHORT_DELAY)
            return msg["result"]
        except Exception:
            print(msg)
            print("[Controller] Disconnected from simulator")
            exit(-1)
        return msg["result"]

    def _print(self, msg, force=False):
        """Helper print routine"""
        if not self.quiet and not force:
            print(f"[Controller] {msg}")
