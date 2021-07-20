
Creating Custom Sensor Configurations
=====================================

Overview
********

You can create arbitrary configurations of vehicle sensors with our environment. The following sensors are supported and can be placed, if applicable, at any location relative to the vehicle:

- RGB cameras
- Depth cameras
- Ground truth segmentation cameras
- Fisheye cameras
- Ray trace LiDARs
- Depth 2D LiDARs
- Radars

We will go over a brief tutorial on how to create and configure a custom camera using our environment.

.. note::
   The L2R `SimulatorController <l2r.core.html#l2r.core.controller.SimulatorController>`_ cannot create new sensors in the simulator, but it can enable sensors and modify their configuration.


Creating Sensors
****************

By default, the simulator will load the previously saved vehicle configuration.

(Option 1) User Interface
~~~~~~~~~~~~~~~~~~~~~~~~~

1. Run the simulator and select an arbitrary map
2. On the right panel, select "Vehicle Sensor Settings"
3. In the top right, select "Add Sensor"
4. Select the sensor you wish to create
5. Configure the sensor to your choosing
6. Press "ESC" then select "Save All" or "Save Vehicle" in the top panel

Optionally, you can export the vehicle configuration by selecting "Vehicle Settings" in the right panel, scrolling down, and exports to a JSON or Yaml file.


(Option 2) Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our repository contains a default `vehicle configuration file <https://github.com/hermgerm29/learn-to-race/blob/main/l2r/configs/l2r_vehicle_conf.yaml>`_ which serves as a valuable reference tool.

1. Simply modify parameters or add new sensors as you see fit
2. Add your file to the appropriate simulator directory, for example, ``ArrivalSim-linux-0.7.0.182276/LinuxNoEditor/Engine/Binaries/Linux/l2r_vehicle_conf.yaml``
3. Run the simulator and select an arbitrary map
4. Select "Vehicle Settings" in the right panel
5. Scroll down, and load your updated configuration
6. Press "ESC" then select "Save All" or "Save Vehicle" in the top panel

For example, we can add a rear facing camera named "CameraRearRGB" by appending this camera configuration to `l2r_vehicle_conf.yaml <https://github.com/hermgerm29/learn-to-race/blob/main/l2r/configs/l2r_vehicle_conf.yaml>`_ then following the steps above.

.. code-block:: yaml

    version: 1.1
    cameras
      - name: CameraRearRGB
        enabled: false
        model: ARRIVAL Generic Camera
        pose:
          x: -1.000  # 1m behind the reference point, slightly outside the vehicle
          y: 0.000
          z: 0.500   # 50cm above the reference point
          pitch: 0.0
          roll: 0.0
          yaw: 180.0  # rotate the camera around the Z-axis by 180 degrees
        transport: 
          zmq: tcp://0.0.0.0:9999


Using Your New Sensor
*********************

To use your newly created camera, you simply need to add it to your parameter configuration file. The `random action baseline configuration <https://github.com/hermgerm29/learn-to-race/blob/main/l2r/configs/params_random.yaml>`_ serves as a good reference. Once added, the observation returned from the environment's ```step()``` method will include the new images which are accessible via the camera's name. For example:


.. code-block:: yaml

    env_kwargs:
      cameras:
        CameraRearRGB:
          Addr: 'tcp://0.0.0.0:8008'  # make sure this address is unique
          Format: ColorBGR8
          FOVAngle: 90
          Width: 512
          Height: 384
          bAutoAdvertise: True

Rear facing images will be available like:

.. code-block:: python

    while not done:
        action = self.select_action()
        state, reward, done, info = self.env.step(action)
        rear_image = state['CameraRearRGB'] # numpy array
