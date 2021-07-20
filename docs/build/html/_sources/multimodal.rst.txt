
Multimodal
==========

Setting the Environment
-----------------------

The multimodal option provides visual features to the agent in an identical manner to the `visual only <vision.html>`_ feature set, but it also includes data from the vehicle's `IMU sensor <https://en.wikipedia.org/wiki/Inertial_measurement_unit>`_ along with a few other pieces of data. We expect that the performance of agents with multimodal sensory data to be better than that of visual only agents. To set the environment to multimodal, simply modify the ``multimodal`` parameter to ``True`` in the ``configs/params.yaml`` file:

.. code-block:: yaml

   env_kwargs:
      multimodal: True       # when True, both images and pose data are provided to agent
      max_timesteps: 5000
      ...

Environment Observations
------------------------

Setting this parameter to ``True`` will change the return type of the ``step()`` method of the `RacingEnv <l2r.envs.html#module-roboracer.envs.env>`_ class to return a ``spaces.Dict`` containing:

Track ID
  a numeric identifier of the current track, relevant for multi-track training

Camera Images
  a numpy array of shape **(image_width, image_height, 3)**

Additional Data
  a numpy array of shape (30,) with the following data:

  +----------------+-------------------------------------------------------+
  | Array Indicies | Data                                                  |
  +================+=======================================================+
  | 0              | steering request                                      |
  +----------------+-------------------------------------------------------+
  | 1              | gear request                                          |
  +----------------+-------------------------------------------------------+
  | 2              | mode                                                  |
  +----------------+-------------------------------------------------------+
  | 3,4,5          | direction velocity in m/s                             |
  +----------------+-------------------------------------------------------+
  | 6,7,8          | directional acceleration in m/s^2                     |
  +----------------+-------------------------------------------------------+
  | 9,10,11        | directional angular velocity                          |
  +----------------+-------------------------------------------------------+
  | 12,13,14       | vehicle yaw, pitch, and roll, respectively            |
  +----------------+-------------------------------------------------------+
  | 15,16,17       | center of vehicle coordinates in the format (y, x, z) |
  +----------------+-------------------------------------------------------+
  | 18,19,20,21    | wheel revolutions per minute (per wheel)              |
  +----------------+-------------------------------------------------------+
  | 22,23,24,25    | wheel braking (per wheel)                             |
  +----------------+-------------------------------------------------------+
  | 26,27,28,29    | wheel torque (per wheel)                              |
  +----------------+-------------------------------------------------------+

