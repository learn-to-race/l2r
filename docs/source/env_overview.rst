
Environment Overview
====================


Introduction
-------------
`Learn-to-Race <https://learn-to-race.org>`_ (L2R) is an `OpenAI-gym` compliant, multimodal control environment, where agents learn how to race. 

Unlike many simplistic learning environments, ours is built around high-fidelity simulators, based on Unreal Engine 4, such as the Arrival Autonomous Racing Simulator—featuring full software-in-the-loop (SIL) and even hardware-in-the-loop (HIL) simulation capabilities. This simulator has played a key role in bringing autonomous racing technology to real life in the `Roborace series <https://roborace.com/>`_, the world’s first extreme competition of teams developing self-driving AI. The L2R framework is the official training environment for Carnegie Mellon University's Roborace team, the first North American team to join the international challenge.

Autonomous Racing poses a significant challenge for artificial intelligence, where agents must make accurate and high-risk control decisions in real-time, while operating autonomous systems near their physical limits. The L2R framework presents *objective-centric* tasks, rather than providing abstract rewards, and provides numerous quantitative metrics to measure the racing performance and trajectory quality of various agents.

For more information, please read a couple of our conference papers:

- `Learn-to-Race: A Multimodal Control Environment for Autonomous Racing <https://arxiv.org/abs/2103.11575>`_

- `Safe Autonomous Racing via Approximate Reachability on Ego-vision <https://arxiv.org/abs/2110.07699>`_

Baseline Models
---------------
We provide three baseline models, to demonstrate how to use L2R: a random agent, a model predictive control (MPC) agent, a `Soft Actor-Critic <https://arxiv.org/abs/1801.01290v1>`_ reinforcement learning (RL) agent, and an imitation learning agent based on the MPC's demonstrations.

The `RandomAgent <getting_started.html#basic-example>`_ executes actions, completely at random. The `MPCAgent <getting_started.html#basic-example>`_ recursively plans trajectories according to a dynamics model of the vehicle, then executes actions according to the current plan. The `SACAgent <getting_started.html#basic-example>`_ is a learning-based method, which relies on the optimisation of a stochastic policy, model-free.

Action Space
------------
In the Learn-to-Race tasks, agents execute actions in the environment, according to steering and acceleration control, each supported by the simulator on a continuous range from -1.0 to 1.0.

.. table::
   :widths: auto

   ============ ============ ==============
   Action       Type         Range
   ============ ============ ==============
   Steering     Continuous   [-1.0, 1.0]
   
   Acceleration Continuous   [-1.0, 1.0]
   ============ ============ ==============

To provide additional flexibility for learning-based approaches, the L2R framework supports a scaled action space of [-1.0, 1.0] for steering control and [-16.0, 6.0] for acceleration control, by default. You can modify the boundaries of the action space by changing the parameters for `env_kwargs.action_if_kwargs` in `params-env.yaml`.

Negative acceleration commands perform braking actions, until the vehicle is stationary. If negative acceleration commands continue after the vehicle is stationary, the vehicle will reverse.

While you *can* change the gear, in practice we suggest forcing the agent to stay in drive since the others would not be advantageous in completing the tasks we present (we don't include it as a part of the action space). Note that negative acceleration values will brake the vehicle.

Observation Space
-----------------
We offer two high-level settings for the observation space: `vision-only <vision.html>`_ and `multimodal <multimodal.html>`_. In both, the agent receives RGB images from the vehicle's front-facing camera, examples below. In the latter, the environment also provides sensor data, including pose data from the vehicle's IMU sensor.

.. raw:: html

    <div style="text-align: center;">
      <figure style="display:inline-block; width:42%;">
        <img src='_static/sample_image_lvms.png' alt='missing'/ width=92%>
        <figcaption style="padding: 10px 15px 15px;"><i>Sample image from the Las Vegas track</i></figcaption>
      </figure>
      <figure style="display:inline-block; width:42%;">
        <img src='_static/sample_image_thruxton.png' alt='missing' width=92%/>
        <figcaption style="padding: 10px 15px 15px;"><i>Sample image from the Thruxton track</i></figcaption>
      </figure>
    </div>


Customizable Sensor Configurations
----------------------------------
One of the key features of this environment is the ability to create arbitrary configurations of vehicle sensors. This provides users a rich sandbox for multimodal, learning based approaches. The following sensors are supported and can be placed, if applicable, at any location relative to the vehicle:

- RGB cameras
- Depth cameras
- Ground truth segmentation cameras
- Fisheye cameras
- Ray trace LiDARs
- Depth 2D LiDARs
- Radars

Additionally, these sensors are parameterized and can be customized further; for example, cameras have modifiable image size, field-of-view, and exposure. Default sensor configurations are provided in `env_kwargs.cameras` and `sim_kwargs` in `params-env.yaml`. We provide further description on `sensor configuration <sensors.html#creating-custom-sensor-configurations>`_

.. raw:: html

    <div style="text-align: center;">
      <figure style="display:inline-block; width:32%;">
        <img src='_static/sample_vehicle_imgs/CameraLeftRGB.png' alt='missing'/ width=92%>
        <figcaption style="padding: 3px 3px 3px;"><i>Left facing</i></figcaption>
      </figure>
      <figure style="display:inline-block; width:32%;">
        <img src='_static/sample_vehicle_imgs/CameraFrontRGB.png' alt='missing'/ width=92%>
        <figcaption style="padding: 3px 3px 3px;"><i>Front facing</i></figcaption>
      </figure>
      <figure style="display:inline-block; width:32%;">
        <img src='_static/sample_vehicle_imgs/CameraRightRGB.png' alt='missing'/ width=92%>
        <figcaption style="padding: 3px 3px 3px;"><i>Right facing</i></figcaption>
      </figure>
    </div>

.. raw:: html

    <div style="text-align: center;">
      <figure style="display:inline-block; width:32%;">
        <img src='_static/sample_vehicle_imgs/CameraLeftSegm.png' alt='missing'/ width=92%>
        <figcaption style="padding: 3px 3px 3px;"></figcaption>
      </figure>
      <figure style="display:inline-block; width:32%;">
        <img src='_static/sample_vehicle_imgs/CameraFrontSegm.png' alt='missing'/ width=92%>
        <figcaption style="padding: 3px 3px 3px;"></figcaption>
      </figure>
      <figure style="display:inline-block; width:32%;">
        <img src='_static/sample_vehicle_imgs/CameraRightSegm.png' alt='missing'/ width=92%>
        <figcaption style="padding: 3px 3px 20px;"></figcaption>
      </figure>
    </div>


You can create cameras anywhere relative to the vehicle, allowing unique points-of-view such as a birdseye perspective which we include in the vehicle configuration file. 

.. raw:: html

    <div style="text-align: center;">
      <figure style="display:inline-block; width:42%;">
        <img src='_static/sample_vehicle_imgs/CameraBirdsEye.png' alt='missing'/ width=92%>
        <figcaption style="padding: 3px 3px 3px;"></figcaption>
      </figure>
      <figure style="display:inline-block; width:42%;">
        <img src='_static/sample_vehicle_imgs/CameraBirdsSegm.png' alt='missing'/ width=92%>
        <figcaption style="padding: 3px 3px 20px;"></figcaption>
      </figure>
    </div>

For more information, see `Creating Custom Sensor Configurations <sensors.html#creating-custom-sensor-configurations>`_

Whereas we encourage the use of all sensors for training and experimentation, only the CameraFrontRGB camera will be used for official L2R task evaluation, e.g., in our Learn-to-Race Autonomous Racing Virtual Challenges.

Interfaces and configuration
----------------------------

The environment interacts with additional modules in the overall L2R framework, such as the racetrack mapping (for loading and configuring the world), the Controller (which interfaces with an underlying simulator or vehicle stack) and the Tracker (which tracks the vehicle state and measures progress along the racetrack).

Whereas each of these interfaces can be further configured from `params-env.yaml`, the default values provided will be used for official L2R task evaluation, e.g., in our Learn-to-Race Autonomous Racing Virtual Challenges.

- Tracker (l2r/core/tracker.py), configured via `env_kwargs` in `configs/params-env.yaml`

- Controller (l2r/core/controller.py), configured via `env_kwargs.controller_kwargs` in `configs/params-env.yaml`

- racetrack (l2r/racetracks/mapping.py), configured via `sim_kwargs` in `params-env.yaml`

Racetracks
----------
We currently support two racetracks in our environment, both of which emulate real-world tracks. The first is the Thruxton Circuit, modeled off the track at the Thruxton Motorsport Centre in the United Kingdom. The second is the Anglessey National Circuit, located in Ty Croes, Anglesey, Wales. 

Additional tracks are used for evaluation, e.g., in open Learn-to-Race Autonomous Racing Virtual Challenges, such as the Vegas North Road track, located at Las Vegas Motor Speedway in the United States.

We will continue to add more racetracks in the future, for both training an evaluation.

Research Citation
-----------------

Please cite this work if you use L2R as a part of your research.

.. code-block:: text

  @inproceedings{herman2021learn,
              title={Learn-to-Race: A Multimodal Control Environment for Autonomous Racing},
              author={Herman, James and Francis, Jonathan and Ganju, Siddha and Chen, Bingqing and Koul, Anirudh and Gupta, Abhinav and Skabelkin, Alexey and Zhukov, Ivan and Kumskoy, Max and Nyberg, Eric},
              booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
              pages={9793--9802},
              year={2021}
            }
