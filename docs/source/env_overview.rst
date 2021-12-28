
Environment Overview
====================


Introduction
-------------
`Learn-to-Race <https://github.com/hermgerm29/learn-to-race>`_ (L2R) is an `OpenAI gym <https://gym.openai.com/>`_ compliant, multimodal control environment where agents learn how to race. Unlike many simplistic learning environments, ours is built around Arrival’s high-fidelity racing simulator featuring full software-in-the-loop (SIL), and even hardware-in-the-loop (HIL), simulation capabilities. This simulator has played a key role in bringing autonomous racing technology to real life in the `Roborace series <https://roborace.com/>`_, the world’s first extreme competition of teams developing self-driving AI. Racing is an extreme challenge where agents must make real-time, high-risk decisions while finely controlling hardware operating near its physical limits. The L2R framework presents *objective-centric* tasks, rather than providing abstract rewards, and provides numerous quantitative metrics to measure the racing performance and trajectory quality of various agents.

For more information, please read our `paper <https://arxiv.org/abs/2103.11575>`_ on arXiv.

Baseline Models
---------------
We provide 'out-of-the-box' baseline models to demonstrate how to use the environment including a ``RandomActionAgent`` as a starting point. We also provide a model predictive control (MPC) agent, which is non-learning, as well as a `Soft Actor-Critic <https://arxiv.org/abs/1801.01290v1>`_ agent, which only relies on the vehicle's camera as input, and finally, an imitation learning implementation based on the MPC's demonstrations.

.. sidebar:: Untrained Policy

   Our `RandomActionAgent <getting_started.html#basic-example>`_
   completes a handful of episodes on the Las Vegas North Road track (with little success)

.. image:: ../assets/untrained.gif
    :width: 55 %


Action Space
------------
While you *can* change the gear, in practice we suggest forcing the agent to stay in drive since the others would not be advantageous in completing the tasks we present (we don't include it as a part of the action space). Note that negative acceleration values will brake the vehicle.

.. table::
   :widths: auto

   ============ ============ ==============
   Action       Type         Range
   ============ ============ ==============
   Steering     Continuous   [-1.0, 1.0]
   
   Acceleration Continuous   [-1.0, 1.0]
   ============ ============ ==============

The scaled action space is [-1.0, 1.0] for steering and [-16.0, 6.0] for acceleration. You can modify the boundaries of the action space, limiting acceleration, for example, if you would like by changing the parameters in ``action_if_kwargs`` in ``params.yaml``.

Observation Space
-----------------
We offer two high-level settings for the observation space: `vision-only <vision.html>`_ and `multimodal <multimodal.html>`_. In both, the agent receives RGB images from the vehicle's front-facing camera, examples below. The latter, however, also provides sensor data, including pose data from the vehicle's IMU sensor.

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

Additionally, these sensors are parameterized and can be customized further; for example, cameras have modifiable image size, field-of-view, and exposure. We provide a sample configuration below which has front and side facing cameras in both RGB mode and with ground truth segmentation. 

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

Racetracks
----------
We currently support three racetracks in our environment, both of which emulate real tracks. The first is the Vegas North Road track which is located at Las Vegas Motor Speedway in the United States. This track is used as the evaluation track, so users will only have access to this during evaluation. The second is the Thruxton Circuit, modeled off the track at the Thruxton Motorsport Centre in the United Kingdom. We will continue to add more racetracks in the future.

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
