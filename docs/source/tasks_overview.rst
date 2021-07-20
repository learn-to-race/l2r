
Task Overview
=============

Overview
********

The task is to learn how to race using either vision-only or multiple input modalities. While Arrival's racing simulator and the L2R framework is suitable for a wide variety of approaches including:

- classical control
- pre-planning trajectories
- reinforcement learning (RL)
- imiatation learning (IL)

We are most interested in agents which learn how to perceive their surroundings and effectively control the race car. We are also familiar with the success of RL approaches and fully expect, and encourage, the community to create agents which are significantly better than our human benchmarks. However, learning-based approaches often have poor sample efficiency and fail at generalizing to new scenarios. Humans, on the other hand, are good at quickly adapting to new situations, such as racing on a new, unseen track after only a short warmup period. The L2R task will challenges agents' ability race in such a way.

Task Definition
---------------

Rather than giving agents an unbounded amount of time to perfectly overfit to a racetrack, the L2R task allows only a limited look at a new track, much like a Formula 1 driver gets only a brief practice session prior to the actual race. More concretely, agent assessment involves two stages:

(1) **Pre-evaluation:** agents will have access to an unseen, evaluation racetrack for 60 minutes with unfrozen weights. The agent is free to use this time for any purpose they deem necessary, but we generally expect agents to transfer their prior racing knowledge to the new environment after a brief exploration period.

(2) **Evaluation:** to qualify for evaluation, the agent must demonstrate that it can complete at least 1 lap during the pre-evaluation stage, subject to a modest maximum time limit. Successful agents will be evaluated on the testing racetrack and their metrics, defined below, will be recorded and updated on a leaderboard. To prevent competitors from unfairly learning on the test track, submissions will only be able to see their agent's results and will not have access to any model updates during the pre-evaluation stage.

.. note:: We currently only support single vehicle racing, but hope to introduce a multi-agent environment in the future.

Input Modalities
----------------

We present two distinct sets of information available to agents. All information available to agents use virtual sensors that emulate their real counterparts, so the agent does not have access to any privileged information. A separate leadboard will be used for agent's using the more restricted *vision-only* input mode.

`Vision-Only <vision.html>`_
  *The agent only has access to raw pixel values from the vehicle's cameras*

`Multimodal <multimodal.html>`_
  *In addition to the cameras, we provide the agent with sensor data, primarily from the vehicle's IMU sensor*

Metrics
*******

Learn-to-Race defines numerous metrics for the assessment of an agent's performance listed below. These are
provided to agents upon episode termination in the *info* of the last environment step.

.. list-table:: 
   :widths: 25 75
   :header-rows: 1

   * - Metric
     - Definition
   * - *Episode Completion Percentage*
     - Percentage of the 3-lap episode completed
   * - *Episode Duration*
     - Duration of the episode, in seconds
   * - *Average Adjusted Track Speed*
     - Average speed, adjusted for environmental conditions, in km/h
   * - *Average Displacement Error*
     - Euclidean displacement from the track centerline, in meters
   * - *Trajectory Admissibility*
     - Measurement of the safety of the trajectory
   * - *Trajectory Efficiency*
     - Ratio of track curvature to trajectory curvature
   * - *Movement Smoothness*
     - Log dimensionless jerk based on accelerometer data


Basic Metrics
-------------

A successful episode is defined as completing 3 laps, from a standing start, without 2 wheels going out-of-bounds at any point in time (1 is permissable, but considered *unsafe*). L2R provides basic metrics like the percentage of the 3 laps completed, the lap times for each successfully completed lap, the total time of the episode, and the average speed of the vehicle.


Trajectory Quality
------------------

To understand the quality of an agent's trajectories, L2R also includes metrics like *average displacement error* which is simply the agent's average distance from the centerline which is particularly useful measuring a controllers ability to stay near its target. We also include *trajectory admissibility* or :math:`\alpha`, shown below, where :math:`t_{u}` is the cumulative time spent unsafely with 1 wheel out-of-bounds and :math:`t_{e}` is the total length of the episode.

.. math::
   \alpha = 1 - \sqrt{\frac{t_{\text{u}}}{t_{e}}}

A perfectly admissable trajectory is 1.0 with no time spent outside of the drivable area. Furthermore, we provide a *trajectory efficiency* ratio which is the ratio of curvature of the racetrack's centerline to the curvature of the trajectory, measured parametrically using the root mean square. Strong racing agents should minimize their curvature to maintain high speeds, for example, by cutting corners, and have an efficiency of at least 1.0.

.. warning::
   If the agent doesn't complete the entire episode, the trajectory efficiency metric will likely be distorted since it would be comparing a partial trajectory, which may exclude high curvature areas of the track, to the entire racetrack.

Good racing agents should also be able to anticipate the need for changes in velocity and have the ability to smoothly control such changes. L2R also includes a *movement smoothness* measure, the negated log dimensionless jerk, :math:`\eta_{ldj}`, which quantifies the smoothness of the agent's acceleration profile.

.. math::
   \eta_{ldj} = \ln \left( \frac{(t_{2}-t_{1})^{3}}{v_{peak}^{2}} \int_{t_{1}}^{t_{2}}\left\lvert\frac{d^{2}v}{dt^{2}}\right\rvert^{2} dt \right)

Agents that tend to jerk the vehicle or brake violently, both dangerous maneuvers, will have a worse movement smoothness measure.


Legal Modifications
*******************

For the purpose of benchmarking, we require that you adhere to some degree of requirements. There are no restrictions in the modification or usage of:

- exploration or learning method
- incentive method (reward function)
- network architecture
- pre-trained perception models
- the delay between action and observation
- changing the action space

Training Only
*************

Certain camera settings must be considered training-only if they are realistically accessible to a physical racecar. The following camera settings are not available during evaluation:

- Segmentation cameras
- Cameras not touching the vehicle (for example, birdseye views)

Illegal Modifications
*********************

- Not using the default vehicle in the simulator (DevBot 2.0)
- Changing any physical parameters of the simulator such as the friction settings; we are *not* concerned about sim2real transfer
- Modifying the tracker method that would influence termination conditions or lap timing
