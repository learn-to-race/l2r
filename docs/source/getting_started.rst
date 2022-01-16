
Setup & Installation
====================
   
Racing Simulator
----------------

To use the Learn-to-Race environment, you must first `request access <https://learn-to-race.org/sim>`_, by filling out and returning a signed academic-use license. 

Our environment interfaces with the Arrival Autonomous Racing Simulator via a `SimulatorController <l2r.core.html#l2r.core.controller.SimulatorController>`_ object which can launch, restart, and control the simulator. 

Simulator Requirements
**********************

**Operating System**: The racing simulator has been tested on Ubuntu Linux 18.04 OS.

**Graphics Hardware**: The simulator has been tested to run smoothly on NVIDIA GeForce GTX 970 graphics cards. The simulator has been additionally tested on the following cards:

* NVIDIA GeForce GTX 1070
* NVIDIA GeForce GTX 1080, 1080 Ti
* NVIDIA GeForce GTX 2080, 2080 Ti
* NVIDIA GeForce GTX 3080, 3080 Ti
* NVIDIA GeForce GTX 3090 

**Software Dependencies**:

* Please install the appropriate CUDA and NVIDIA drivers.
* Please additionally install the following software dependencies:

.. code-block:: shell

	$ sudo apt-get install libhdf5-dev libglib2.0-dev libglib2.0-dev ffmpeg libsm6 libxext6 apt-transport-https


Running the Simulator
*********************

After the signed academic-use license is returned and approved, you will be given the opportunity to download the Arrival Autonomous Racing Simulator (*.tar.gz file). The simulator is currently being distributed as part of the Learn-to-Race Autonomous Racing Virtual Challenge, with a base file footprint of 2.8 GB. 

Open a temrinal screen and untar the simulator source, to a location of your choice:

.. code-block:: shell

	$ cd /path/to/simulator/download/location
	$ tar -xvzf /path/to/simulator/ArrivalSim-linux-{VERSION}.tar.gz
	$ chmod -R 777 /path/to/simulator/ArrivalSim-linux-{VERSION}/

We recommend running the simulator as a dedicated Python process, by executing: 

.. code-block:: shell

	$ bash /path/to/simulator/ArrivalSim-linux-0.7.0.182276/LinuxNoEditor/ArrivalSim.sh -openGL

Note: Users may receive a pop-up window, warning about OpenGL being deprecated in favour of Vulkan. It is safe to click 'Ok', to continue initialisation and use of the simulator.


Learn-to-Race Framework
-----------------------

Installation 
************

Simply download the source code from the `Github repository <https://github.com/hermgerm29/learn-to-race>`_. 

We recommend using a python virtual environment, such as `Anaconda <https://www.anaconda.com/products/individual>`_. Please download the appropriate version for your system. We have tested Learn-to-Race with Python versions 3.6 and 3.7.

Create a new conda environment, activate it, then install the Learn-to-Race python package dependencies:

.. code-block:: shell

	$ conda env create -n l2r -m python=3.6		# create virtual environment
	$ conda activate l2r               		# activate the environment
	(l2r) $ cd /path/to/repository/
	(l2r) $ pip install -r requirements.txt

Runtime Steps
-------------

1. Start the simulator (e.g., in a separate terminal window), if it has not already been started:

.. code-block:: shell

	$ bash /path/to/simulator/ArrivalSim-linux-0.7.0.182276/LinuxNoEditor/ArrivalSim.sh -openGL

2. Run/train/evaluate an agent, using the Learn-to-Race framework (e.g., within a `tmux` window):

.. code-block:: shell

	$ cd /path/to/repository
	$ cd l2r
	$ tmux new -s development
	$ conda activate l2r
	(l2r) $ chmod +x run.bash
	(l2r) $ ./run.bash -b random


Basic Agent Example (Random Agent)
**********************************

Here is an example of an agent that chooses random actions from the action space, provided by the environment. 

We provide such an agent called a ``RandomAgent`` with the source code below:

.. code-block:: python

	
   from core.templates import AbstractAgent
   from envs.env import RacingEnv
    
   class RandomActionAgent(AbstractAgent):
      """Reinforcement learning agent that simply chooses random actions.
    
      :param dict training_kwargs: training keyword arguments
      """
      def __init__(self, training_kwargs):
         self.num_episodes = training_kwargs['num_episodes']
    
      def race(self):
         """Demonstrative training method.
         """
         for e in range(self.num_episodes):
            print(f'Episode {e+1} of {self.num_episodes}')
            ep_reward = 0
            state, done = self.env.reset(), False

            while not done:
               action = self.select_action()
               state, reward, done, info = self.env.step(action)
               ep_reward += reward
                
         print(f'Completed episode with total reward: {ep_reward}')
         print(f'Episode info: {info}\n')

      def select_action(self):
         """Select a random action from the action space.

         :return: random action to take
         :rtype: numpy array
         """
         return self.env.action_space.sample()

      def create_env(self, env_kwargs, sim_kwargs):
         """Instantiate a racing environment

         :param dict env_kwargs: environment keyword arguments
         :param dict sim_kwargs: simulator setting keyword arguments
         """
         self.env = RacingEnv(
            max_timesteps=env_kwargs['max_timesteps'],
            obs_delay=env_kwargs['obs_delay'],
            not_moving_timeout=env_kwargs['not_moving_timeout'],
            controller_kwargs=env_kwargs['controller_kwargs'],
            reward_pol=env_kwargs['reward_pol'],
            reward_kwargs=env_kwargs['reward_kwargs'],
            action_if_kwargs=env_kwargs['action_if_kwargs'],
            pose_if_kwargs=env_kwargs['pose_if_kwargs'],
            cameras=env_kwargs['cameras']
         )

         self.env.make(
            level=sim_kwargs['racetrack'],
            multimodal=env_kwargs['multimodal'],
            driver_params=sim_kwargs['driver_params']
         )

         print(f'Environment created with observation space: ')
         for k, v in self.env.observation_space.spaces.items():
            print(f'\t{k}: {v}')

**Run the random agent baseline model**

For convenience, we have provided a number of files to assist with training a model. To run the random agent baseline, you can simply run the script in the top level of the repository with the baseline flag ``-b`` with argument ``random``:

.. code-block:: shell

   $ chmod +x run.bash  # make our script executable
   $ ./run.bash -b random

The agent will begin randomly taking actions in the environment and will print the reward for each episode upon completion.

**Convenience Scripts**

``run.bash`` simply passes parameters files to Python scripts. The baseline configuration files contains a variety of parameters including:

	1.  training parameters
	2.  environment parameters (for the RL environment)
	3.  simulator parameters (for the simulator)

We recommend using this structure, or following a similar practice, to train models with the environment and keep track of different training runs.
