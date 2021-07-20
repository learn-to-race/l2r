
Getting Started
===============

.. warning::
   The L2R framework is coupled with the Arrival racing simulator which has not yet been released. To gain access to the racing simulator, you must sign a licensing agreement with Arrival, the creator and owner of the simulation software. The simulator is expected to be available in May, 2021. Please complete this `form <https://forms.gle/PXNM6hHkEgiAzhoa8>`_ to be notified of its release.

   
Racing Simulator
----------------

To use our environment, you mst first complete a licensing agreement with Arrival. Our environment interfaces with the Arrival racing simulator via a `SimulatorController <l2r.core.html#l2r.core.controller.SimulatorController>`_ object which can launch, restart, and control the Arrival simulator. We recommend running the simulator as a Python subprocess by specifying the path of the simulator in your configuration file in the ``env_kwargs.controller_kwargs.sim_path`` field. Alternatively, you can specify that the controller start the simulator as a Docker container by setting ``env_kwargs.controller_kwargs.start_container`` to True. If you choose the latter, you can load the docker image:

.. code-block:: shell

	$ docker load < arrival-sim-image.tar.gz

Requirements
------------

**Python**: We use Learn-to-Race with Python 3.6 and 3.7

**Graphics Hardware**: The racing simulator runs in a container with a `Ubuntu 18.04 cudagl <https://gitlab.com/nvidia/container-images/cudagl/-/tree/ubuntu18.04>`_ base image. Running the container requires a GPU with Nvidia drivers installed. A Nvidia 970 GTX graphics card is minimally sufficient.

**Docker**: If you would like to run the racing simulator in a `Docker <https://www.docker.com/>`_ image, you will need Docker installed.

**Container GPU Access**: If using Docker, the container needs to access the GPU, so `nvidia-container-runtime <https://github.com/NVIDIA/nvidia-container-runtime>`_ is also required. For Ubuntu distributions:

.. code-block:: shell

	$ curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
	  sudo apt-key add -
	$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
	$ curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
	  sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
	$ sudo apt-get update

And finally,

.. code-block:: shell

	$ sudo apt-get install nvidia-container-runtime


.. note:: This documentation assumes a Debian-based operating system (Ubuntu), but this OS is not required.	

Installation 
------------

Package Code & Dependencies
***************************
Simply download the source code from the `Github repository <https://github.com/hermgerm29/learn-to-race>`_. We recommend using a `virtual Python environment <https://docs.python.org/3.6/library/venv.html>`_. Once activated, install the package requirements located in the top-level directory of the repo.

.. code-block:: shell

	$ pip install virtualenv
	$ virtualenv venv                         # create virtual environment
	$ source venv/bin/activated               # activate the environment
	(venv) $ pip install -r requirements.txt

Environment Caveats
*******************

.. important:: Our autonomous racing environment is `OpenAI gym <https://gym.openai.com/>`_ compliant with a few important notes:

	- The simulator must be running to instantiate a `RacingEnv <l2r.envs.html#module-roboracer.envs.env>`_ which is, more or less, a set of interfaces which allow the agent to communicate with the simulator. When a RacingEnv object is constructed, it attempts to establish a connection with the simulator; if it is not running, you will see an error similiar to the one below:

	.. code-block:: shell

	   $ ConnectionRefusedError: [Errno 111] Connection refused

	- Unlike many simplistic reinforcement learning baseline environments, the simulator that our environment uses is not time-invariant. If you, for example, run a gradient update step in the middle of an episode, the agent will continue to move, and perhaps slow down, which may result in unwanted additions to your replay buffer (assuming you are using one). This is clearly undesirable, so a *de-facto* restriction of the environment is that you treat each episode as if it were true inference.


Basic Example
*************

Let's first get familiar with our environment by creating an agent the simply chooses random actions from the action space. We provide such an agent called a ``RandomActionAgent`` with the source code below:

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



**Step 1: Start the simulator (optional)** 

The `SimulatorController <l2r.core.html#l2r.core.controller.SimulatorController>`_ will run the simulator as Python subprocess if ``env_kwargs.controller_kwargs.sim_path`` is specified or start the container containing the simulator if ``env_kwargs.controller_kwargs.start_container`` is set to True. If you would like to run the container manually, you can do so as follows:

.. code-block:: shell

   $ docker run -it --rm \              # interactive mode, clean-up on exit 
       --user=ubuntu \                  # login to container as user "ubuntu"
       --gpus all \                     # allow container to access all host GPUs
       --name racing-simulator \        # name of the container
       --net=host \                     # allow container to share host's namespace
       --entrypoint="./ArrivalSim.sh"   # run the simulator start up script on entry
       arrival-sim                      # name of the docker image


Please note that running the container, by default, will not render a display. Alternatively, you can run the launch script:

.. code-block:: shell

	$ ./ArrivalSim-linux-0.7.0.182276/LinuxNoEditor/ArrivalSim.sh -openGL


**Step 2: Run the random agent baseline model**

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
