# learn-to-race

Learn-to-Race is an [OpenAI gym](https://gym.openai.com/) compliant, multimodal reinforcement learning environment where agents learn how to race. Unlike many simplistic learning environments, ours is built around Arrival’s CARLA-based software-in-the-loop racing simulator which was built specifically to accurately model race cars. This simulator is used in practice in [Roborace](https://roborace.com/), the world’s first extreme competition of teams developing self-driving AI.

<div style="text-align: center;">
  <h4>Las Vegas Track</h4>
  <span style="display:inline-block; text-align: center; width: 50%">
    <p>Our agent is able to consistently, but not always, complete laps in under
    2 minutes each.</p>
  </span>
  <figure style="display:inline-block; width:60%;">
    <img src='./asset/imgs/lvms_1.gif' alt='missing'/ width=92%>
    <figcaption style="padding: 10px 15px 15px;"><i>Episode 1</i></figcaption>
  </figure>
  <figure style="display:inline-block; width:60%;">
    <img src='./assets/imgs/lvms_100.gif' alt='missing' width=92%/>
    <figcaption style="padding: 10px 15px 15px;"><i>Episode 100</i></figcaption>
  </figure>
  <figure style="display:inline-block; width:60%;">
    <img src='./assets/imgs/lvms_1000.gif' alt='missing' width=92%/>
    <figcaption style="padding: 10px 15px 15px;"><i>Episode 1000</i></figcaption>
  </figure>
</div>

## Documentation

Please visit our [official docs](https://learn2race.com/) for a comprehensive guide on getting started with the environment. Happy racing!


## Requirements

**Python:** We use Learn-to-Race with Python 3.6 or 3.7.

**Graphics Hardware:** The racing simulator runs in a container, but it requires a GPU with Nvidia drivers installed. A Nvidia 970 GTX graphics card is minimally sufficient to simply run the simulator.

**Docker:** The racing simulator runs in a [Docker](https://www.docker.com/get-started) container.

**Container GPU Access:** The container needs access to the GPU, so [nvidia-container-runtime](https://github.com/NVIDIA/nvidia-container-runtime) is also required.

## Installation

While not technically required, this installation assumes a Debian-based operating system.

1. Request access to the Racing simulator. Once obtained, you can load the docker image:

```bash
$ docker load < arrival-sim-image.tar.gz
```

2. Download the source code from this repository and install the package requirements. We recommend using a virtual environment:

```bash
$ pip install virtualenv
$ virtualenv venv                           # create new virtual environment
$ source venv/bin/activate                  # activate the environment
(venv) $ pip install -r requirements.txt 
```
