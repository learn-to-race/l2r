# L2R ROS Stack Manual

## Overview
This directory is the workspace for all ROS2 nodes in the L2R framework, including the middleware to dock other platforms to L2R and a group of mock nodes to simulate an external ROS2 stack.

## Prerequisite
1. L2R dependencies
2. ROS2 Foxy: https://docs.ros.org/en/foxy/Installation.html
3. Arrival Simulator (for mock ROS2 nodes)

## Packages
1. l2r_ros: The middleware package responsible for connecting to various platforms
2. mock_control: Mock ROS2 nodes backed by Arrival Simulator

## Usage
### Build
```
colcon build
. install/setup.bash
```
Replace `.bash` to the shell in use (e.g. `install/setup.zsh`).

### Launch Mock Nodes
Launch the Arrival Simulator before running the following commands. Each command won't stop until explicitly killed. Run in different shells or append `&` to run in background. Make sure `l2r` is installed and included in the `$PYTHONPATH`.
```
ros2 run mock_control img_pub
ros2 run mock_control pose_pub
ros2 run mock_control action_sub
```

### Lauch Middleware
```
ros2 run l2r_ros mock_ros2_middleware
```

### Run Example Agent
After launching the mock nodes and the middleware. Run the following command to run the agent through the middleware and mock nodes.
```
python3 race.py
```
`race.py` is located in `examples/middleware/` from the repository root.

