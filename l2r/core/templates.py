import abc
from typing import List

import matplotlib.path


class AbstractInterface(abc.ABC):
    """Abstract simulator interface to receive data from the simulator."""

    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def start(self):
        """The start method is used to start communication with the simulator."""
        pass

    @abc.abstractmethod
    def get_data(self):
        """This method is used to return the most up-to-date information from
        the interface."""
        pass

    @abc.abstractmethod
    def reset(self):
        """Used to reset the interface, often to clear existing data."""
        pass


class AbstractReward(abc.ABC):
    """Abstract reward class. It is recommended that new reward policies follow
    this template so that they are compatible with the RacingEnv."""

    def __init__(self, *args, **kwargs):
        pass

    def set_track(
        self,
        inside_path: matplotlib.path,
        outside_path: matplotlib.path,
        centre_path: matplotlib.path,
        car_dims: List[float],
    ):
        """Store the track and vehicle information as class variables. This is
        useful for evaluating the reward based on the position of the vehicle.

        :param inside_path: ENU coordinates of the inside track boundary
        :type inside_path: matplotlib.path
        :param outside_path: ENU coordinates of the outside track boundary
        :type outside_path: matplotlib.path
        :param centre_path: ENU coordinates of the track's centerline
        :type centre_path: matplotlib.path
        :param car_dims: dimensions of the vehicle in meters: [length, width]
        :type car_dims: list
        """
        self.inner_track = inside_path.vertices
        self.outside_track = outside_path.vertices
        self.centre_path = centre_path.vertices
        self.n_inner = len(inside_path)
        self.n_outer = len(outside_path)
        self.n_centre = len(centre_path)
        self.car_dims = car_dims

    @abc.abstractmethod
    def get_reward(self, state, **kwargs):
        """Return the reward for the provided state.

        :param state: the current environment state
        :type state: varies
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """Reset the reward policy."""
        pass
