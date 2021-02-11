# ========================================================================= #
# Filename:                                                                 #
#    tracker.py                                                             #
#                                                                           #
# Description:                                                              #
#    Tracker class for timing and termination conditions                    #
# ========================================================================= #

import time

import numpy as np

from envs.utils import GeoLocation

# Assumed max progression, in number of indicies, in one RacingEnv.step()
MAX_PROGRESSION = 100

# Early termination for very poor performance
PROGRESS_THRESHOLD = 100

class ProgressTracker(object):
	"""Progress tracker for the vehicle. This class serves to track the number
	of laps that the vehicle has completed and how long each one took. It also
	evaluates for termination conditions.

	:param n_indices: number of indicies of the centerline of the track
	:type n_indices: int
	:param inner_track: path of the inner track boundary
	:type inner_track: matplotlib.path
	:param outer_track: path of the outer track boundary
	:type outer_track: matplotlib.path
	:param car_dims: dimensions of the vehicle in meters [len, width]
	:type car_dims: list of floats
	:param obs_delay: time delay between a RacingEnv action and observation
	:type obs_delay: float
	:param max_timesteps: maximum number of timesteps in an episode
	:type max_timesteps: int
	:param not_moving_ct: maximum number of vehicle can be stationary before
	  being considered stuck
	:type not_moving_ct: int
	:param debug: debugging print statement flag
	:type debug: boolean
	"""
	def __init__(self, n_indices, inner_track, outer_track, car_dims,
		         obs_delay, max_timesteps, not_moving_ct, debug=False):
		self.n_indices = n_indices
		self.halfway_idx = n_indices//2
		self.inner_track = inner_track
		self.outer_track = outer_track
		self.car_dims = car_dims
		self.obs_delay = obs_delay
		self.max_timesteps = max_timesteps
		self.not_moving_ct = not_moving_ct
		self.debug = debug
		self.reset(None)

	def reset(self, start_idx):
		"""Reset the tracker for the next episode.

		:param start_idx: index on the track's centerline which the vehicle is
		  nearest to
		:type start_idx: int
		"""
		self.start_idx = start_idx
		self.last_idx = start_idx
		self.lap_start = None
		self.last_time = None
		self.lap_times = []
		self.halfway_flag = False
		self.ep_step_ct = 0

	def update(self, idx):
		"""Update the tracker based on the updated nearest track index.

		:param idx: index on the track's centerline which the vehicle is
		  nearest to
		:type idx: int
		"""
		now = time.time()

		if self.lap_start is None:
			self.last_update_time = now
			self.lap_start = now - self.obs_delay
			return

		if idx >= self.n_indices:
			raise Exception('Index out of bounds')

		# shift idx such that the start index is at 0
		idx -= self.start_idx
		idx += self.n_indices if idx < 0 else 0

		# validate vehicle isn't just oscillating near the starting point
		if self.last_idx <= self.halfway_idx and idx >= self.halfway_idx:
			self.halfway_flag = True

		_ = self.check_lap_completion(idx, now)
		self.ep_step_ct += 1
		self.last_update_time = now
		self.last_idx = idx

	def check_lap_completion(self, shifted_idx, now):
		"""Check if we completed a lap. To prevent vehicles from oscillating
		near the start line, the vehicle must first cross the halfway mark of
		the track. If the vehicle has cross the halfway mark, we then check if
		the vehicle has crossed the index that it started at. If the vehicle
		crosses multiple indicies in one time step when this happens, the lap's
		end time is a linear interpolation between the time of the prior 
		update and the current time

		:param shifted_idx: index on the track's centerline which the vehicle
		  is nearest to shifted based on the starting index on the track
		:type shifted_idx: int
		:param now: time of the update
		:type now: float
		"""
		if not self.halfway_flag:
			return False

		if shifted_idx < MAX_PROGRESSION:
			ct = shifted_idx + (self.n_indices - self.last_idx)
			lap_end = now - (now - self.last_update_time) / ct * shifted_idx
			lap_time = round(lap_end - self.lap_start, 2)
			self.lap_times.append(lap_time)
			self.lap_start = lap_end
			self.halfway_flag = False
			return True

		return False

	def is_complete(self, location_history, yaw):
		"""Determine if the episode is complete due to finishing 3 laps,
		remaining in the same position for too long (stuck), exceeding the
		maximum number of timestepsor, or going out-of-bounds. If all 3 laps
		were successfully completed, the total time is also returned.

		:param location_history: recent coordinates of the vehicle in the 
		  format: (East, North, Up)
		:type location_history: list
		:param yaw: heading of the vehicle, in radians
		:type yaw: float
		:return: complete, info message, lap times, total race time
		:rtype: boolean, str, list of floats, float
		"""
		info = {'stuck': False, 'oob': False, 'success': False, 'dnf': False}
		info['lap_times'] = self.lap_times

		if len(self.lap_times) >= 3:
			info['success'] = True
			info['total_time'] = round(sum(self.lap_times), 2)
			info['pct_complete'] = 100.00
			return True, info

		total_idxs = self.last_idx + self.n_indices*len(self.lap_times)
		info['pct_complete'] = round(100*total_idxs/(3*self.n_indices), 1)

		if len(location_history) > self.not_moving_ct:
			if location_history[-1] == location_history[-self.not_moving_ct]:
				info['stuck'] = True
				return True, info

		if self.ep_step_ct == 299 and total_idxs < PROGRESS_THRESHOLD:
			info['not_progressing'] = True
			return True, info

		if self.ep_step_ct >= self.max_timesteps:
			info['dnf'] = True
			return True, info 

		if self._car_out_of_bounds(location_history, yaw):
			info['oob'] = True
			return True, info

		return False, info

	def _car_out_of_bounds(self, location_history, yaw):
		"""Determine if the car is outside the track boundaries.

		:param location_history: recent coordinates of the vehicle in the 
		  format: (East, North)
		:type location_history: collections.deque
		:param yaw: heading of the vehicle, in radians
		:type yaw: float
		:return: True if the car is oob, False otherwise
		:rtype: boolean
		"""
		center = location_history[-1]
		car_corners = GeoLocation.get_corners(center, yaw, self.car_dims)

		# At most 1 wheel can be out-of-track
		if np.count_nonzero(self.inner_track.contains_points(car_corners)) > 1:
			return True

		if np.count_nonzero(self.outer_track.contains_points(car_corners)) < 3:
			return True

		return False

