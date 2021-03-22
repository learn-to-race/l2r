# ========================================================================= #
# Filename:                                                                 #
#    tracker.py                                                             #
#                                                                           #
# Description:                                                              #
#    Tracks the vehicle and evaluates metrics on the path taken             #
# ========================================================================= #

import time

import numpy as np

from envs.utils import GeoLocation

# Assumed max progression, in number of indicies, in one RacingEnv.step()
MAX_PROGRESSION = 100

# Early termination for very poor performance
PROGRESS_THRESHOLD = 100
CHECK_PROGRESS_AT = 299

# constants
MPS_TO_KPH = 3.6
EPSILON = 1e-6
GRAVITY = 9.81


class ProgressTracker(object):
    """Progress tracker for the vehicle. This class serves to track the number
    of laps that the vehicle has completed and how long each one took. It also
    evaluates for termination conditions.

    :param int n_indices: number of indicies of the centerline of the track
    :param matplotlib.path inner_track: path of the inner track boundary
    :param matplotlib.path outer_track: path of the outer track boundary
    :param numpy.array centerline: path of the centerline
    :param list car_dims: dimensions of the vehicle in meters [len, width]
    :param float obs_delay: time delay between a RacingEnv action and
      observation
    :param int max_timesteps: maximum number of timesteps in an episode
    :param int not_moving_ct: maximum number of vehicle can be stationary before
      being considered stuck
    :param boolean debug: debugging print statement flag
    """

    def __init__(self, n_indices, inner_track, outer_track, centerline,
                 car_dims, obs_delay, max_timesteps, not_moving_ct,
                 debug=False):
        self.n_indices = n_indices
        self.inner_track = inner_track
        self.outer_track = outer_track
        self.centerline = centerline
        self.car_dims = car_dims
        self.obs_delay = obs_delay
        self.max_timesteps = max_timesteps
        self.not_moving_ct = not_moving_ct
        self.debug = debug
        self.reset(None)

    def reset(self, start_idx):
        """Reset the tracker for the next episode.

        :param int start_idx: index on the track's centerline which the vehicle
          is nearest to
        """
        self.start_idx = start_idx
        self.lap_start = None
        self.last_update_time = None
        self.lap_times = []
        self.halfway_flag = False
        self.ep_step_ct = 0
        self.transitions = []
        if start_idx is not None:
            self.last_idx = 0
            self.halfway_idx = self.n_indices // 2

    def update(self, idx, e, n, u, yaw, ac, bp):
        """Update the tracker based on current position. The tracker also keeps
        track of the yaw, brake pressure, centerline displacement, and
        calculates the offical metrics for the environment.

        :param int idx: index on the track's centerline which the vehicle is
          nearest to
        :param float e: east coordinate
        :param float n: north coordinate
        :param float u: up coordinate
        :param float yaw: vehicle heading, in radians
        :param numpy.array ac: directional acceleration, shape of (3,)
        :param numpy.array bp: brake pressure, per wheel, shape of (4,)
        """
        now = time.time()

        if self.lap_start is None:
            self.start_time = now
            self.last_update_time = now
            self.lap_start = now - self.obs_delay
            return

        if idx >= self.n_indices:
            raise Exception('Index out of bounds')

        n_out = self._count_wheels_oob(e, n, yaw)
        c_dist = self._dist_to_segment([e, n], idx)
        dt = now - self.last_update_time

        # shift idx such that the start index is at 0
        idx -= self.start_idx
        idx += self.n_indices if idx < 0 else 0

        # validate vehicle isn't just oscillating near the starting point
        if self.last_idx <= self.halfway_idx and idx >= self.halfway_idx:
            self.halfway_flag = True

        self._store(e, n, u, idx, yaw, c_dist, dt, ac, bp, n_out)

        # set halfway flag, if necessary
        _ = self.check_lap_completion(idx, now)
        self.ep_step_ct += 1
        self.last_update_time = now
        self.last_idx = idx

    def _store(self, e, n, u, idx, yaw, c_dist, dt, ac, bp, n_out):
        """Transitions are stored as a list of lists

        :param float e: east coordinate
        :param float n: north coordinate
        :param float u: up coordinate
        :param int idx: nearest centerline index to current position (shifted)
        :param float yaw: vehicle heading, in radians
        :param float c_dist: distance to centerline
        :param float dt: time delta, in seconds, from last update
        :param float ac: magnitude of the vehicles acceleration, adjusted for
          gravity
        :param numpy.array bp: brake pressure, per wheel, shape of (4,)
        :param int n_out: number of wheels out-of-bounds
        """
        b = np.average(bp)
        a = np.linalg.norm(ac) - GRAVITY
        self.transitions.append([e, n, u, idx, c_dist, yaw, dt, a, b, n_out])

    def check_lap_completion(self, shifted_idx, now):
        """Check if we completed a lap. To prevent vehicles from oscillating
        near the start line, the vehicle must first cross the halfway mark of
        the track. If the vehicle has cross the halfway mark, we then check if
        the vehicle has crossed the index that it started at. If the vehicle
        crosses multiple indicies in one time step when this happens, the lap's
        end time is a linear interpolation between the time of the prior
        update and the current time

        :param int shifted_idx: index on the track's centerline which the
          vehicle is nearest to shifted based on the starting index on the
          track
        :param float now: time of the update
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

    def is_complete(self):
        """Determine if the episode is complete due to finishing 3 laps,
        remaining in the same position for too long (stuck), exceeding the
        maximum number of timestepsor, or going out-of-bounds. If all 3 laps
        were successfully completed, the total time is also returned.

        :return: complete, info which includes metrics if successful
        :rtype: boolean, str, list of floats, float
        """
        info = self._is_terminal()

        if info['stuck'] or info['not_progressing'] or info['dnf'] or info['oob'] or info['success']:
            return True, self.append_metrics(info)

        return False, info

    def append_metrics(self, info):
        """Calculate metrics and append to info

        :param dict info: episode information
        :return: info with appended metrics
        :rtype: dict
        """
        transitions = np.asarray(self.transitions).T
        path = transitions[0:2]
        total_distance = ProgressTracker._path_length(path)
        total_time = self.last_update_time - self.start_time
        total_idxs = self.last_idx + self.n_indices * len(self.lap_times)
        avg_speed = total_distance / total_time * MPS_TO_KPH
        avg_cline_displacement = np.average(transitions[4])
        avg_curvature = ProgressTracker._path_curvature(path)
        track_curvature = ProgressTracker._path_curvature(self.centerline.T)
        brake_pressure = transitions[-2]
        accel = transitions[-3]
        sampling_freq = len(self.transitions) / total_time
        ms = ProgressTracker._log_dimensionless_jerk(accel, sampling_freq)
        proportion_unsafe = np.dot(transitions[-4], transitions[-1]) / total_time

        metrics = dict()
        metrics['pct_complete'] = round(100 * total_idxs / (3 * self.n_indices), 1)
        metrics['total_time'] = round(total_time, 2)
        metrics['total_distance'] = round(total_distance, 2)
        metrics['average_speed_kph'] = round(avg_speed, 2)
        metrics['average_displacement_error'] = round(avg_cline_displacement, 2)
        metrics['trajectory_efficiency'] = round(track_curvature / avg_curvature, 3)
        metrics['trajectory_admissibility'] = round(1 - (proportion_unsafe**0.5), 3)
        metrics['movement_smoothness'] = ms
        info['metrics'] = metrics
        return info

    @staticmethod
    def _path_length(path):
        """Calculate length of a path.

        :param numpy.array path: set of (x,y) pairs with shape (2, N)
        :return: path length in meters
        :rtype: float
        """
        x, y = path[0], path[1]
        _x = np.square(x[:-1] - x[1:])
        _y = np.square(y[:-1] - y[1:])
        return np.sum(np.sqrt(_x + _y))

    @staticmethod
    def _path_curvature(path):
        """Returns the root mean square of the curvature of the path where
        curvature is calculate parametrically using finite differences.

        :param numpy.array path: set of (x,y) pairs with shape (2, N)
        :return: RMS of the path's curvature
        :rtype: float
        """
        x, y = path[0], path[1]

        dx = 0.5 * (x[2:] - x[:-2])
        dy = 0.5 * (y[2:] - y[:-2])
        d2x = x[2:] - 2 * x[1:-1] + x[0:-2]
        d2y = y[2:] - 2 * y[1:-1] + y[0:-2]
        k = (dx * d2y - dy * d2x) / (np.square(dx) + np.square(dy) + EPSILON)**(3.0 / 2.0)
        k_rms = np.sqrt(np.mean(np.square(k)))

        return k_rms

    @staticmethod
    def _log_dimensionless_jerk(movement, freq, data_type='accl'):
        """Sivakumar Balasubramanian's implmentation of log dimensionless
        jerk.

        Source: https://github.com/siva82kb/SPARC/blob/master/scripts/smoothness.py

        :param numpy.array movement: movement speed profile of t data points
          with shape (t,)
        :param float freq: the sampling frequency of the data
        :param str data_type: type of movement data provided. must be in
          ['speed','accl','jerk']
        :return: dimensionless jerk, a smoothness metric
        :rtype: float
        """
        if data_type not in ('speed', 'accl', 'jerk'):
            raise ValueError('\n'.join(("The argument data_type must be either",
                                        "'speed', 'accl' or 'jerk'.")))

        movement = np.array(movement)
        movement_peak = max(abs(movement))
        dt = 1. / freq
        movement_dur = len(movement) * dt
        _p = {
            'speed': 3,
            'accl': 1,
            'jerk': -1
        }
        p = _p[data_type]
        scale = pow(movement_dur, p) / pow(movement_peak, 2)

        # estimate jerk
        if data_type == 'speed':
            jerk = np.diff(movement, 2) / pow(dt, 2)
        elif data_type == 'accl':
            jerk = np.diff(movement, 1) / pow(dt, 1)
        else:
            jerk = movement

        # estimate dj
        dim_jerk = - scale * sum(pow(jerk, 2)) * dt
        return np.log(abs(dim_jerk))

    def _dist_to_segment(self, p, idx):
        """Returns the shortest distance between point p a line segment
        between the two nearest points on the centerline of the track.

        :param array-like p: (x,y) reference point
        :param idx: centerline index to compare to
        :type idx: int
        """
        p = np.array(p) if isinstance(p, list) else p
        l1 = self.centerline[idx]
        l2 = self.centerline[(idx + 1) % len(self.centerline)]
        return abs(np.cross(l2 - l1, p - l1) / np.linalg.norm(l2 - l1))

    def _is_terminal(self):
        """Determine if the environment is in a terminal state
        """
        info = {
            'stuck': False,
            'oob': False,
            'success': False,
            'dnf': False,
            'not_progressing': False,
            'lap_times': self.lap_times
        }

        if len(self.lap_times) >= 3:
            info['success'] = True
            info['total_time'] = round(sum(self.lap_times), 2)

        if len(self.transitions) > self.not_moving_ct:
            if self.transitions[-1] == self.transitions[-self.not_moving_ct]:
                info['stuck'] = True

        total_idxs = self.last_idx + self.n_indices * len(self.lap_times)

        if self.ep_step_ct == CHECK_PROGRESS_AT and total_idxs < PROGRESS_THRESHOLD:
            info['not_progressing'] = True

        if self.ep_step_ct >= self.max_timesteps:
            info['dnf'] = True

        if self._car_out_of_bounds():
            info['oob'] = True

        return info

    def _car_out_of_bounds(self):
        """At most 1 wheel can be outside of the driveable area

        :return: True if vehicle out-of-bounds
        :rtype: bool
        """
        return len(self.transitions) > 0 and self.transitions[-1][-1] > 1

    def _count_wheels_oob(self, e, n, yaw):
        """Count number of wheels that are out of the drivable area

        :param float e: east coordinate
        :param float n: north coordinate
        :param float yaw: vehicle heading, in radians
        :return: Number of wheels out of drivable area
        :rtype: int
        """
        car_corners = GeoLocation.get_corners((e, n), yaw, self.car_dims)

        # At most 1 wheel can be out-of-track
        n_out_inside = np.count_nonzero(self.inner_track.contains_points(car_corners))
        if n_out_inside > 0:
            return n_out_inside

        return 4 - np.count_nonzero(self.outer_track.contains_points(car_corners))
