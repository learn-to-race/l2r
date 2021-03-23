# ========================================================================= #
# Filename:                                                                 #
#    test_tracker.py                                                        #
#                                                                           #
# Description:                                                              #
#    unit test cases for envs.utils.Tracker                                 #
# ========================================================================= #

import time

from utils import Tracker

# Tracker params
N_INDICES = 300
OBS_DELAY = 0.0
MAX_TIMESTEPS = 30
DEBUG = False
T = 0.075

# Failure threshold (extreme precision not necessary)
THRESHOLD = 0.01

# Test Cases
COMPLETED_3_LAPS_CASES = [
    (0, [0, 100, 140, 160, 298, 299, 0, 299, 2, 299, 0], T * 10 + OBS_DELAY),
    (15, [15, 160, 165, 175, 299, 14, 15, 175, 14, 16, 290, 10, 15], T * 12 + OBS_DELAY),
    (150, [299, 0, 148, 158, 160, 0, 140, 155, 290,
           0, 149, 159], T * 10 + T / 10 + OBS_DELAY),
    (290, [290, 299, 0, 175, 299, 50, 150, 299, 160,
           284, 294], T * 9 + (T * 6 / 10) + OBS_DELAY)
]

DNF_CASES = [
    (0, [0, 155, 0, 155, 0, 1, 0], None),  # vehicle oscillates near start
    (125, list(range(MAX_TIMESTEPS + 5)), 'Completed max timesteps')
]


def test_completed_laps(tracker):
    """Unit test cases for 3 completed laps
    """
    print('\n' + '=' * 50 + '\nTEST GROUP - COMPLETED 3 LAPS')
    for n, case in enumerate(COMPLETED_3_LAPS_CASES):  # COMPLETED_3_LAPS_CASES):
        start_idx, updates, expected = case
        tk.reset(start_idx)

        for u in updates:
            time.sleep(T)
            tk.update(u)
            done, info, times, total = tk.is_finished()

        if info == 'Successfully completed 3 laps':
            delta = abs(expected - total)
            result = 'PASSED' if delta <= THRESHOLD else 'FAILED'
            print(f'[{n+1}] {result}\texp: {expected:.2f}\tact: {total:.2f}')


def test_dnf_cases(tracker):
    """Unit test for 'did not finish' cases
    """
    print('\n' + '=' * 50 + '\nTEST GROUP: DID NOT FINISH')
    for n, case in enumerate(DNF_CASES):
        start_idx, updates, expected = case
        tk.reset(start_idx)

        for u in updates:
            time.sleep(T)
            tk.update(u)
            done, info, times, total = tk.is_finished()

        if isinstance(info, str):
            result = 'PASSED' if info == expected else 'FAILED'
        else:
            result = 'PASSED' if expected is None else 'FAILED'

        print(f'CASE {n+1}: {result}\texp: {expected}\tact: {info}')


if __name__ == '__main__':
    tk = Tracker(
        n_indices=N_INDICES,
        obs_delay=OBS_DELAY,
        max_timesteps=MAX_TIMESTEPS,
        debug=DEBUG
    )
    test_completed_laps(tk)
    test_dnf_cases(tk)
