import dataclasses
import json
import os
import pathlib
from typing import List

import matplotlib.path as mplPath
import numpy as np
from scipy.spatial import KDTree

from l2r.constants import N_SEGMENTS
from l2r.utils.space import smooth_yaw
from .utils import level_2_trackmap


@dataclasses.dataclass
class Racetrack:
    """A representation of a racetrack"""

    ref_point: np.array
    inside_arr: np.array
    outside_arr: np.array
    centerline_arr: np.array
    manual_segments: bool
    segment_poses: List[List[float]]
    random_poses: List[List[float]]

    def __post_init__(self) -> None:
        # Path representation of the track boundaries
        self.inside_path = mplPath.Path(self.inside_arr)
        self.outside_path = mplPath.Path(self.outside_arr)
        self.centre_path = mplPath.Path(self.centerline_arr)

        # Additional representations
        self.n_indices = len(self.centerline_arr)
        self.kdtree = KDTree(self.centerline_arr)

        local_segment_idxs_manual = self.poses_to_local_segment_idxs(self.segment_poses)
        local_segment_idxs_linspace = np.round(
            np.linspace(0, self.n_indices - 2, N_SEGMENTS + 1)
        ).astype(int)

        self.local_segment_idxs = (
            local_segment_idxs_linspace
            if not self.manual_segments
            else local_segment_idxs_manual
        )

        # Compute yaw values
        race_x = self.centerline_arr[:, 0]
        race_y = self.centerline_arr[:, 1]
        X_diff = np.concatenate([race_x[1:] - race_x[:-1], [race_x[0] - race_x[-1]]])
        Y_diff = np.concatenate([race_y[1:] - race_y[:-1], [race_y[0] - race_y[-1]]])
        race_yaw = np.arctan(X_diff / Y_diff)  # (L-1, n)
        race_yaw[Y_diff < 0] += np.pi

        self.race_yaw = smooth_yaw(race_yaw)
        self.max_yaw = np.max(self.race_yaw)
        self.min_yaw = np.min(self.race_yaw)

        # Local segment indicies
        # If not manual, use linspace to automatically determine segment indices,
        # otherwise use manual points
        """self.local_segment_idxs = (
            np.round(np.linspace(0, self.n_indices - 2, N_SEGMENTS + 1)).astype(int)
            if not self.manual_segments
            else [self.nearest_idx(point=[x, y]) for (x, y, _, _) in self.segment_poses]
        )"""

        self.segment_tree = KDTree(
            np.expand_dims(np.array(self.local_segment_idxs), axis=1)
        )

    def nearest_idx(self, point: np.array) -> int:
        """Get the nearest index on the track to a point"""
        return self.kdtree.query(point)[1]

    def poses_to_local_segment_idxs(self, poses):

        segment_idxs = []
        for (x, y, z, yaw) in poses:
            # enu_x, enu_y, enu_z = self.geo_location.convert_to_ENU((x, y, z))
            # idx = self.kdtree.query(np.asarray([enu_x, enu_y]))[1]
            idx = self.kdtree.query(np.asarray([x, y]))[1]
            segment_idxs.append(idx)

        return segment_idxs


def load_track(level: str, manual_segments: bool = False) -> Racetrack:
    """Load a map into a Racetrack object"""
    map_file, random_poses, segment_poses = level_2_trackmap(level=level)

    # Read map file
    with open(os.path.join(pathlib.Path().absolute(), map_file), "r") as f:
        original_map = json.load(f)

    _out = np.asarray(original_map["Outside"])
    _in = np.asarray(original_map["Inside"])

    # Read track boundaries
    outside_arr = _out if _out.shape[-1] == 2 else _out[:, :-1]
    inside_arr = _in if _in.shape[-1] == 2 else _in[:, :-1]
    centerline_arr = np.asarray(original_map["Centre"])

    return Racetrack(
        ref_point=original_map["ReferencePoint"],
        inside_arr=inside_arr,
        outside_arr=outside_arr,
        centerline_arr=centerline_arr,
        manual_segments=manual_segments,
        segment_poses=segment_poses,
        random_poses=random_poses,
    )
