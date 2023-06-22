import numpy as np
from functools import lru_cache


class Mesh:
    """
    Quadrilateral FEM mesh representation
    """

    def __init__(self, nx, ny, points, connectivity_list, size_x, size_y):
        self.points = points
        self.connectivity_list = connectivity_list
        self.n_points = (nx + 1) * (ny + 1)
        self.nx = nx
        self.ny = ny
        self.size_x = size_x
        self.size_y = size_y
        self.points_in_elements = self._build_points_in_elements()
        self._closest_point_id_cache = {}
        self.n_elements = self.nx * self.ny
        self.n_nodes_per_element = 4

    def closest_point_id(self, desired_position):
        """
        :param np.array desired_position:
        """
        cache = self._closest_point_id_cache.get(desired_position)
        if cache is not None:
            return cache

        closest_distance = +np.inf
        closest_id = None
        for i, point in enumerate(self.points):
            d = np.linalg.norm(point - desired_position, ord=2)
            if d < closest_distance:
                closest_distance = d
                closest_id = i
        self._closest_point_id_cache[desired_position] = closest_id
        return closest_id

    @lru_cache(maxsize=None)
    def elements_containing_pid(self, pid):
        """
        :param int desired_position:
        """
        return [
            eid
            for eid, pids in enumerate(self.connectivity_list)
            if pid in pids
        ]

    def _build_points_in_elements(self):
        return np.array([
            [self.points[pid] for pid in e]
            for e in self.connectivity_list
        ])
