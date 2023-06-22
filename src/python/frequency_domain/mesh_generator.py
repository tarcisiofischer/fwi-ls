from frequency_domain.mesh import Mesh
import numpy as np
import _fwi_ls


def build_2d_quad_mesh(size_x, size_y, nx, ny):
    """
    Build a 2d quad mesh with the desired sizes
    """
    build_connectivity_list = _fwi_ls.build_connectivity_list

    x = np.linspace(0, size_x, nx + 1)
    y = np.linspace(0, size_y, ny + 1)
    xv, yv = np.meshgrid(x, y)
    points = np.vstack([xv.ravel(), yv.ravel()]).T
    connectivity_list = build_connectivity_list(nx, ny)

    return Mesh(nx, ny, points, connectivity_list, size_x, size_y)
