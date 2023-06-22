from scipy.sparse.coo import coo_matrix
import numpy as np


def assembly(mesh, f, **kwargs):
    """
    Generic assembly function for the Finite Element Method. Will assembly the sparse matrix K
    for a given mesh. Element values will be computed using function f.
    """
    sparse_matrix_size = mesh.n_elements * mesh.n_nodes_per_element * mesh.n_nodes_per_element
    Ke_coo_i = np.empty(shape=(sparse_matrix_size,), dtype=np.int)
    Ke_coo_j = np.empty(shape=(sparse_matrix_size,), dtype=np.int)
    Ke_coo_data = np.empty(shape=(sparse_matrix_size,), dtype=np.complex)

    connectivity_list = mesh.connectivity_list
    points_in_elements = mesh.points_in_elements
    _Ke_coo_idx = 0
    for eid, (element_connectivity, element_points) in enumerate(zip(connectivity_list, points_in_elements)):
        v = f(eid, element_points, **kwargs)
        for k, p1 in enumerate(element_connectivity):
            for l, p2 in enumerate(element_connectivity):
                Ke_coo_i[_Ke_coo_idx] = p1
                Ke_coo_j[_Ke_coo_idx] = p2
                Ke_coo_data[_Ke_coo_idx] = v[k, l]
                _Ke_coo_idx += 1

    return coo_matrix(
        (Ke_coo_data, (Ke_coo_i, Ke_coo_j)), 
        shape=(mesh.n_points, mesh.n_points),
        dtype=np.complex,
    )
