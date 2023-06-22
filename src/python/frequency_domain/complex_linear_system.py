from scipy.sparse.coo import coo_matrix
import numpy as np


def convert_complex_linear_system_to_real(A, b):
    """
    Given a linear system Ax=b where A and b have complex values, this function
    will produce an augmented problem A'x'=b' where A' and b' have all real values.
    The result of such problem (x') must be then converted back to complex using
    the extract_complex_solution from this module.

    :param coo_matrix A:
        Must be a sparse A matrix
    """
    import _fwi_ls
    build_extended_A = _fwi_ls.build_extended_A

    big_A_coo_i, big_A_coo_j, big_A_coo_data = build_extended_A(
        A.row,
        A.col,
        A.data,
        A.shape[0],
        A.shape[1],
    )

    big_A = coo_matrix(
        (big_A_coo_data, (big_A_coo_i, big_A_coo_j)), 
        shape=(2 * A.shape[0], 2 * A.shape[1]),
        dtype=np.float
    )

    # b vector is not sparse, so we just need to copy the values
    big_b = np.zeros(shape=(2*len(b), 1))
    big_b[0:len(b)] = b.real
    big_b[len(b):2*len(b)] = b.imag

    return big_A, big_b


def extract_complex_solution(big_x):
    x = np.zeros(shape=(len(big_x) // 2, 1), dtype=np.complex)
    x[:, 0].real = big_x[0:len(x)]
    x[:, 0].imag = big_x[len(x):2*len(x)]
    return x
