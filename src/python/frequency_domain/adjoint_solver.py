from frequency_domain.complex_linear_system import convert_complex_linear_system_to_real, \
    extract_complex_solution
import numpy as np
import pypardiso

def solve_adjoint_problem(
    mesh,
    omega,
    mu_for_integration,
    eta_for_integration,
    source_name,
    receivers,
    measured_data,
    simulated_data,
    precomputed_K=None
):
    '''
    Solves the adjoint problem for a given (omega, source) pair.
    Output will be the lambda_s_o, that is, lambda_(source)_(omega).

    If precomputed_K is given, will not compute K matrix. This is useful because this matrix is the same as the one
    computed for the forward problem, so it is possible to compute it once.
    '''
    from frequency_domain.forward_solver import _generate_K_for_omega

    if precomputed_K is None:
        K = _generate_K_for_omega(mu_for_integration, eta_for_integration, omega, mesh)
    else:
        K = precomputed_K

    f = np.zeros(shape=(mesh.n_points, 1))
    for receiver in receivers:
        key_pair = (source_name, receiver.name, omega)

        closest_pid = mesh.closest_point_id(receiver.original_position)
        Z_rs = simulated_data[key_pair]
        S_rs = measured_data[key_pair]

        f[closest_pid, 0] += (Z_rs - S_rs)

    K, f = convert_complex_linear_system_to_real(K, f)
    lambda_s_o = pypardiso.spsolve(K.tocsr(), f)
    lambda_s_o = extract_complex_solution(lambda_s_o)
    return lambda_s_o
