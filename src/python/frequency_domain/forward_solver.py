from frequency_domain.complex_linear_system import convert_complex_linear_system_to_real,\
    extract_complex_solution
from frequency_domain.callbacks import dispatch_callbacks
import numpy as np
import pypardiso
import _fwi_ls
from frequency_domain.fem_common import assembly
from frequency_domain.function_builders import generate_discrete_field


def solve_2d_helmholtz(forward_case, callbacks=None, cache=None):
    """
    Solves the Helmholtz equation using the Finite Element Method in a 2d Mesh
    """
    if callbacks is None:
        callbacks = {}
    if cache is None:
        cache = {}
    dispatch_callbacks(callbacks, 'on_before_solve_2d_helmholtz')

    mesh = forward_case.mesh
    omega = forward_case.omega
    source = forward_case.source
    mu_function = forward_case.mu_function
    eta_function = forward_case.eta_function

    mu_for_integration = cache.get('mu_for_integration', generate_discrete_field(mesh, mu_function).reshape((mesh.nx * mesh.ny, 1)))
    eta_for_integration = cache.get('eta_for_integration', generate_discrete_field(mesh, eta_function).reshape((mesh.nx * mesh.ny, 1)))
    source_closest_pid = _generate_source_closest_pid(forward_case)
    source_affected_eids = mesh.elements_containing_pid(source_closest_pid)
    eids_to_source_value = {eid: (source_closest_pid, source.expression(omega)) for eid in source_affected_eids}

    dispatch_callbacks(callbacks, 'on_before_prepare_linear_system')
    Ke, f = _prepare_linear_system(mesh, mu_for_integration, eta_for_integration, eids_to_source_value, omega, cache)
    dispatch_callbacks(callbacks, 'on_after_prepare_linear_system', Ke=Ke, f=f)

    dispatch_callbacks(callbacks, 'on_before_convert_complex_linear_system_to_real')
    Ke, f = convert_complex_linear_system_to_real(Ke, f)
    dispatch_callbacks(callbacks, 'on_after_convert_complex_linear_system_to_real', Ke=Ke, f=f)

    dispatch_callbacks(callbacks, 'on_before_solve')
    P = pypardiso.spsolve(Ke.tocsc(), f)
    dispatch_callbacks(callbacks, 'on_after_solve', Ke=Ke, f=f, P=P)

    dispatch_callbacks(callbacks, 'on_before_extract_complex_solution')
    P = extract_complex_solution(P)
    dispatch_callbacks(callbacks, 'on_after_extract_complex_solution')

    P = P.reshape((mesh.nx + 1, mesh.ny + 1))
    dispatch_callbacks(callbacks, 'on_after_solve_2d_helmholtz', P=P, mesh=mesh)

    return P


def _generate_source_closest_pid(forward_case):
    mesh = forward_case.mesh
    source = forward_case.source
    return mesh.closest_point_id(source.original_position)


def _generate_K_for_omega(mu_for_integration, eta_for_integration, omega, mesh):
    return assembly(
        mesh,
        _fwi_ls.build_local_Ke,
        omega=omega,
        mu_field=mu_for_integration,
        eta_field=eta_for_integration,
    )


def _prepare_linear_system(mesh, mu_for_integration, eta_for_integration, eids_to_source_value, omega, cache=None):
    if cache is None:
        cache = {}
    K_for_omega = cache.get('K_for_omega', {})

    K = K_for_omega.get(omega, _generate_K_for_omega(mu_for_integration, eta_for_integration, omega, mesh))

    f = np.zeros(shape=(mesh.n_points, 1))
    for eid in eids_to_source_value:
        pid, value = eids_to_source_value[eid]
        f[pid, 0] = value

    return K, f
