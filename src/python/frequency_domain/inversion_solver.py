import joblib

from frequency_domain.forward_solver import solve_2d_helmholtz
import numpy as np
import pypardiso
from frequency_domain.complex_linear_system import convert_complex_linear_system_to_real,\
    extract_complex_solution
from frequency_domain.forward_case import ForwardCase
from frequency_domain.adjoint_solver import solve_adjoint_problem
from frequency_domain.fem_common import assembly
from frequency_domain.function_builders import SmoothHeavySideBasedFunctionBuilder
from frequency_domain.callbacks import dispatch_callbacks
import _fwi_ls


def solve_fwi(inversion_case, callbacks=None):
    """
    Solves the Full Waveform Inversion using Level Set functions to approximate the field property
    distribution (Inclusions).
    """
    if callbacks is None:
        callbacks = {}
    dispatch_callbacks(callbacks, 'on_before_solve_fwi')

    # Configure pypardiso's parallelization
    import pypardiso
    pypardiso.scipy_aliases.pypardiso_solver.set_num_threads(inversion_case.parallel_specs.n_parallel_linear_solver)

    mesh = inversion_case.mesh
    converged = inversion_case.converged
    omega_list = inversion_case.experimental_data.omega_list
    sources = inversion_case.experimental_data.sources
    receivers = inversion_case.experimental_data.receivers
    measured_data = inversion_case.experimental_data.measured_data
    eta_function = inversion_case.eta_function
    initial_guess = inversion_case.initial_guess
    
    phi = initial_guess.phi
    F0 = initial_guess.F0
    F1 = initial_guess.F1

    mu_for_integration, mu_function = _build_mu_from_phi(mesh, phi, F0, F1)

    def compute_function_inside_elements(mesh, interpolation_function):
        mean_points = np.mean(mesh.points_in_elements, axis=1)
        xs = mean_points[0:mesh.ny,0]
        ys = mean_points[::mesh.nx,1]
        return interpolation_function(xs, ys).reshape((mesh.nx * mesh.ny, 1))

    eta_for_integration = compute_function_inside_elements(mesh, eta_function)
    eta_in_nodes = eta_function(mesh.points[0:mesh.ny + 1, 0], mesh.points[::mesh.nx + 1, 1]).reshape(
        ((mesh.nx + 1) * (mesh.ny + 1), 1)
    )

    source_closest_pid_per_source = {}
    for source in sources:
        source_closest_pid_per_source[source.name] = mesh.closest_point_id(source.original_position)

    i = 0
    K_for_omega = _precompute_K_for_each_omega(mesh, omega_list, mu_for_integration, eta_for_integration)

    dispatch_callbacks(callbacks, 'on_before_first_evaluation')
    J, simulated_data, pressures = _objective_f(
        inversion_case,
        mu_function,
        cache={
            'K_for_omega': K_for_omega,
            'mu_for_integration': mu_for_integration,
            'eta_for_integration': eta_for_integration,
            'source_closest_pid_per_source': source_closest_pid_per_source,
        },
        callbacks=callbacks
    )
    J_ref = J
    dispatch_callbacks(callbacks, 'on_after_step', i=i, J=J, J_ref=J, mu=mu_function, phi_new=phi, phi_old=None)

    J_old = None
    while not converged(i, J_old, J, J_ref):
        lambdas = _compute_lambdas_from_adjoint_problem(
            mesh,
            omega_list,
            sources,
            mu_for_integration,
            eta_for_integration,
            receivers,
            measured_data,
            simulated_data,
            K_for_omega,
        )

        phi_new = _compute_next_level_set(
            phi.reshape((mesh.n_points, 1)),
            mesh,
            omega_list,
            sources,
            lambdas,
            pressures,
            F0,
            F1,
            eta_in_nodes,
            J_ref,

            inversion_case,
            F0,
            F1,
            {
                'K_for_omega': K_for_omega,
                'mu_for_integration': mu_for_integration,
                'eta_for_integration': eta_for_integration,
                'source_closest_pid_per_source': source_closest_pid_per_source,
            },
            callbacks,
        ).reshape(phi.shape)

        mu_for_integration, mu_function = _build_mu_from_phi(mesh, phi_new, F0, F1)
        K_for_omega = _precompute_K_for_each_omega(mesh, omega_list, mu_for_integration, eta_for_integration)
        J_old = J
        J, simulated_data, pressures = _objective_f(inversion_case, mu_function, cache={
            'K_for_omega': K_for_omega,
            'mu_for_integration': mu_for_integration,
            'eta_for_integration': eta_for_integration,
            'source_closest_pid_per_source': source_closest_pid_per_source,
        })
        i += 1

        dispatch_callbacks(callbacks, 'on_after_step', i=i, J=J, mu=mu_function, phi_new=phi_new, phi_old=phi, J_ref=J_ref)
        phi = phi_new
    return phi


def _compute_lambdas_from_adjoint_problem(
    mesh,
    omega_list,
    sources,
    mu_for_integration,
    eta_for_integration,
    receivers,
    measured_data,
    simulated_data,
    K_for_omega=None,
):
    if K_for_omega is None:
        K_for_omega = {}

    lambdas = {}
    for source in sources:
        lambdas[source.name] = {}
        for omega in omega_list:
            lambdas[source.name][omega] = solve_adjoint_problem(
                mesh,
                omega,
                mu_for_integration,
                eta_for_integration,
                source.name,
                receivers,
                measured_data,
                simulated_data,
                K_for_omega.get(omega)
            )

    return lambdas


def _solve_source(mesh, source, receivers, omega_list, measured_data, mu_function, eta_function, cache):
    simulated_data_s = {}
    pressure_fields_s = {}
    J_s = 0.0
    for omega in omega_list:
        case = ForwardCase(
            mesh=mesh,
            omega=omega,
            mu_function=mu_function,
            eta_function=eta_function,
            source=source,
        )

        P = solve_2d_helmholtz(case, cache=cache)

        P_1d = P.reshape((mesh.n_points, 1))
        pressure_fields_s[omega] = P_1d

        for receiver in receivers:
            key_pair = (source.name, receiver.name, omega)
            closest_pid = mesh.closest_point_id(receiver.original_position)

            Z_rs = P_1d[closest_pid, 0].real
            S_rs = measured_data[key_pair]
            J_s += .5 * (Z_rs - S_rs) ** 2

            simulated_data_s[key_pair] = Z_rs
    return simulated_data_s, pressure_fields_s, J_s


def _objective_f(inversion_case, mu_function, cache=None, callbacks=None):
    if cache is None:
        cache = {}
    if callbacks is None:
        callbacks = {}
    omega_list = inversion_case.experimental_data.omega_list
    sources = inversion_case.experimental_data.sources
    receivers = inversion_case.experimental_data.receivers
    mesh = inversion_case.mesh
    eta_function = inversion_case.eta_function
    measured_data = inversion_case.experimental_data.measured_data
    simulated_data = {}

    J = 0.0
    pressures = {}
    results = joblib.Parallel(n_jobs=inversion_case.parallel_specs.n_parallel_sources, prefer='processes')(
        joblib.delayed(_solve_source)(
            mesh,
            source,
            receivers,
            omega_list,
            measured_data,
            mu_function,
            eta_function,
            cache
        )
        for source in sources
    )
    for source, (simulated_data_s, pressure_fields_s, J_s) in zip(sources, results):
        J += J_s
        pressures[source.name] = pressure_fields_s
        simulated_data.update(simulated_data_s)

    J *= inversion_case.d_omega
    return J, simulated_data, pressures


def _compute_next_level_set(
    phi,
    mesh,
    omega_list,
    sources,
    lambdas,
    pressures,
    mu0,
    mu1,
    eta_phi,
    J_ref,
    
    inversion_case,
    F0,
    F1,
    cache,
    
    callbacks,
):
    tau = 5e-6
    C_L = assembly(mesh, _fwi_ls.build_local_C_L)
    K_L = assembly(mesh, _fwi_ls.build_local_K_L, tau=tau)

    V = _compute_level_set_velocity(
        inversion_case,
        omega_list,
        inversion_case.d_omega,
        sources,
        lambdas,
        pressures,
        mu0,
        mu1,
        eta_phi
    )

    dispatch_callbacks(callbacks, 'on_after_compute_ls_velocity', V=V.reshape(mesh.nx + 1, mesh.ny + 1))
    return _search_new_phi(
        mesh,
        phi,
        F0,
        F1,
        inversion_case,
        K_L,
        C_L,
        V,
        J_ref,
        cache,
        callbacks
    )


def _search_new_phi(
    mesh,
    phi,
    F0,
    F1,
    inversion_case,
    K_L,
    C_L,
    V,
    J_ref,
    cache,
    callbacks
):
    def rebuild_mu(phi):
        return SmoothHeavySideBasedFunctionBuilder(
            phi.reshape(mesh.nx + 1, mesh.ny + 1),
            mesh.size_x,
            mesh.size_y,
            F0,
            F1,
            1.0
        ).build()

    def compute_mu_for(dtheta):
        K = (K_L + (1. / dtheta) * C_L)
        # Note: Must use phi, not phi_new here, in order to keep the previous
        # solution, and only change dtheta.
        f = C_L @ ((1. / dtheta) * phi - V)
        K, f = convert_complex_linear_system_to_real(K.tocoo(), f)
        phi_new = pypardiso.spsolve(K.tocsc(), f)
        phi_new = extract_complex_solution(phi_new)
        phi_new = phi_new.real
        mu_f = rebuild_mu(phi_new)
        obj_f, _, _ = _objective_f(inversion_case, mu_f, cache=None)
        return phi_new, obj_f

    initial_mu = rebuild_mu(phi)
    initial_obj, _, _ = _objective_f(inversion_case, initial_mu, cache)

    # Initial guess for dtheta
    dtheta = inversion_case.ls_evolution_options.initial_evolution_step
    phi_new, obj_new = compute_mu_for(dtheta)
    dispatch_callbacks(callbacks, 'on_before_compute_new_phi', initial_obj=initial_obj, obj_new=obj_new, J_ref=J_ref)
    while True:
        if dtheta <= inversion_case.ls_evolution_options.min_evolution_step:
            # Could not find a better level set using the current information.
            # Give up using the old one.
            phi_new = phi
            break

        phi_new, obj_new = compute_mu_for(dtheta)
        dispatch_callbacks(callbacks, 'on_compute_new_phi_iteration', dtheta=dtheta, obj_new=obj_new, J_ref=J_ref)

        if obj_new < initial_obj:
            # Found a good new phi.
            break
        dtheta /= 4.

    dispatch_callbacks(callbacks, 'on_after_compute_new_phi', initial_obj=initial_obj, obj_new=obj_new)

    return phi_new


def _compute_level_set_velocity(
    inversion_case,
    omega_list,
    d_omega,
    sources,
    lambdas,
    pressures,
    mu0,
    mu1,
    eta_phi
):
    mesh = inversion_case.mesh

    V_ref = 1e-9

    mu_phi = mu1 - mu0
    V = 0.0
    for source in sources:
        for omega in omega_list:
            lambda_P = lambdas[source.name][omega] * pressures[source.name][omega]
            V += (omega**2 * mu_phi - 1j * omega * eta_phi) * lambda_P * d_omega

    if inversion_case.ls_evolution_options.ignore_LS_V_at_sponge:
        V[eta_phi != 0.0] = 0.0

    return V / V_ref


def _precompute_K_for_each_omega(mesh, omega_list, mu_for_integration, eta_for_integration):
    from frequency_domain.forward_solver import _generate_K_for_omega

    K_for_omega = {}
    for omega in omega_list:
        K_for_omega[omega] = _generate_K_for_omega(mu_for_integration, eta_for_integration, omega, mesh)
    return K_for_omega


def _build_mu_from_phi(mesh, phi, f0, f1, h=1.0):
    mu_function = SmoothHeavySideBasedFunctionBuilder(
        phi,
        mesh.size_x,
        mesh.size_y,
        f0,
        f1,
        h
    ).build()

    mu_for_integration = np.empty(shape=(len(mesh.points_in_elements),))
    for i, points in enumerate(mesh.points_in_elements):
        x, y = np.mean(points, axis=0)
        mu_for_integration[i] = mu_function(x, y)

    return mu_for_integration, mu_function
