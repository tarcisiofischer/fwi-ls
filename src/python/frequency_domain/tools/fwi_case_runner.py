import importlib

from frequency_domain import inversion_solver
from frequency_domain.callbacks import merge_callbacks
import numpy as np
from frequency_domain.function_builders import generate_discrete_field
from frequency_domain.inversion_case import ParallelSpecs, LSEvolutionOptions


def run_fwi_case(
    case_name,
    n_iterations=1,
    n_threads=1,
    plot_results=True,
    ignore_LS_V_at_sponge=False,
    initial_evolution_step=1.0,
    min_evolution_step=1e-12,
    minimum_obj_function_ratio_change=1e-20,
):
    try:
        fwd_case = importlib.import_module(f"frequency_domain.experiments.forward_cases.{case_name}")
    except ModuleNotFoundError:
        print(f"Forward case not found: {case_name}")
        return
    try:
        fwi_case = importlib.import_module(f"frequency_domain.experiments.inversion_cases.{case_name}_fwi")
    except ModuleNotFoundError:
        print(f"Inversion case not found: {case_name}")
        return

    print(f"Running case. Will run {n_iterations} iterations. (Solving with {n_threads} threads).")

    CONVERGED = True
    STILL_ITERATING = False
    def convergence_test(i, J_old, J_new, J_ref, *args, **kwargs):
        if i >= n_iterations:
            return CONVERGED
        if J_old is not None and (J_old - J_new) / J_ref <= minimum_obj_function_ratio_change:
            return CONVERGED
        return STILL_ITERATING

    phi_initial_guess = -1.0 * np.ones(shape=(61, 61))
    inversion_case = fwi_case.build_inversion_case(
        phi_initial_guess,
        fwd_case.F1,
        fwd_case.F2,
        fwd_case.eta_function(),
        convergence_test,
    )
    inversion_case.parallel_specs = ParallelSpecs(
        n_parallel_sources=n_threads,
        n_parallel_linear_solver=1
    )
    inversion_case.ls_evolution_options = LSEvolutionOptions(
        ignore_LS_V_at_sponge=ignore_LS_V_at_sponge,
        initial_evolution_step=initial_evolution_step,
        min_evolution_step=min_evolution_step,
    )
    obj_function = []
    i = 0

    def monitor(J, J_ref, *args, **kwargs):
        nonlocal i
        if i == 0:
            print(f"Initial guess. Residual = {J} ({J / J_ref})")
        else:
            print(f"Iteration {i}. Residual = {J} ({J / J_ref})")
        i += 1

    # def display_velocity(V, *args, **kwargs):
    #     from matplotlib import pyplot as plt
    #     plt.imshow(V.real)
    #     plt.show()

    phi = inversion_solver.solve_fwi(
        inversion_case,
        callbacks=merge_callbacks([
            {
                'on_before_solve_fwi': [
                    lambda *args, **kwargs: print(f"Preparing internal data structures...")
                ],
                'on_before_first_evaluation': [
                    lambda *args, **kwargs: print(f"Computing first evaluation of objective function...")
                ],
                'on_before_solve_obj': [
                    lambda s_i, o_i, n_sources, n_omega, *args, **kwargs: print(f"Solving objective function... Source {s_i}/{n_sources}, Omega {o_i}/{n_omega}"),
                ],
                'on_after_step': [
                    monitor,
                    lambda J, J_ref, *args, **kwargs: obj_function.append(J / J_ref),
                ],
                # 'on_after_compute_ls_velocity': [
                #     display_velocity,
                # ],
                'on_before_compute_new_phi': [
                    lambda initial_obj, J_ref, *args, **kwargs: print(f"Searching for new phi... [Objective function={initial_obj / J_ref}]")
                ],
                'on_compute_new_phi_iteration': [
                    lambda obj_new, J_ref, dtheta, *args, **kwargs: print(f"... [Objective function={obj_new / J_ref}] [dtheta={dtheta}]")
                ],
                'on_after_compute_new_phi': [
                    lambda initial_obj, obj_new: print("Found new phi!") if obj_new < initial_obj else print("Could not evolve LS (DIVERGED)")
                ],
            },
        ])
    )

    from matplotlib import pyplot as plt
    plt.subplot(2, 2, 1)
    plt.title('Objective Function')
    plt.plot(range(len(obj_function)), obj_function, 'ro-')

    plt.subplot(2, 2, 2)
    plt.title('Expected solution')
    plt.imshow(generate_discrete_field(fwd_case.mesh(), fwd_case.mu_function()))

    plt.subplot(2, 2, 3)
    plt.title('Initial guess')
    plt.imshow(phi_initial_guess)

    plt.subplot(2, 2, 4)
    plt.title(f'Final solution for {i} iterations')
    plt.imshow(phi > 0)

    print("Finished successfully!")
    if plot_results:
        plt.show()
