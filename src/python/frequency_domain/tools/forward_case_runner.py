import importlib

from frequency_domain import forward_solver
from frequency_domain.plotting import plot_case, plot_forward_results
import numpy as np


def run(case_name, omega, n, n_threads):
    """
    :param str case_name:
        Must be a valid file on experiments.forward_cases
    :param double omega:
        Frequency in rad/s
    :param int n:
        Number of elements in each side (n = nx = ny)
    :param int n_threads:
        Controls the number of threads for the linear solver
    """
    try:
        fwd_case = importlib.import_module(f"frequency_domain.experiments.forward_cases.{case_name}")
    except ModuleNotFoundError:
        print(f"Case not found: {case_name}")
        return

    import pypardiso
    pypardiso.scipy_aliases.pypardiso_solver.set_num_threads(n_threads)

    sources = fwd_case.sources()
    receivers = fwd_case.receivers()
    case = fwd_case.build_forward_case(None, None, n)
    plot_case(
        case,
        sources=np.array([s.original_position for s in sources]),
        receivers=np.array([r.original_position for r in receivers]),
    )

    results = []
    for i, s in enumerate(sources):
        print(f"Running case {i + 1}/{len(sources)}")
        case = fwd_case.build_forward_case(s, omega, n)
        pressure_field = forward_solver.solve_2d_helmholtz(case).real
        results.append(pressure_field)
    plot_forward_results(case, results)
