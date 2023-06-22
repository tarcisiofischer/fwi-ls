import invoke as inv
import pathlib

CRED = '\033[91m'
CEND = '\033[0m'


def _setup_pythonpath():
    import sys
    current_path = pathlib.Path(__file__).parent.absolute()
    sys.path += [f"{current_path}/src/cpp/build", f"{current_path}/src/python"]


@inv.task
def run_fwi_case(
    c,
    case_name="",
    iterations=10,
    threads=1,
    no_plot=False,
    ignore_LS_V_at_sponge=False,
    initial_evolution_step=1.0,
    min_evolution_step=1e-12,
    min_obj_function_ratio_change=1e-20,
):
    _setup_pythonpath()

    from frequency_domain.tools import fwi_case_runner
    fwi_case_runner.run_fwi_case(
        case_name,
        iterations,
        threads,
        not no_plot,
        ignore_LS_V_at_sponge,
        initial_evolution_step,
        min_evolution_step,
        min_obj_function_ratio_change,
    )


@inv.task
def generate_fwi_case(c, case_name=""):
    _setup_pythonpath()

    import importlib
    from frequency_domain.tools.generate_fwi_problem import generate_inverse_crime

    try:
        fwd_case = importlib.import_module(f"frequency_domain.experiments.forward_cases.{case_name}")
    except ModuleNotFoundError:
        print(f"{CRED}Case not found: {case_name}{CEND}")
        return

    current_path = pathlib.Path(__file__).parent.absolute()
    output_file = f'{current_path}/src/python/frequency_domain/experiments/inversion_cases/{case_name}_fwi.py'

    generate_inverse_crime(
        fwd_case.build_forward_case,
        fwd_case.mesh(),
        fwd_case.sources(),
        fwd_case.receivers(),
        fwd_case.OMEGA_I,
        fwd_case.OMEGA_F,
        fwd_case.N_OMEGAS,
        output_file
    )


@inv.task
def run_fwd_case(c, case_name="", omega=5.0, size=80, threads=1):
    _setup_pythonpath()

    from frequency_domain.tools import forward_case_runner
    forward_case_runner.run(case_name, omega, size, threads)


@inv.task
def compile(c):
    with c.cd('src/cpp'):
        c.run(f'mkdir -p build')
        with c.cd('build'):
            c.run(f'cmake -DPYTHON_EXECUTABLE=$(which python) ..')
            c.run(f'make')


@inv.task
def clean(c):
    c.run(f'rm -rf src/cpp/build')
