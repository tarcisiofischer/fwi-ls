class InversionCase:
    def __init__(
        self,
        mesh,
        experimental_data,
        initial_guess,
        converged,
        eta_function,
        omega_i,
        omega_f,
        n_omegas,
        ls_evolution_options=None,
        parallel_specs=None
    ):
        if parallel_specs is None:
            parallel_specs = ParallelSpecs(1, 1)
        self.mesh = mesh
        self.experimental_data = experimental_data
        self.initial_guess = initial_guess
        self.converged = converged
        self.eta_function = eta_function
        self.omega_i = omega_i
        self.omega_f = omega_f
        self.n_omega = n_omegas
        self.d_omega = (omega_f - omega_i) / n_omegas
        self.parallel_specs = parallel_specs
        self.ls_evolution_options = ls_evolution_options


class ParallelSpecs:
    def __init__(self, n_parallel_sources=1, n_parallel_linear_solver=1):
        self.n_parallel_sources = n_parallel_sources
        self.n_parallel_linear_solver = n_parallel_linear_solver


class LSEvolutionOptions:
    def __init__(self, ignore_LS_V_at_sponge, initial_evolution_step, min_evolution_step):
        self.ignore_LS_V_at_sponge = ignore_LS_V_at_sponge
        self.initial_evolution_step = initial_evolution_step
        self.min_evolution_step = min_evolution_step
