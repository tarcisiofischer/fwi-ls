class ForwardCase():
    """
    Represents the forward (direct) simulation problem
    """
    def __init__(self, mesh, omega, mu_function, eta_function, source):
        self.mesh = mesh
        self.omega = omega
        self.mu_function = mu_function
        self.eta_function = eta_function
        self.source = source
