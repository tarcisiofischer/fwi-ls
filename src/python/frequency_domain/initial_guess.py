class InitialGuess:
    """
    Initial guess for the inversion solver. Phi is the Level Set phi function,
    while F0 and F1 are the two values for the field components.
    """
    
    def __init__(self, phi, F0, F1):
        self.phi = phi
        self.F0 = F0
        self.F1 = F1
