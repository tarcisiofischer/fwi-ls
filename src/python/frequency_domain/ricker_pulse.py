def ricker(omega_0, omega):
    """
    Ricker pulse in frequency domain.
    omegas must be given in rad/s

    :param float omega_0:
        Central frequency (rad/s)
    :param float omega:
        Domain frequency (rad/s)
    """
    import numpy as np
    return (2. * omega**2) / (np.sqrt(np.pi) * omega_0**3) * np.e ** (-omega**2 / omega_0**2)


def hz_to_rads(hz):
    """
    Helper function to convert hz to rad/s
    """
    return 6.283185 * hz
