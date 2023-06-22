from frequency_domain import ricker_pulse
import numpy as np
from frequency_domain.ricker_pulse import hz_to_rads
from matplotlib import pyplot as plt


omega_domain = np.linspace(0.0, 100.0, 200)
for omega_0 in [hz_to_rads(2.0), hz_to_rads(4.0), hz_to_rads(6.0)]:
    plt.plot(omega_domain, ricker_pulse.ricker(omega_0, omega_domain))
plt.show()
