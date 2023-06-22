import numpy as np

from frequency_domain.ricker_pulse import ricker


class ConstantSourceExpression:
    def __init__(self, value):
        self.value = value

    def __call__(self, omega):
        return self.value

    def __repr__(self):
        return f"ConstantSourceExpression({self.value})"


class RickerPulseSourceExpression:
    def __init__(self, omega_0):
        self.omega_0 = omega_0

    def __call__(self, omega):
        return ricker(self.omega_0, omega)

    def __repr__(self):
        return f"RickerPulseSourceExpression({self.omega_0})"


class Source:
    def __init__(self, name, original_position, expression):
        self.name = name
        self.original_position = original_position
        self.expression = expression


def build_sources(start_position, end_position, amount, expression):
    """
    Helper function to produce various sources from start_position to
    end_position. Source will produce a signal in the given amplitude, and is
    independent from a frequency.
    """
    start_x, start_y = start_position
    end_x, end_y = end_position
    x_positions = np.linspace(start_x, end_x, amount)
    y_positions = np.linspace(start_y, end_y, amount)
    sources = []
    for i, position in enumerate(zip(x_positions, y_positions)):
        sources.append(
            Source(f"Source {i}", position, expression)
        )
    return sources
