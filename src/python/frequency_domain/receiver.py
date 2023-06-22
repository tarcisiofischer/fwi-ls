import numpy as np


class Receiver:
    def __init__(self, name, original_position):
        self.name = name
        self.original_position = original_position


def build_receivers(start_position, end_position, amount, prefix=""):
    """
    Will produce several receiver from start_position to end_position. A prefix
    can be given, so that it'll be used ans the receiver name's prefix.
    """
    start_x, start_y = start_position
    end_x, end_y = end_position
    x_positions = np.linspace(start_x, end_x, amount)
    y_positions = np.linspace(start_y, end_y, amount)
    receivers = []
    for i, position in enumerate(zip(x_positions, y_positions)):
        receivers.append(
            Receiver(f"Receiver {prefix}{i}", position)
        )
    return receivers
