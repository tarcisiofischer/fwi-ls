class ExperimentalData:
    """
    Represents the "experimental" data from a set of sources and receivers.
    The actual data is stored in 'measured_data', and follows the example:

    measured_data={
        (SOURCE_NAME, RECEIVER_NAME, FREQUENCY): VALUE,
        (SOURCE_NAME, RECEIVER_NAME, FREQUENCY): VALUE,
        ...
    }
    """
    def __init__(self, sources, receivers, omega_list, measured_data):
        self.sources = sources
        self.receivers = receivers
        self.omega_list = omega_list
        self.measured_data = measured_data
