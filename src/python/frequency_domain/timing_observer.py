import time
import functools


class WallClockTiming:

    def __init__(self):
        self.measurements = {}

    def register(self, name):
        self.measurements[name] = {
            'start_time': None,
            'stop_time': None,
        }

    def start(self, name, *args, **kwargs):
        assert name in self.measurements, f"Unknown measurement '{name}'"
        self.measurements[name]['start_time'] = time.time()

    def stop(self, name, *args, **kwargs):
        assert name in self.measurements, f"Unknown measurement '{name}'"
        self.measurements[name]['stop_time'] = time.time()

    def print_report(self):
        report_str = 'Wall Clock Timing Report\n'
        report_str += '=' * len(report_str) + '\n'
        for name, measurement in self.measurements.items():
            report_str += f'{name}: '
            if measurement['start_time'] is None:
                report_str += f'never started'
            elif measurement['stop_time'] is None:
                report_str += f'never stoped'
            else:
                total_time = measurement['stop_time'] - measurement['start_time']
                report_str += f'{total_time} [s]'
            report_str += '\n'
        print(report_str)


def get_callbacks():
    wc_timing = WallClockTiming()
    wc_timing.register('prepare_linear_system')
    wc_timing.register('convert_complex_linear_system_to_real')
    wc_timing.register('solve')
    wc_timing.register('solve_2d_helmholtz')

    return {
        'on_before_solve_2d_helmholtz': [functools.partial(wc_timing.start, name='solve_2d_helmholtz'),],

        'on_before_prepare_linear_system': [functools.partial(wc_timing.start, name='prepare_linear_system'),],
        'on_after_prepare_linear_system': [functools.partial(wc_timing.stop, name='prepare_linear_system'),],

        'on_before_convert_complex_linear_system_to_real': [functools.partial(wc_timing.start, name='convert_complex_linear_system_to_real'),],
        'on_after_convert_complex_linear_system_to_real': [functools.partial(wc_timing.stop, name='convert_complex_linear_system_to_real'),],

        'on_before_solve': [functools.partial(wc_timing.start, name='solve'),],
        'on_after_solve': [functools.partial(wc_timing.stop, name='solve'),],

        'on_after_solve_2d_helmholtz': [
            functools.partial(wc_timing.stop, name='solve_2d_helmholtz'),
            lambda *args, **kwargs: wc_timing.print_report(),
        ],
    }
