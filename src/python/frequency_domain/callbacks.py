"""
Callbacks are used in many places in the project. They are used to print, debug or plot
results in the simulation and/or optimization process.

To use callbacks simply call:
    dispatch_callbacks(callbacks, CALLBACK_UNIQUE_NAME)
anywhere that has the 'callbacks' data structure (If not available, just pass as parameter)

To register callbacks in the main function just follow the example:
    inversion_solver.solve_fwi(
        inversion_case,
        callbacks=merge_callbacks([
            {
                CALLBACK_UNIQUE_NAME: [
                    function_1,
                    function_2,
                    ...,
                    function_n
                ],
                OTHER_CALLBACK_UNIQUE_NAME: [
                    function_1,
                    function_2,
                    ...,
                    function_n
                ],
                ...
            },
        ])
    )
"""


def dispatch_callbacks(callback_dict, callback_name, *args, **kwargs):
    callback_list = callback_dict.get(callback_name, [])
    for callback in callback_list:
        callback(*args, **kwargs)


def merge_callbacks(callbacks_list_of_dicts):
    result = {}
    for callbacks in callbacks_list_of_dicts:
        for k, v in callbacks.items():
            if k not in result:
                result[k] = v
            else:
                result[k] += v
    return result
