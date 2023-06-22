import pstats
try:
    import cProfile as profile
except ImportError:
    import profile


def profile_method(filename, rows=50, sort=(('cumul',), ('time',))):
    """
    Decorator to profile the decorated function or method.

    To use:

    @profile_method('out.prof')
    def my_func_to_profile():
        ...

    my_func_to_profile()
    """
    def wrapper(method):
        def inner(*args, **kwargs):
            prof = profile.Profile()
            result = prof.runcall(method, *args, **kwargs)
            prof.dump_stats(filename)
            tup_sort = sort
            s = tup_sort[0]
            if isinstance(s, str):
                tup_sort = [tup_sort]

            stats = pstats.Stats(prof)
            for s in tup_sort:
                stats.strip_dirs().sort_stats(*s).print_stats(int(rows))

            return result
        return inner
    return wrapper
