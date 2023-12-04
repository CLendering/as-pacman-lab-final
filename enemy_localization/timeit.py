from functools import wraps
import time
from collections import defaultdict
import numpy as np

times = defaultdict(list)

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        N = 1 if func.__name__ in ['__init__', 'register_initial_state'] else 100
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        times[func.__name__].append(total_time)
        if len(times[func.__name__]) % N == 0:
            print(f'Function {func.__name__} took {np.mean(times[func.__name__]):.4f} seconds on average over the last {len(times[func.__name__])} calls')
        return result
    return timeit_wrapper
