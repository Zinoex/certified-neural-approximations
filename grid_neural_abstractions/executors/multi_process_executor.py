from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from multiprocessing import Value
import threading
import types

import numpy as np
from tqdm import tqdm  # Added tqdm for progress tracking

from .multi_thread_executor import update_progress_bar


class MultiprocessRefLong:
    def __init__(self, value):
        self.shared_value = Value("i", value)

    def __iadd__(self, x):
        with self.shared_value.get_lock():
            self.shared_value.value += x

        return self

    @property
    def value(self):
        return self.shared_value.value


class Local:
    def __init__(self, initializer, process):
        self.initializer = initializer
        self.process = process

    def initialize(self):
        global _LOCAL
        _LOCAL = types.SimpleNamespace()
        self.initializer(_LOCAL)

    def process_sample(self, data):
        self.process(_LOCAL, data)


class MultiprocessExecutor:

    def __init__(self, num_workers=None):
        # If num_workers is not provided, use the default of ProcessPoolExecutor os.process_cpu_count()
        self.num_workers = num_workers

    def execute(
        self,
        initializer, process_sample, select_sample, num_samples, aggregate
    ):
        local = Local(initializer, process_sample)
        agg = None

        with ProcessPoolExecutor(max_workers=self.num_workers, initializer=local.initialize) as executor:
            with tqdm(total=num_samples, desc="Overall Progress", smoothing=0.1) as pbar:
                futures = []

                for i in range(num_samples):
                    data = select_sample(i)

                    future = executor.submit(local.process_sample, data)
                    future.add_done_callback(lambda p: pbar.update())
                    futures.append(future)

                for future in as_completed(futures):
                    result = future.result()
                    agg = aggregate(agg, result)

        return agg
