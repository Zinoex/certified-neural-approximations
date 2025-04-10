from concurrent.futures import ProcessPoolExecutor
from .multi_thread_executor import ExpandableAsCompleted
import types

from tqdm import tqdm  # Added tqdm for progress tracking


class Local:
    def __init__(self, initializer, process):
        self.initializer = initializer
        self.process = process

    def initialize(self):
        global _LOCAL
        _LOCAL = types.SimpleNamespace()
        self.initializer(_LOCAL)

    def process_sample(self, sample):
        return self.process(_LOCAL, sample)


class MultiprocessExecutor:

    def __init__(self, num_workers=None):
        # If num_workers is not provided, use the default of ProcessPoolExecutor os.process_cpu_count()
        self.num_workers = num_workers

    def execute(
        self,
        initializer, process_sample, aggregate, samples
    ):
        local = Local(initializer, process_sample)
        agg = None

        with ProcessPoolExecutor(max_workers=self.num_workers, initializer=local.initialize) as executor:
            with tqdm(desc="Overall Progress", smoothing=0.1) as pbar:
                futures = []
                for sample in samples:
                    future = executor.submit(local.process_sample, sample)
                    future.add_done_callback(lambda p: pbar.update())
                    futures.append(future)

                waiter = ExpandableAsCompleted(futures)

                for future in waiter.as_completed():
                    new_samples, result = future.result()
                    pbar.set_description_str(f"Overall Progress (remaining samples: {len(waiter)})")
                    agg = aggregate(agg, result)

                    for new_sample in new_samples:
                        new_future = executor.submit(local.process_sample, new_sample)
                        new_future.add_done_callback(lambda p: pbar.update())
                        waiter.add(new_future)

        return agg
