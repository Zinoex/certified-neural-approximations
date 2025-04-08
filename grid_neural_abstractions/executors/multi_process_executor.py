from concurrent.futures import ProcessPoolExecutor, as_completed
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
            with tqdm(total=len(samples), desc="Overall Progress", smoothing=0.1) as pbar:
                futures = []

                for sample in samples:
                    future = executor.submit(local.process_sample, sample)
                    future.add_done_callback(lambda p: pbar.update())
                    futures.append(future)

                for future in as_completed(futures):
                    result = future.result()
                    agg = aggregate(agg, result)

        return agg
