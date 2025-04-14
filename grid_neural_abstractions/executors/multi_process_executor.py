from concurrent.futures import ProcessPoolExecutor
from .multi_thread_executor import ExpandableAsCompleted
import types
from queue import LifoQueue

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
        
        # Calculate the total domain size
        total_domain_size = sum(sample.calculate_size() for sample in samples)
        certified_domain_size = 0

        with ProcessPoolExecutor(max_workers=self.num_workers, initializer=local.initialize) as executor:
            executor._work_ids = LifoQueue()

            with tqdm(desc="Overall Progress", smoothing=0.1) as pbar:
                futures = []
                for sample in samples:
                    future = executor.submit(local.process_sample, sample)
                    future.add_done_callback(lambda p: pbar.update())
                    futures.append(future)

                waiter = ExpandableAsCompleted(futures)

                for future in waiter.as_completed():
                    new_samples, result, certified_samples = future.result()
                    
                    for certified_sample in certified_samples:
                        # Sample was succesfully verified, no new samples to process
                        # Update certified domain size in a thread-safe manner
                        certified_domain_size += certified_sample.calculate_size()
                    
                    agg = aggregate(agg, result)

                    for new_sample in new_samples:
                        new_future = executor.submit(local.process_sample, new_sample)
                        new_future.add_done_callback(lambda p: pbar.update())
                        waiter.add(new_future)
                
                    certified_percentage = (certified_domain_size / total_domain_size) * 100
                    pbar.set_description_str(
                        f"Overall Progress (remaining samples: {len(waiter)}, certified: {certified_percentage:.2f}%)"
                    )
        return agg
