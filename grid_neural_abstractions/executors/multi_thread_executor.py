from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from tqdm import tqdm  # Added tqdm for progress tracking


class MultithreadExecutor:
    """
    Note: Normally, the global interpreter lock (GIL) in Python can limit the performance of multi-threadeding.
    However, in this case, since Marabou is a C++ library and the heavy lifting is done in C++, the lock is released
    during the execution of the C++ code, thus not prohibiting the performance benefits of threading.
    """

    def __init__(self, num_workers=None):
        # If num_workers is not provided, use the default of ThreadPoolExecutor min(32, os.cpu_count() + 4)
        self.num_workers = num_workers

    def execute(
        self,
        initializer, process_sample, select_sample, num_samples, aggregate
    ):
        local = threading.local()
        agg = None

        with ThreadPoolExecutor(max_workers=self.num_workers, initializer=initializer, initargs=(local,)) as executor:
            with tqdm(total=num_samples, desc="Overall Progress", smoothing=0.1) as pbar:
                futures = []

                for i in range(num_samples):
                    data = select_sample(i)

                    future = executor.submit(process_sample, local, data)
                    future.add_done_callback(lambda p: pbar.update())
                    futures.append(future)

                for future in as_completed(futures):
                    result = future.result()
                    agg = aggregate(agg, result)

        return agg
