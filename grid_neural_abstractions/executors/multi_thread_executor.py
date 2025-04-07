import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from threading import Event, Thread
import threading

import numpy as np
from tqdm import tqdm  # Added tqdm for progress tracking


def update_progress_bar(pbar, worker_progress_counters, progress_done):
    """
    Update the progress bar based on worker progress counters.

    Args:
        pbar: tqdm progress bar instance
        worker_progress_counters: List of progress counters for each worker
        progress_done: Threading event to indicate when processing is complete
    """
    last_update = time.time()
    stall_timer = 0
    while not progress_done.is_set():
        try:
            total_progress = sum(
                map(lambda counter: counter.value, worker_progress_counters)
            )  # Aggregate progress
            pbar.n = total_progress
            pbar.refresh()

            # Check for progress stalls
            if total_progress > 0 and pbar.n == pbar.last_print_n:
                stall_timer += 1
                if stall_timer > 60:  # No progress for 1 minute
                    print("Warning: Progress appears stalled")
                    stall_timer = 0
            else:
                stall_timer = 0

            last_update = time.time()
            time.sleep(1)  # Update every second
        except Exception as e:
            print(f"Progress bar update error: {e}")
        time.sleep(1)  # Update every second


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
            with tqdm(total=num_samples, desc="Overall Progress") as pbar:
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
