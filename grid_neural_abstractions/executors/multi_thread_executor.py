import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from threading import Event, Thread

import numpy as np
from atomic import AtomicLong
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
        process_batch,
        batch_selector,
        num_samples,
        aggregate,
    ):
        batch_size = int(
            np.ceil(num_samples / self.num_workers)
        )  # Round up to ensure all samples are covered

        # Split the samples into batches
        batches = [
            (batch_id, batch_selector(i, batch_size))
            for batch_id, i in enumerate(range(0, num_samples, batch_size))
        ]
        worker_progress_counters = [
            AtomicLong(0) for _ in range(self.num_workers)
        ]  # One counter per worker
        progress_done = Event()  # Use Event for thread-safe completion signal

        agg = None

        # Create a progress bar and run the verification
        with tqdm(total=num_samples, desc="Overall Progress") as pbar:
            # Start the progress tracking thread
            progress_thread = Thread(
                target=update_progress_bar,
                args=(pbar, worker_progress_counters, progress_done),
                daemon=True,
            )
            progress_thread.start()

            # Execute the batches
            try:
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = [
                        executor.submit(
                            process_batch,
                            worker_progress_counters,
                            batch_id,
                            data,
                        )
                        for batch_id, data in batches
                    ]

                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if agg is None:
                                agg = result
                            else:
                                agg = aggregate(agg, result)
                        except TimeoutError:
                            print("A batch timed out")
                        except Exception as e:
                            print(
                                f"Error in batch: {str(e)[:200]}"
                            )  # Limit error message length
            except Exception as e:
                print(f"Error during execution: {e}")
            finally:
                # Ensure resources are cleaned up
                progress_done.set()
                progress_thread.join()
                print("Finished")

        return agg
