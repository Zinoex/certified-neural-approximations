from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from multiprocessing import Value
from threading import Event, Thread

import numpy as np
from tqdm import tqdm  # Added tqdm for progress tracking

from .multi_thread_executor import update_progress_bar


class MultiprocessRefLong:
    def __init__(self, value):
        self.shared_value = Value('i', value)

    def __iadd__(self, x):
        self.shared_value.value += x

        return self

    @property
    def value(self):
        return self.shared_value.value


class MultiprocessExecutor:

    def __init__(self, num_workers=None):
        # If num_workers is not provided, use the default of ProcessPoolExecutor os.process_cpu_count()
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
            MultiprocessRefLong(0) for _ in range(self.num_workers)
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
                            worker_progress_counters[batch_id],
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
                        # except Exception as e:
                        #     print(
                        #         f"Error in batch: {str(e)[:200]}"
                        #     )  # Limit error message length
            # except Exception as e:
            #     print(f"Error during execution: {e}")
            finally:
                # Ensure resources are cleaned up
                progress_done.set()
                progress_thread.join()
                print("Finished")

        return agg
