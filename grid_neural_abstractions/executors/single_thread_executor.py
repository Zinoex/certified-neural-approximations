from threading import Event, Thread

import numpy as np
from tqdm import tqdm  # Added tqdm for progress tracking

from .multi_thread_executor import update_progress_bar, RefLong


class SinglethreadExecutor:
    def execute(self, process_batch, batch_selector, num_samples, aggregate):

        # Split the samples into batches
        data = batch_selector(0, num_samples)
        progress_counter = RefLong(0)  # One counter per worker
        progress_done = Event()  # Use Event for thread-safe completion signal

        result = None

        # Create a progress bar and run the verification
        with tqdm(total=num_samples, desc="Overall Progress") as pbar:
            # Start the progress tracking thread
            progress_thread = Thread(
                target=update_progress_bar,
                args=(pbar, [progress_counter], progress_done),
                daemon=True,
            )
            progress_thread.start()

            # Execute the batches
            try:
                result = process_batch(progress_counter, data)
            finally:
                # Ensure resources are cleaned up
                progress_done.set()
                progress_thread.join()
                print("Finished")

        return result
