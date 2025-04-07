import numpy as np
from atomic import AtomicLong
from tqdm import tqdm  # Added tqdm for progress tracking
from threading import Thread, Event
from .multi_thread_executor import update_progress_bar


class SinglethreadExecutor:
    def execute(self, process_batch, batch_selector, num_samples, aggregate):

        # Split the samples into batches
        data = batch_selector(0, num_samples)
        worker_progress_counters = [AtomicLong(0)]  # One counter per worker
        progress_done = Event() # Use Event for thread-safe completion signal

        result = None

        # Create a progress bar and run the verification
        with tqdm(total=num_samples, desc="Overall Progress") as pbar:
            # Start the progress tracking thread
            progress_thread = Thread(target=update_progress_bar, 
                                    args=(pbar, worker_progress_counters, progress_done), 
                                    daemon=True)
            progress_thread.start()
            
            # Execute the batches
            try:
                result = process_batch(worker_progress_counters, 0, data)
            finally:
                # Ensure resources are cleaned up
                progress_done.set()
                progress_thread.join()
                print("Finished")

        return result