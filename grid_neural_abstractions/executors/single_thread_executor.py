from .stats import Statistics
from tqdm import tqdm  # Added tqdm for progress tracking
from queue import LifoQueue
import time


class SinglethreadExecutor:
    def __init__(self, linearization_strategy, verification_strategy, linear_batch_size=1):
        self.linearization_strategy = linearization_strategy
        self.verification_strategy = verification_strategy
        self.linear_batch_size = linear_batch_size

    def execute(self, aggregate, samples, plotter=None):
        agg = None
        statistics = Statistics(samples)

        self.verification_strategy.initialize_worker()

        # Task queue (pre-linearize samples)
        queue = LifoQueue()
        linearization_result = self.linearization_strategy.linearize(samples)
        lin_batch = []
        for new_sample in linearization_result:
            queue.put(new_sample)

        start_time = time.time()

        # Initialize the progress bar
        with tqdm(desc="Overall Progress", smoothing=0.1) as pbar:
            while not queue.empty() or lin_batch:
                # Get the next sample from the queue and verify
                sample = queue.get()
                result = self.verification_strategy.verify_sample(sample)

                # Update statistics
                statistics.add_sample(result)

                # If plotting is enabled, update the plot
                if result.isleaf() and plotter is not None:
                    plotter.update_figure(result)

                # Store results however caller wants
                agg = aggregate(agg, result)

                # Add new results to the queue
                if result.hasnewsamples():
                    for new_sample in result.newsamples():
                        lin_batch.append(new_sample)

                # If the batch is full or the queue is empty, linearize the batch
                # and put the results back into the queue
                if len(lin_batch) >= self.linear_batch_size or queue.empty():
                    linearization_result = self.linearization_strategy.linearize(lin_batch)
                    lin_batch = []

                    for new_sample in linearization_result:
                        queue.put(new_sample)

                # Update the progress bar
                pbar.update(1)
                pbar.set_description_str(
                    (
                        f"Overall Progress (remaining samples: {queue.qsize()}, "
                        f"certified: {statistics.get_certified_percentage():.4f}%, "
                        f"uncertified: {statistics.get_uncertified_percentage():.4f}%)"
                    )
                )

        end_time = time.time()
        computation_time = end_time - start_time

        return agg, statistics.get_certified_percentage(), statistics.get_uncertified_percentage(), computation_time
