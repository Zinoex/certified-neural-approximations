import time
from tqdm import tqdm  # Added tqdm for progress tracking
from queue import LifoQueue

from .stats import Statistics


class SinglethreadExecutor:
    def execute(self, process_sample, aggregate, samples, plotter=None):
        agg = None
        statistics = Statistics(samples)

        start_time = None

        # Use a LifoQueue to achieve DFS (Depth-First Search)-like behavior.
        # For a single-threaded executor, this is true DFS, but for a multi-threaded
        # executor, it depends on the order results are available.
        queue = LifoQueue()
        for sample in samples:
            queue.put(sample)

        with tqdm(desc="Overall Progress", smoothing=0.1) as pbar:
            while not queue.empty():
                sample = queue.get()

                # Execute the batches
                result = process_sample(sample)

                # Take earliest start time from all futures.
                # This is to subtract the process spawn time
                # from the computation time.
                if start_time is None:
                    start_time = result.start_time
                else:
                    start_time = min(start_time, result.start_time)

                # Update statistics
                statistics.add_sample(result)

                # Update visualization if plotter is provided
                if result.isleaf() and plotter is not None:
                    plotter.update_figure(result)

                # Store results however caller wants
                agg = aggregate(agg, result)

                # Add new results to the queue
                if result.hasnewsamples():
                    # Get the new samples
                    new_samples = result.newsamples()

                    # Put the new samples back into the queue
                    for new_sample in new_samples:
                        queue.put(new_sample)

                # Update the progress bar
                pbar.update(1)
                pbar.set_description_str(
                    f"Overall Progress (remaining samples: {queue.qsize()}, "
                    f"certified: {statistics.get_certified_percentage():.4f}%, "
                    f"uncertified: {statistics.get_uncertified_percentage():.4f}%)"
                )

        end_time = time.time()
        computation_time = end_time - start_time

        return agg, statistics.get_certified_percentage(), statistics.get_uncertified_percentage(), computation_time
