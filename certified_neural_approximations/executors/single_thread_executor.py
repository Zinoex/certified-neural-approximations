import time
import types
from tqdm import tqdm  # Added tqdm for progress tracking
from queue import LifoQueue


class SinglethreadExecutor:
    def execute(self, initializer, process_sample, aggregate, samples, plotter=None):
        agg = None
        initializer()

        # Calculate the total domain size
        total_domain_size = sum(sample.lebesguemeasure() for sample in samples)
        certified_domain_size = 0.0
        uncertified_domain_size = 0.0

        start_time = None

        queue = LifoQueue()
        for sample in samples:
            queue.put(sample)

        with tqdm(desc="Overall Progress", smoothing=0.1) as pbar:
            while not queue.empty():
                sample = queue.get()

                # Execute the batches
                result = process_sample(sample)

                if start_time is None:
                    start_time = result.start_time
                else:
                    start_time = min(start_time, result.start_time)
                
                if result.issat():
                    # Sample was succesfully verified, no new samples to process
                    # Update certified domain size in a thread-safe manner
                    certified_domain_size += result.lebesguemeasure()
                    # Update visualization if plotter is provided
                    if plotter is not None:
                        plotter.update_figure(result)
                
                if result.isunsat():
                    # Sample was not verified, add to the uncertified domain size
                    uncertified_domain_size += result.lebesguemeasure()
                    # Update visualization if plotter is provided
                    if plotter is not None:
                        plotter.update_figure(result)

                agg = aggregate(agg, result)

                if result.hasnewsamples():
                    # Get the new samples
                    new_samples = result.newsamples()

                    # Put the new samples back into the queue
                    for new_sample in new_samples:
                        queue.put(new_sample)
                
                pbar.update(1)
                certified_percentage = (certified_domain_size / total_domain_size) * 100
                uncertified_percentage = (uncertified_domain_size / total_domain_size) * 100
                
                pbar.set_description_str(
                    f"Overall Progress (remaining samples: {queue.qsize()}, certified: {certified_percentage:.4f}%, uncertified: {uncertified_percentage:.4f}%)"
                )

        end_time = time.time()
        computation_time = end_time - start_time

        return agg, certified_percentage, uncertified_percentage, computation_time
