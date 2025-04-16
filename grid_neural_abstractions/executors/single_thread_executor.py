import types
from tqdm import tqdm  # Added tqdm for progress tracking
from queue import LifoQueue


class SinglethreadExecutor:
    def execute(self, initializer, process_sample, aggregate, samples):
        agg = None
        local = types.SimpleNamespace()
        initializer(local)

        # Calculate the total domain size
        total_domain_size = sum(sample.lebesguemeasure() for sample in samples)
        certified_domain_size = 0.0
        uncertified_domain_size = 0.0
        queue = LifoQueue()
        for sample in samples:
            queue.put(sample)

        with tqdm(desc="Overall Progress", smoothing=0.1) as pbar:
            while not queue.empty():
                sample = queue.get()

                # Execute the batches
                result = process_sample(local, sample)
                
                if result.issat():
                    # Sample was succesfully verified, no new samples to process
                    # Update certified domain size in a thread-safe manner
                    certified_domain_size += result.lebesguemeasure()
                
                if result.isunsat():
                    # Sample was not verified, add to the uncertified domain size
                    uncertified_domain_size += result.lebesguemeasure()

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
                    f"Overall Progress (remaining samples: {queue.qsize()}, certified: {certified_percentage:.2f}%, uncertified: {uncertified_percentage:.2f}%)"
                )

        return agg
