import types
from tqdm import tqdm  # Added tqdm for progress tracking
from queue import LifoQueue


class SinglethreadExecutor:
    def execute(self, initializer, process_sample, aggregate, samples):
        agg = None
        local = types.SimpleNamespace()
        initializer(local)

        # Calculate the total domain size
        total_domain_size = sum(sample.calculate_size() for sample in samples)
        certified_domain_size = 0
        queue = LifoQueue()
        for sample in samples:
            queue.put(sample)

        with tqdm(desc="Overall Progress", smoothing=0.1) as pbar:
            while not queue.empty():
                sample = queue.get()

                # Execute the batches
                new_samples, result, certified_samples = process_sample(local, sample)
                
                for certified_sample in certified_samples:
                        # Sample was succesfully verified, no new samples to process
                        # Update certified domain size in a thread-safe manner
                        certified_domain_size += certified_sample.calculate_size()
                
                agg = aggregate(agg, result)

                # Put the new samples back into the queue
                for new_sample in new_samples:
                    queue.put(new_sample)
                
                pbar.update(1)
                certified_percentage = (certified_domain_size / total_domain_size) * 100
                pbar.set_description_str(
                    f"Overall Progress (remaining samples: {queue.qsize()}, certified: {certified_percentage:.2f}%)"
                )

        return agg
