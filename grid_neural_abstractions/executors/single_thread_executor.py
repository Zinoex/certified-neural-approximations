import types
from tqdm import tqdm  # Added tqdm for progress tracking
from queue import SimpleQueue


class SinglethreadExecutor:
    def execute(self, initializer, process_sample, aggregate, samples):
        agg = None
        local = types.SimpleNamespace()
        initializer(local)

        queue = SimpleQueue()
        for sample in samples:
            queue.put(sample)

        with tqdm(desc="Overall Progress", smoothing=0.1) as pbar:
            while not queue.empty():
                sample = queue.get()

                # Execute the batches
                new_samples, result = process_sample(local, sample)
                agg = aggregate(agg, result)

                # Update the progress bar
                pbar.update(1)

                # Put the new samples back into the queue
                for new_sample in new_samples:
                    queue.put(new_sample)

        return agg
