import types
from tqdm import tqdm  # Added tqdm for progress tracking
from queue import LifoQueue


class SinglethreadExecutor:
    def execute(self, initializer, process_sample, aggregate, samples):
        agg = None
        local = types.SimpleNamespace()
        initializer(local)

        queue = LifoQueue()
        for sample in samples:
            queue.put(sample)

        with tqdm(desc="Overall Progress", smoothing=0.1) as pbar:
            while not queue.empty():
                sample = queue.get()

                # Execute the batches
                new_samples, result = process_sample(local, sample)
                agg = aggregate(agg, result)

                # Put the new samples back into the queue
                for new_sample in new_samples:
                    queue.put(new_sample)

                # Update the progress bar
                pbar.set_description_str(f"Overall Progress (remaining samples: {queue.qsize()})")
                pbar.update(1)

        return agg
