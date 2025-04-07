import threading
from tqdm import trange  # Added tqdm for progress tracking


class SinglethreadExecutor:
    def execute(self, initializer, process_sample, select_sample, num_samples, aggregate):
        agg = None
        local = threading.local()
        initializer(local)

        # Create a progress bar and run the verification
        for i in trange(num_samples, desc="Overall Progress", smoothing=0.1):
            data = select_sample(i)

            # Execute the batches
            result = process_sample(local, data)
            agg = aggregate(agg, result)

        return agg
