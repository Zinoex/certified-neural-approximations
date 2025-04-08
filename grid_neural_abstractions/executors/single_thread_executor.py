import types
from tqdm import tqdm  # Added tqdm for progress tracking


class SinglethreadExecutor:
    def execute(self, initializer, process_sample, aggregate, samples):
        agg = None
        local = types.SimpleNamespace()
        initializer(local)

        # Create a progress bar and run the verification
        for sample in tqdm(samples, desc="Overall Progress", smoothing=0.1):

            # Execute the batches
            result = process_sample(local, sample)
            agg = aggregate(agg, result)

        return agg
