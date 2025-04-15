from functools import partial

import numpy as np
from executors import (
    MultiprocessExecutor,
    MultithreadExecutor,
    SinglethreadExecutor,
)
from maraboupy import Marabou
from train_nn import generate_data
from dynamics import VanDerPolOscillator, Quadcopter

from verification import MarabouLipschitzStrategy, MarabouTaylorStrategy
from certification_results import Region


def process_sample(
    strategy,
    dynamics_model,
    epsilon,
    local,
    data,
    precision=1e-6
):
    """
    Process a batch of input points to check for counterexamples.

    Args:
        network: The Marabou network to verify
        batch: List of (sample_number, sample) tuples to process
        batch_id: ID of the current batch
        worker_progress_counters: List to track progress across workers
        worker_locks: List of locks for each worker
        delta: The delta parameter for verification
        L: The Lipschitz constant
        y_train: Training output data

    Returns:
        - split regions: List of regions to be processed further, if any
        - counterexamples: List of counterexamples found
        - the original sample: The original sample, if Marabou fails to find a counterexample
    """
    network = local.network
    return strategy.verify(
        network,
        dynamics_model,
        data,
        epsilon=epsilon,
        precision=precision
    )


# This function has to be in the global scope to be pickled for multiprocessing
def read_onnx_into_local(onnx_path, local):
    network = Marabou.read_onnx(onnx_path)
    local.network = network


def aggregate(agg, result):
    if not result.isunsat():
        return agg

    if agg is None:
        return result.counterexamples()
    
    return agg + result.counterexamples()


def verify_nn(
    onnx_path, delta=0.01, epsilon=0.1, num_workers=16
):
    strategy = MarabouTaylorStrategy()
    dynamics_model = VanDerPolOscillator()

    input_dim = dynamics_model.input_dim

    # Compute the number of samples for a fixed grid
    range_min, range_max = -1.0, 1.0  # Match the range in generate_data
    num_samples_per_dim = int((range_max - range_min) / delta) + 1
    num_samples = num_samples_per_dim**input_dim
    print(f"Number of initial samples: {num_samples}")

    partial_process_sample = partial(process_sample, strategy, dynamics_model, epsilon)

    X_train, _ = generate_data(input_dim, delta=delta, grid=True)
    samples = [
        Region(X_train[i], np.full_like(X_train[i], delta)) for i in range(num_samples)
    ]

    initializer = partial(read_onnx_into_local, onnx_path)

    executor = SinglethreadExecutor()

    cex_list = executor.execute(initializer, partial_process_sample, aggregate, samples)
    num_cex = len(cex_list) if cex_list else 0

    print(f"Number of counterexamples found: {num_cex}")
    print("Finished")


if __name__ == "__main__":
    verify_nn(
        "data/simple_nn.onnx",
        delta=0.1
    )
