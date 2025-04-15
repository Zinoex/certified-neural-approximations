from functools import partial

import numpy as np
from .executors import (
    MultiprocessExecutor,
    MultithreadExecutor,
    SinglethreadExecutor,
)
from maraboupy import Marabou
from .dynamics import VanDerPolOscillator, Quadcopter

from .verification import MarabouLipschitzStrategy, MarabouTaylorStrategy
from .certification_results import Region

from .train_nn import generate_data

def process_sample(
    strategy,
    dynamics_model,
    epsilon,
    local,
    data,
    precision=1e-6
):
    """
    Process a sample to check for counterexamples.

    Args:
        strategy: The verification strategy to use.
        dynamics_model: The dynamics model being verified.
        epsilon: The perturbation bound for verification.
        local: Local instance for the network.
        data: The input data sample.
        precision: The precision for verification.

    Returns:
        The result of the verification process.
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


def onnx_input_shape(onnx_path):
    """
    Get the input shape of the ONNX model.
    """
    network = Marabou.read_onnx(onnx_path)
    inputVars = network.inputVars
    return inputVars[0].shape[1:]


def aggregate(agg, result):
    if not result.isunsat():
        return agg

    if agg is None:
        return result.counterexamples()
    
    return agg + result.counterexamples()


def verify_nn(
    onnx_path, delta=0.01, epsilon=0.1, num_workers=1, dynamics_model=None
):
    if dynamics_model is None:
        dynamics_model = Quadcopter()
    
    strategy = MarabouTaylorStrategy(dynamics_model)

    input_dim = dynamics_model.input_dim
    onnx_input_dim = onnx_input_shape(onnx_path)
    assert len(onnx_input_dim) == 1, f"Only 1D input dims are supported, was {len(onnx_input_dim)}"
    assert onnx_input_dim[0] == input_dim, f"Input dim mismatch: {onnx_input_dim[0]} != {input_dim}"

    # Compute the number of samples for a fixed grid
    range_min, range_max = -1.0, 1.0  # Match the range in generate_data
    num_samples_per_dim = int((range_max - range_min) / delta) + 1
    num_samples = num_samples_per_dim**input_dim
    print(f"Number of initial samples: {num_samples}")

    partial_process_sample = partial(process_sample, strategy, dynamics_model, epsilon)

    X_train, _ = generate_data(input_dim, delta=delta, grid=True, device="cpu")
    samples = [
        Region(X_train[i].numpy(), np.full_like(X_train[i], delta)) for i in range(num_samples)
    ]

    initializer = partial(read_onnx_into_local, onnx_path)

    if num_workers == 1:
        executor = SinglethreadExecutor()
    elif num_workers > 1:
        executor = MultiprocessExecutor(num_workers)

    cex_list = executor.execute(initializer, partial_process_sample, aggregate, samples)
    num_cex = len(cex_list) if cex_list else 0

    print(f"Number of counterexamples found: {num_cex}")
    print("Finished")


if __name__ == "__main__":
    verify_nn(
        "data/simple_nn.onnx",
        delta=0.1
    )
