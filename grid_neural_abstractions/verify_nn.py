from functools import partial

import numpy as np
from .executors import (
    MultiprocessExecutor,
    MultithreadExecutor,
    SinglethreadExecutor,
)
from maraboupy import Marabou
from .dynamics import VanDerPolOscillator

from .verification import MarabouLipschitzStrategy, MarabouTaylorStrategy
from .certification_results import CertificationRegion
from .visualization import DynamicsNetworkPlotter  # Import the new plotter

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
    onnx_path, dynamics_model, delta=0.01, epsilon=0.1, num_workers=1, visualize=True
):
    
    strategy = MarabouTaylorStrategy(dynamics_model)

    input_dim = dynamics_model.input_dim
    onnx_input_dim = onnx_input_shape(onnx_path)
    assert len(onnx_input_dim) == 1, f"Only 1D input dims are supported, was {len(onnx_input_dim)}"
    assert onnx_input_dim[0] == input_dim, f"Input dim mismatch: {onnx_input_dim[0]} != {input_dim}"

    partial_process_sample = partial(process_sample, strategy, dynamics_model, epsilon)

    delta = np.full(input_dim, delta)
    X_train, _ = generate_data(input_dim, dynamics_model.input_domain, delta=delta, grid=True, device="cpu")
    output_dim = dynamics_model.output_dim
    samples = [
        CertificationRegion(x.double().numpy(), delta, j)
        for j in range(output_dim) for x in X_train
    ]

    initializer = partial(read_onnx_into_local, onnx_path)

    # Initialize plotter if visualization is enabled (supports both 1D and 2D)
    plotter = None
    if visualize and input_dim in [1, 2]:
        plotter = DynamicsNetworkPlotter(dynamics_model, Marabou.read_onnx(onnx_path))
        print(f"Initialized visualization for {input_dim}D dynamics")

    if num_workers == 1:
        executor = SinglethreadExecutor()
        # Pass the plotter to the executor
        cex_list = executor.execute(initializer, partial_process_sample, aggregate, samples, plotter)
    elif num_workers > 1:
        executor = MultiprocessExecutor(num_workers)
        # Note: Visualization is not supported in multiprocessing mode
        cex_list = executor.execute(initializer, partial_process_sample, aggregate, samples)

    num_cex = len(cex_list) if cex_list else 0

    print(f"Number of counterexamples found: {num_cex}")
    print("Finished")
    
    # Keep the plot window open if we're visualizing
    if plotter is not None:
        input("Press Enter to close the visualization...")


if __name__ == "__main__":
    verify_nn(
        "data/simple_nn.onnx",
        VanDerPolOscillator(),
        delta=0.1
    )
