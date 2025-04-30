from functools import partial

import numpy as np

from grid_neural_abstractions.train_nn import load_onnx_model
from .executors import (
    MultiprocessExecutor,
    SinglethreadExecutor,
)
from maraboupy import Marabou

from .verification import MarabouTaylorStrategy
from .certification_results import CertificationRegion

from .generate_data import generate_grid



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
    onnx_path, dynamics_model, num_workers=8, visualize=True
):
    
    strategy = MarabouTaylorStrategy()

    input_dim = dynamics_model.input_dim
    onnx_input_dim = onnx_input_shape(onnx_path)
    assert len(onnx_input_dim) == 1, f"Only 1D input dims are supported, was {len(onnx_input_dim)}"
    assert onnx_input_dim[0] == input_dim, f"Input dim mismatch: {onnx_input_dim[0]} != {input_dim}"

    network = load_onnx_model(onnx_path)
    partial_process_sample = partial(strategy.verify, network, dynamics_model, epsilon=dynamics_model.epsilon)

    X_train, _ = generate_grid(input_dim, dynamics_model.input_domain, delta=dynamics_model.delta)
    output_dim = dynamics_model.output_dim
    samples = [
        CertificationRegion(x, dynamics_model.delta, j)
        for j in range(output_dim) for x in X_train
    ]

    prepare_strategy = partial(strategy.prepare_strategy, dynamics_model)

    # Initialize plotter if visualization is enabled (supports both 1D and 2D)
    plotter = None
    if visualize and input_dim in [1, 2] and num_workers == 1:
        from .visualization import DynamicsNetworkPlotter  # Import the new plotter
        # Create a plotter for 1D or 2D dynamics
        plotter = DynamicsNetworkPlotter(dynamics_model, Marabou.read_onnx(onnx_path))
        print(f"Initialized visualization for {input_dim}D dynamics")

    if num_workers == 1:
        executor = SinglethreadExecutor()
        # Pass the plotter to the executor
        cex_list = executor.execute(prepare_strategy, partial_process_sample, aggregate, samples, plotter)
    elif num_workers > 1:
        executor = MultiprocessExecutor(num_workers)
        # Note: Visualization is not supported in multiprocessing mode
        cex_list = executor.execute(prepare_strategy, partial_process_sample, aggregate, samples)

    num_cex = len(cex_list) if cex_list else 0

    print(f"Number of counterexamples found: {num_cex}")
    print("Finished")
    
    # Keep the plot window open if we're visualizing
    if plotter is not None:
        pause(30)  # Keep the plot open for 30 seconds
