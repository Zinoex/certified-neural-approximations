from functools import partial
from grid_neural_abstractions.executors import (
    MultiprocessExecutor,
    SinglethreadExecutor,
)
from grid_neural_abstractions.linearization import default_linearization
from grid_neural_abstractions.verification import MarabouTaylorStrategy
from grid_neural_abstractions.certification_results import CertificationRegion

from grid_neural_abstractions.generate_data import generate_grid
from grid_neural_abstractions.visualization import DynamicsNetworkPlotter  # Import the new plotter


def load_onnx(onnx_path):
    """
    Get the input shape of the ONNX model.
    """

    from maraboupy import Marabou

    network = Marabou.read_onnx(onnx_path)
    return network


def aggregate(agg, result):
    if not result.isunsat():
        return agg

    if agg is None:
        return result.counterexamples()

    return agg + result.counterexamples()


def verify_nn(
    onnx_path, dynamics_model, num_workers=8, visualize=True
):

    input_dim = dynamics_model.input_dim
    network = load_onnx(onnx_path)

    onnx_input_dim = network.inputVars[0].shape[1:]
    assert len(onnx_input_dim) == 1, f"Only 1D input dims are supported, was {len(onnx_input_dim)}"
    assert onnx_input_dim[0] == input_dim, f"Input dim mismatch: {onnx_input_dim[0]} != {input_dim}"

    linearization_strategy = default_linearization(dynamics_model)
    verification_strategy = MarabouTaylorStrategy(network, dynamics_model, epsilon=dynamics_model.epsilon)

    X_train, _ = generate_grid(input_dim, dynamics_model.input_domain, delta=dynamics_model.delta)
    output_dim = dynamics_model.output_dim
    samples = [
        CertificationRegion(x, dynamics_model.delta, j)
        for j in range(output_dim) for x in X_train
    ]

    # Initialize plotter if visualization is enabled (supports both 1D and 2D)
    plotter = None
    if visualize and input_dim in [1, 2] and num_workers == 0:
        from maraboupy import Marabou
        # Create a plotter for 1D or 2D dynamics
        plotter = DynamicsNetworkPlotter(dynamics_model, Marabou.read_onnx(onnx_path))
        print(f"Initialized visualization for {input_dim}D dynamics")

    if num_workers == 0:
        executor = SinglethreadExecutor(linearization_strategy, verification_strategy)
    else:
        executor = MultiprocessExecutor(linearization_strategy, verification_strategy, num_workers=num_workers)

    # Pass the plotter to the executor
    cex_list, certified_percentage, uncertified_percentage, computation_time = executor.execute(aggregate, samples, plotter)

    num_cex = len(cex_list) if cex_list else 0

    print(f"Number of counterexamples found: {num_cex}")
    print(
        f"Certified percentage: {certified_percentage}%, uncertified percentage: {uncertified_percentage}%, computation time: {computation_time:.2f} seconds")
    print("Finished")
