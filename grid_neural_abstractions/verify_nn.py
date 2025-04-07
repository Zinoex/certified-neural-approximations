from functools import partial

import numpy as np
from executors import SinglethreadExecutor
from maraboupy import Marabou, MarabouCore, MarabouUtils
from train_nn import generate_data


def compute_lipschitz_constant(mu, R):
    """
    Compute the Lipschitz constant for the van der Pol oscillator.
    L = max(1 + 2 * mu * R^2, mu * (1 + R^2))
    """
    return max(1 + 2 * mu * R**2, mu * (1 + R**2))


def process_batch(
    onnx_path, delta, L, worker_progress_counters, batch_id, data
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
        List of counterexamples found
    """

    network = Marabou.read_onnx(onnx_path)  # Updated to read ONNX models

    cex_list = []
    outputVars = network.outputVars[0].flatten()
    inputVars = network.inputVars[0].flatten()
    options = Marabou.createOptions(verbosity=0)

    X_train, y_train = data  # Unpack the data tuple

    for idx in range(len(X_train)):
        sample, dynamics_value = X_train[idx], y_train[idx].flatten()

        # Set the input variables to the sampled point
        for i, inputVar in enumerate(inputVars):
            network.setLowerBound(inputVar, sample[i] - delta)
            network.setUpperBound(inputVar, sample[i] + delta)

        # Create disjunctive constraint for all output dimensions
        equation_list = []
        disjuncts = []
        # We need to verify that for all x: |nn_output - f| < delta * L
        # To find a counterexample, we look for x where: |nn_output - f| >= delta * L
        # Which means nn_output - f >= delta * L OR nn_output - f <= -delta * L
        for i, outputVar in enumerate(outputVars):
            # nn_output >= delta * L + f
            equation_GE = MarabouUtils.Equation(MarabouCore.Equation.GE)
            equation_GE.addAddend(1, outputVar)
            equation_GE.setScalar(delta * L + dynamics_value[i])
            equation_list.append([equation_GE])

            # nn_output <= -delta * L + f
            equation_LE = MarabouUtils.Equation(MarabouCore.Equation.LE)
            equation_LE.addAddend(1, outputVar)
            equation_LE.setScalar(-delta * L + dynamics_value[i])
            equation_list.append([equation_LE])

            # For this dimension, either GE or LE must be true to violate the property
            disjuncts.append([equation_GE, equation_LE])
        network.addDisjunctionConstraint(disjuncts)

        # Find a counterexample for lower bound
        res, vals, _ = network.solve(verbose=False, options=options)
        if res == "sat":
            cex = np.empty(len(inputVars))
            for i, inputVar in enumerate(inputVars):
                cex[i] = vals[inputVar]
                assert cex[i] >= sample[i] - delta
                assert cex[i] <= sample[i] + delta
            violation_found = False
            for i, outputVar in enumerate(outputVars):
                if (
                    vals[outputVar] > dynamics_value[i] + delta * L
                    or vals[outputVar] < dynamics_value[i] - delta * L
                ):
                    violation_found = True
                    break
            assert (
                violation_found
            ), "No dimension violates the bound, this is not a valid counterexample"
            cex_list.append(cex)

        # Remove the disjunctive constraint to avoid mutation issues
        network.disjunctionList = []

        worker_progress_counters[batch_id] += 1

    return cex_list


def verify_nn(
    onnx_path, delta=0.01, mu=1.0, R=1.0, model_path="simple_nn.pth"
):
    # Compute Lipschitz constant
    L = compute_lipschitz_constant(mu, R)
    print(f"Computed Lipschitz constant L: {L}")

    input_dim = 2

    # Compute the number of samples for a fixed grid
    range_min, range_max = -1.0, 1.0  # Match the range in generate_data
    num_samples_per_dim = int((range_max - range_min) / delta) + 1
    num_samples = num_samples_per_dim**input_dim
    print(f"Number of samples: {num_samples}")

    local_process_batch = partial(process_batch, onnx_path, delta, L)

    X_train, y_train = generate_data(input_dim, delta=delta, grid=True)
    batch_selector = lambda i, batch_size: (
        X_train[i : i + batch_size],
        y_train[i : i + batch_size],
    )
    num_samples = len(X_train)

    aggregate = lambda agg, x: agg + x

    executor = SinglethreadExecutor()
    sat_counter = executor.execute(
        local_process_batch, batch_selector, num_samples, aggregate
    )

    print(f"Number of counterexamples found: {len(sat_counter)}")
    print("Finished")


if __name__ == "__main__":
    verify_nn(
        "data/simple_nn.onnx",
        delta=0.01,
        mu=1.0,
        R=1.0,
        model_path="data/simple_nn.pth",
    )
