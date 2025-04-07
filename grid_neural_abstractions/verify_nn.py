from functools import partial

import numpy as np
import torch
from executors import (
    MultiprocessExecutor,
    MultithreadExecutor,
    SinglethreadExecutor,
)
import os
from maraboupy import Marabou, MarabouCore, MarabouUtils
from train_nn import generate_data
from dynamics import VanDerPolOscillator, Quadcopter


def compute_lipschitz_constant(mu, R):
    """
    Compute the Lipschitz constant for the van der Pol oscillator.
    L = max(1 + 2 * mu * R^2, mu * (1 + R^2))
    """
    return max(1 + 2 * mu * R**2, mu * (1 + R**2))


def process_sample(
    delta,
    L,
    local,
    data,
    epsilon=1e-6,
    progress_stride=10,
    num_marabou_workers=4,
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
    network = local.network

    outputVars = network.outputVars[0].flatten()
    inputVars = network.inputVars[0].flatten()
    options = Marabou.createOptions(
        verbosity=0, numWorkers=num_marabou_workers
    )

    sample, dynamics_value = data  # Unpack the data tuple
    dynamics_value = dynamics_value.flatten()

    # Set the input variables to the sampled point
    for i, inputVar in enumerate(inputVars):
        network.setLowerBound(inputVar, sample[i] - delta)
        network.setUpperBound(inputVar, sample[i] + delta)

    # We need to verify that for all x: |nn_output - f| < delta * L
    # To find a counterexample, we look for x where: |nn_output - f| >= delta * L
    # Which means nn_output - f >= delta * L OR nn_output - f <= -delta * L
    for j, outputVar in enumerate(outputVars):
        # nn_output >= delta * L + f
        equation_GE = MarabouUtils.Equation(MarabouCore.Equation.GE)
        equation_GE.addAddend(1, outputVar)
        equation_GE.setScalar(dynamics_value[j] + delta * L)
        network.addEquation(equation_GE, isProperty=True)

        # Find a counterexample for lower bound
        res, vals, _ = network.solve(verbose=False, options=options)
        if res == "sat":
            cex = np.empty(len(inputVars))
            for i, inputVar in enumerate(inputVars):
                cex[i] = vals[inputVar]
                assert cex[i] + epsilon >= sample[i].item() - delta
                assert cex[i] - epsilon <= sample[i].item() + delta

            violation_found = (
                vals[outputVar] + epsilon
                >= dynamics_value[j].item() + delta * L
            )
            assert (
                violation_found
            ), "The counterexample violates the bound, this is not a valid counterexample"

            return [cex]

        # Reset the equation for the other bound
        network.additionalEquList.clear()

        # nn_output <= -delta * L + f
        equation_LE = MarabouUtils.Equation(MarabouCore.Equation.LE)
        equation_LE.addAddend(1, outputVar)
        equation_LE.setScalar(dynamics_value[j] - delta * L)
        network.addEquation(equation_LE, isProperty=True)

        # Find a counterexample for lower bound
        res, vals, _ = network.solve(verbose=False, options=options)
        if res == "sat":
            cex = np.empty(len(inputVars))
            for i, inputVar in enumerate(inputVars):
                cex[i] = vals[inputVar]
                assert cex[i] + epsilon >= sample[i].item() - delta
                assert cex[i] - epsilon <= sample[i].item() + delta
            violation_found = (
                vals[outputVar] - epsilon
                <= dynamics_value[j].item() - delta * L
            )
            assert (
                violation_found
            ), "The counterexample violates the bound, this is not a valid counterexample"

            return [cex]

        # Reset the equation for the next iteration
        network.additionalEquList.clear()

    return []


# This function has to be in the global scope to be pickled for multiprocessing
def read_onnx_into_local(onnx_path, local):
    network = Marabou.read_onnx(onnx_path)
    local.network = network


def verify_nn(
    onnx_path, delta=0.01
):
    dynamics_model = VanDerPolOscillator()

    # Compute Lipschitz constant
    L = dynamics_model.compute_lipschitz_constant(delta)
    print(f"Computed Lipschitz constant L: {L}")

    input_dim = dynamics_model.input_dim

    # Compute the number of samples for a fixed grid
    range_min, range_max = -1.0, 1.0  # Match the range in generate_data
    num_samples_per_dim = int((range_max - range_min) / delta) + 1
    num_samples = num_samples_per_dim**input_dim
    print(f"Number of samples: {num_samples}")

    partial_process_sample = partial(process_sample, delta, L)

    X_train, y_train = generate_data(input_dim, delta=delta, grid=True, dynamics_model=dynamics_model)
    num_samples = len(X_train)

    initializer = partial(read_onnx_into_local, onnx_path)

    def select_sample(i):
        return X_train[i], y_train[i]

    def aggregate(agg, x):
        if agg is None:
            return x
        return agg + x

    executor = MultiprocessExecutor(num_workers=12)
    cex_list = executor.execute(initializer, partial_process_sample, select_sample, num_samples, aggregate)
    num_cex = len(cex_list)

    print(f"Number of counterexamples found: {len(num_cex)}")
    print("Finished")


if __name__ == "__main__":
    verify_nn(
        "data/simple_nn.onnx",
        delta=0.01
    )
