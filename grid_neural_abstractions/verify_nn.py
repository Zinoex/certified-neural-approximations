from functools import partial
from itertools import product

import numpy as np
import torch
from executors import (
    MultiprocessExecutor,
    MultithreadExecutor,
    SinglethreadExecutor,
)
from maraboupy import Marabou, MarabouCore, MarabouUtils
from train_nn import generate_data
from dynamics import VanDerPolOscillator, Quadcopter
from copy import deepcopy


class Region:
    def __init__(self, center: torch.Tensor, radius: torch.Tensor):
        self.center = center
        # radius in the sense of a hyperrectangle
        # {x : x[i] = c[i] + \alpha[i] r[i], \alpha \in [-1, 1]^n, i = 1..n}
        self.radius = radius

    def __iter__(self):
        return iter((self.center, self.radius))


def process_sample(
    dynamics_model,
    epsilon,
    local,
    data,
    precision=1e-6,
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

    sample, delta = data  # Unpack the data tuple
    dynamics_value = dynamics_model(sample).flatten()

    L_max = dynamics_model.max_gradient_norm(sample, delta)
    # That we sum over delta comes from the Lagrange remainder term
    # in the 1st order multivariate Taylor expansion.
    # (Bound the higher order derivate + bound the norm in a closed region)
    # https://en.wikipedia.org/wiki/Taylor%27s_theorem#Taylor's_theorem_for_multivariate_functions

    # delta * L 
    L_step = torch.matmul(L_max, delta)

    if torch.any(L_step > epsilon):
        # consider the largest term of L_step and the delta that affects this, this is the delta we need to reduce.
        split_dim = np.argmax(L_max[np.argmax(L_step), :] * delta)
        sample_left, sample_right = split_sample(data, delta, split_dim)
        return [sample_left, sample_right], []

    # Set the input variables to the sampled point
    for i, inputVar in enumerate(inputVars):
        network.setLowerBound(inputVar, sample[i] - delta[i])
        network.setUpperBound(inputVar, sample[i] + delta[i])

    # We need to verify that for all x: |nn_output - f| < delta * L
    # To find a counterexample, we look for x where: |nn_output - f| >= delta * L
    # Which means nn_output - f >= delta * L OR nn_output - f <= delta * L
    for j, outputVar in enumerate(outputVars):
        # nn_output >= delta * L + f
        equation_GE = MarabouUtils.Equation(MarabouCore.Equation.GE)
        equation_GE.addAddend(1, outputVar)
        equation_GE.setScalar(dynamics_value[j] + L_step[j].item())
        network.addEquation(equation_GE, isProperty=True)

        # Find a counterexample for lower bound
        res, vals, _ = network.solve(verbose=False, options=options)
        if res == "sat":
            cex = np.empty(len(inputVars))
            for i, inputVar in enumerate(inputVars):
                cex[i] = vals[inputVar]
                assert cex[i] + precision >= sample[i].item() - delta[i]
                assert cex[i] - precision <= sample[i].item() + delta[i]

            violation_found = (
                vals[outputVar] + precision >= dynamics_value[j] + L_step[j].item()
            )

            assert (
                violation_found
            ), "The counterexample violates the bound, this is not a valid counterexample"

            nn_cex = network.evaluateWithoutMarabou([cex])[0]
            f_cex = dynamics_model(torch.tensor(cex)).flatten().numpy()
            if np.all(np.abs(nn_cex - f_cex) < epsilon):
                split_dim = np.argmax(L_max[j, :] * delta)
                sample_left, sample_right = split_sample(data, delta, split_dim)
                return [sample_left, sample_right], []

            return [], [cex]

        # Reset the equation for the other bound
        network.additionalEquList.clear()

        # nn_output <= -delta * L + f
        equation_LE = MarabouUtils.Equation(MarabouCore.Equation.LE)
        equation_LE.addAddend(1, outputVar)
        equation_LE.setScalar(dynamics_value[j] - L_step[j].item())
        network.addEquation(equation_LE, isProperty=True)

        # Find a counterexample for lower bound
        res, vals, _ = network.solve(verbose=False, options=options)
        if res == "sat":
            cex = np.empty(len(inputVars))
            for i, inputVar in enumerate(inputVars):
                cex[i] = vals[inputVar]
                assert cex[i] + precision >= sample[i].item() - delta[i]
                assert cex[i] - precision <= sample[i].item() + delta[i]
            violation_found = (
                vals[outputVar] - precision
                <= dynamics_value[j] - L_step[j].item()
            )
            assert (
                violation_found
            ), "The counterexample violates the bound, this is not a valid counterexample"

            nn_cex = network.evaluateWithoutMarabou([cex])[0]
            f_cex = dynamics_model(torch.tensor(cex)).flatten().numpy()
            if np.all(np.abs(nn_cex - f_cex) < epsilon):
                split_dim = np.argmax(np.abs(nn_cex - f_cex))
                sample_left, sample_right = split_sample(data, delta, split_dim)
                return [sample_left, sample_right], []

            return [], [cex]

        # Reset the equation for the next iteration
        network.additionalEquList.clear()

        return [], []

def split_sample(data, delta, split_dim):
    split_radius = delta[split_dim] / 2
        
    sample_left = deepcopy(data)
    sample_left.center[split_dim] -= split_radius
    sample_left.radius[split_dim] = split_radius

    sample_right = deepcopy(data)
    sample_right.center[split_dim] += split_radius
    sample_right.radius[split_dim] = split_radius
    return sample_left, sample_right


# This function has to be in the global scope to be pickled for multiprocessing
def read_onnx_into_local(onnx_path, local):
    network = Marabou.read_onnx(onnx_path)
    local.network = network


def aggregate(agg, x):
    if agg is None:
        return x
    return agg + x


def verify_nn(
    onnx_path, delta=0.01, epsilon=0.1, num_workers=4
):
    dynamics_model = VanDerPolOscillator()

    input_dim = dynamics_model.input_dim

    # Compute the number of samples for a fixed grid
    range_min, range_max = -1.0, 1.0  # Match the range in generate_data
    num_samples_per_dim = int((range_max - range_min) / delta) + 1
    num_samples = num_samples_per_dim**input_dim
    print(f"Number of initial samples: {num_samples}")

    partial_process_sample = partial(process_sample, dynamics_model, epsilon)

    X_train, _ = generate_data(input_dim, delta=delta, grid=True)
    samples = [
        Region(X_train[i], torch.full_like(X_train[i], delta)) for i in range(num_samples)
    ]

    initializer = partial(read_onnx_into_local, onnx_path)

    executor = MultithreadExecutor()
    cex_list = executor.execute(initializer, partial_process_sample, aggregate, samples)
    num_cex = len(cex_list)

    print(f"Number of counterexamples found: {num_cex}")
    print("Finished")


if __name__ == "__main__":
    verify_nn(
        "data/simple_nn.onnx",
        delta=0.1
    )
