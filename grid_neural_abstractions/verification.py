from abc import ABC, abstractmethod
from copy import deepcopy
import time
import types
import numpy as np

from certified_neural_approximations.certification_results import AugmentedSample, \
    SampleResultSAT, SampleResultUNSAT, SampleResultMaybe


def split_sample(data, delta, split_dim):
    split_radius = delta[split_dim] / 2
    sample_left = deepcopy(data)
    sample_left.center[split_dim] -= split_radius
    sample_left.radius[split_dim] = split_radius
    # sample_left.incrementsplitdim()

    sample_right = deepcopy(data)
    sample_right.center[split_dim] += split_radius
    sample_right.radius[split_dim] = split_radius
    # sample_right.incrementsplitdim()

    return sample_left, sample_right


def mean_linear_bound(x, A_lower, b_lower, A_upper, b_upper):
    """
    Evaluate the Taylor approximation at a given point x.

    :param taylor_pol: The Taylor polynomial coefficients.
    :param x: The point at which to evaluate the polynomial.
    :return: The value of the polynomial at x.
    """
    A = (A_upper + A_lower) / 2
    b = (b_upper + b_lower) / 2
    return np.dot(A, x) + b


class VerificationStrategy(ABC):
    @abstractmethod
    def verify_sample(self, data: AugmentedSample):
        """
        Verify the neural network against the dynamics.

        :param network: The neural network to be verified.
        :param dynamics: The dynamics of the system.
        :param data: The region of interest (center and radius).
        :param epsilon: The tolerance for verification.
        :param precision: The precision for numerical checks.
        :return: A result object indicating the verification outcome.
        """
        raise NotImplementedError(
            "This method should be overridden by subclasses."
        )


class MarabouTaylorStrategy(VerificationStrategy):
    def __init__(self, network, dynamics, epsilon, precision=1e-6):
        """
        Initialize the Marabou verification strategy.

        :param network: The neural network to be verified.
        :param dynamics: The dynamics of the system.
        :param epsilon: The tolerance for verification.
        :param precision: The precision for numerical checks.
        """
        self.network = network
        self.dynamics = dynamics
        self.epsilon = epsilon
        self.precision = precision

    def initialize_worker(self):
        """
        Initialize the Marabou worker. This method is called once per worker.
        """
        from maraboupy import Marabou, MarabouCore, MarabouUtils

        global _LOCAL
        _LOCAL = types.SimpleNamespace()
        _LOCAL.network = self.network
        _LOCAL.dynamics = self.dynamics
        _LOCAL.epsilon = self.epsilon
        _LOCAL.precision = self.precision

    @staticmethod
    def verify_sample(data: AugmentedSample):
        return MarabouTaylorStrategy._verify_sample(
            _LOCAL.network,
            _LOCAL.dynamics,
            data,
            _LOCAL.epsilon,
            _LOCAL.precision,
        )

    @staticmethod
    def _verify_sample(network, dynamics, data: AugmentedSample, epsilon, precision):
        outputVars = network.outputVars[0].flatten()
        inputVars = network.inputVars[0].flatten()

        start_time = time.time()

        from maraboupy import Marabou, MarabouCore, MarabouUtils

        # from maraboupy import Marabou

        sample, delta, j = data  # Unpack the data tuple
        (A_lower, b_lower), (A_upper, b_upper), max_gap = data.first_order_model

        # Setup Marabou options using a dynamic timeout based on the size of the region
        # timeout = min(120, max(1, np.log((min(delta)))/np.log(min(data.min_radius))*max_timeout))
        timeout = 2
        options = Marabou.createOptions(verbosity=0, timeoutInSeconds=int(timeout), lpSolver="native")

        # Check if we need to split based on remainder bounds
        if max_gap > epsilon:
            split_dim = data.nextsplitdim(lambda x: mean_linear_bound(x, A_lower, b_lower, A_upper, b_upper), dynamics)
            if delta[split_dim] > data.min_radius[split_dim]:
                sample_left, sample_right = split_sample(data, delta, split_dim)
                return SampleResultMaybe(data, start_time, [sample_left, sample_right])

        # Set the input variables to the sampled point
        for i, inputVar in enumerate(inputVars):
            network.setLowerBound(inputVar, sample[i] - delta[i])
            network.setUpperBound(inputVar, sample[i] + delta[i])

        outputVar = outputVars[j]

        # Reset the query
        network.additionalEquList.clear()

        # x df_c - nn_output >= epsilon + c df_c - f(c) - r_upper
        equation_GE = MarabouUtils.Equation(MarabouCore.Equation.GE)
        for i, inputVar in enumerate(inputVars):
            equation_GE.addAddend(A_upper[i], inputVar)
        equation_GE.addAddend(-1, outputVar)
        equation_GE.setScalar(epsilon - b_upper)
        network.addEquation(equation_GE, isProperty=True)

        # Find a counterexample for upper bound
        res, vals, stats = network.solve(verbose=False, options=options)
        if stats.hasTimedOut():
            split_dim = data.nextsplitdim(lambda x: mean_linear_bound(x, A_lower, b_lower, A_upper, b_upper), dynamics)
            if split_dim is None:
                # Min split radius is reached
                return SampleResultUNSAT(data, start_time, [None])
            else:
                sample_left, sample_right = split_sample(data, delta, split_dim)
                return SampleResultMaybe(data, start_time, [sample_left, sample_right])

        if res == "sat":
            cex = np.empty(len(inputVars))
            for i, inputVar in enumerate(inputVars):
                cex[i] = vals[inputVar]
                assert cex[i] + precision >= sample[i] - delta[i]
                assert cex[i] - precision <= sample[i] + delta[i]

            violation_found = (
                np.dot(A_upper, cex) - vals[outputVar] + precision >= epsilon - b_upper
            )

            assert (
                violation_found
            ), "The counterexample violates the bound, this is not a valid counterexample"

            network.additionalEquList.clear()
            nn_cex = network.evaluateWithMarabou([cex])[0].flatten()
            f_cex = dynamics(cex).flatten()
            if np.abs(nn_cex - f_cex)[j] < epsilon - precision:
                split_dim = data.nextsplitdim(lambda x: mean_linear_bound(x, A_lower, b_lower, A_upper, b_upper), dynamics)
                sample_left, sample_right = split_sample(data, delta, split_dim)
                return SampleResultMaybe(data, start_time, [sample_left, sample_right])
            # else:
            #    print("Counterexample found |N(cex) - f(cex)! > epsilon")

            return SampleResultUNSAT(data, start_time, [cex])

        # Reset the query
        network.additionalEquList.clear()

        # x df_c - nn_output <= -epsilon + c df_c - f(c) - r_lower
        equation_LE = MarabouUtils.Equation(MarabouCore.Equation.LE)
        for i, inputVar in enumerate(inputVars):
            equation_LE.addAddend(A_lower[i], inputVar)
        equation_LE.addAddend(-1, outputVar)
        equation_LE.setScalar(-epsilon - b_lower)
        network.addEquation(equation_LE, isProperty=True)

        # Find a counterexample for lower bound
        res, vals, stats = network.solve(verbose=False, options=options)
        if stats.hasTimedOut():
            split_dim = data.nextsplitdim(lambda x: mean_linear_bound(x, A_lower, b_lower, A_upper, b_upper), dynamics)
            sample_left, sample_right = split_sample(data, delta, split_dim)
            return SampleResultMaybe(data, start_time, [sample_left, sample_right])

        if res == "sat":
            cex = np.empty(len(inputVars))
            for i, inputVar in enumerate(inputVars):
                cex[i] = vals[inputVar]
                assert cex[i] + precision >= sample[i] - delta[i]
                assert cex[i] - precision <= sample[i] + delta[i]
            violation_found = (
                np.dot(A_lower, cex) - vals[outputVar] - precision <= -epsilon - b_lower
            )
            assert (
                violation_found
            ), "The counterexample violates the bound, this is not a valid counterexample"

            network.additionalEquList.clear()
            nn_cex = network.evaluateWithMarabou([cex])[0].flatten()
            f_cex = dynamics(cex).flatten()
            if np.abs(nn_cex - f_cex)[j] < epsilon - precision:
                split_dim = data.nextsplitdim(lambda x: mean_linear_bound(x, A_lower, b_lower, A_upper, b_upper), dynamics)
                if split_dim is None:
                    # Min split radius is reached
                    return SampleResultUNSAT(data, start_time, [None])
                else:
                    sample_left, sample_right = split_sample(data, delta, split_dim)
                    return SampleResultMaybe(data, start_time, [sample_left, sample_right])
            # else:
            #    print("Counterexample found |N(cex) - f(cex)! > epsilon")

            return SampleResultUNSAT(data, start_time, [cex])

        return SampleResultSAT(data, start_time)   # No counterexample found, return the original sample
