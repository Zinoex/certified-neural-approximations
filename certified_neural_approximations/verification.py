from abc import ABC, abstractmethod
from copy import deepcopy
import time
import types
import numpy as np

from certified_neural_approximations.certification_results import SampleResultSAT, SampleResultUNSAT, \
    SampleResultMaybe, CertificationRegion


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


def mean_linear_bound(A_lower, b_lower, A_upper, b_upper):
    """
    Evaluate the Taylor approximation at a given point x.

    :param taylor_pol: The Taylor polynomial coefficients.
    :param x: The point at which to evaluate the polynomial.
    :return: The value of the polynomial at x.
    """
    A = (A_upper + A_lower) / 2
    b = (b_upper + b_lower) / 2

    def mean_linear_bound(x):
        return np.dot(A, x) + b

    return mean_linear_bound


class VerificationStrategy(ABC):
    @abstractmethod
    def verify(self, network, dynamics, data: CertificationRegion, epsilon, precision=1e-6):
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
    def __init__(self, network, dynamics, linearization):
        self.network = network
        self.dynamics = dynamics
        self.linearization = linearization

    def initialize_worker(self):
        global _LOCAL
        _LOCAL = types.SimpleNamespace()
        _LOCAL.network = self.network
        _LOCAL.dynamics = self.dynamics
        _LOCAL.linearization = self.linearization

    @staticmethod
    def verify(data: CertificationRegion, epsilon, precision=1e-6):
        return MarabouTaylorStrategy._verify(
            _LOCAL.network,
            _LOCAL.dynamics,
            _LOCAL.linearization,
            data,
            epsilon,
            precision=precision,
        )

    @staticmethod
    def _verify(network, dynamics, linearization, data: CertificationRegion, epsilon, precision=1e-6):
        outputVars = network.outputVars[0].flatten()
        inputVars = network.inputVars[0].flatten()
        from maraboupy import Marabou, MarabouCore, MarabouUtils
        options = Marabou.createOptions(verbosity=0, timeoutInSeconds=2, lpSolver="native")

        # Run the first-order Taylor expansion twice to not count the precompilation time
        # as part of the verification time (the first run is for precompilation).
        # This is a bit of a hack, but it works.
        linearization.linearize_sample(data)

        start_time = time.time()

        data = linearization.linearize_sample(data)
        sample, delta, j = data  # Unpack the data tuple

        A_lower = data.first_order_model[0][0]
        b_lower = data.first_order_model[0][1]

        A_upper = data.first_order_model[1][0]
        b_upper = data.first_order_model[1][1]

        max_gap = data.first_order_model[2]

        mlb = mean_linear_bound(A_lower, b_lower, A_upper, b_upper)

        # Check if we need to split based on remainder bounds
        if max_gap > epsilon:
            # Try and see if splitting the input_dimension is helpful
            split_dim = data.nextsplitdim(mlb, dynamics)
            if split_dim is not None:
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
            split_dim = data.nextsplitdim(mlb, dynamics)
            if split_dim is not None:
                sample_left, sample_right = split_sample(data, delta, split_dim)
                return SampleResultMaybe(data, start_time, [sample_left, sample_right])
            else:
                print("No split dimension found, returning UNSAT")
                return SampleResultUNSAT(data, start_time, [])

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
                split_dim = data.nextsplitdim(mlb, dynamics)
                if split_dim is not None:
                    sample_left, sample_right = split_sample(data, delta, split_dim)
                    return SampleResultMaybe(data, start_time, [sample_left, sample_right])
                else:
                    print("No split dimension found, returning UNSAT")

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
            split_dim = data.nextsplitdim(mlb, dynamics)
            if split_dim is not None:
                sample_left, sample_right = split_sample(data, delta, split_dim)
                return SampleResultMaybe(data, start_time, [sample_left, sample_right])
            else:
                print("No split dimension found, returning UNSAT")
                return SampleResultUNSAT(data, start_time, [])

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
                split_dim = data.nextsplitdim(mlb, dynamics)
                if split_dim is not None:
                    sample_left, sample_right = split_sample(data, delta, split_dim)
                    return SampleResultMaybe(data, start_time, [sample_left, sample_right])
                else:
                    print("No split dimension found, returning UNSAT")

            return SampleResultUNSAT(data, start_time, [cex])

        return SampleResultSAT(data, start_time)   # No counterexample found, return the original sample
