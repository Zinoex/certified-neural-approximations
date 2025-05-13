from abc import ABC, abstractmethod
from copy import deepcopy
import time
from grid_neural_abstractions.translators.bound_propagation_translator import BoundPropagationTranslator
import numpy as np
from maraboupy import Marabou, MarabouCore, MarabouUtils
from bound_propagation import LinearBounds

from grid_neural_abstractions.taylor_expansion import first_order_certified_taylor_expansion, prepare_taylor_expansion
from grid_neural_abstractions.certification_results import SampleResultSAT, SampleResultUNSAT, SampleResultMaybe, CertificationRegion

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
    def verify(self, network, dynamics, data: CertificationRegion, epsilon, precision=1e-8):
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
    @staticmethod
    def prepare_strategy(dynamics):
        prepare_taylor_expansion(dynamics.input_dim)

    @staticmethod
    def verify(network, dynamics, data: CertificationRegion, epsilon, precision=1e-6, max_timeout=30):
        outputVars = network.outputVars[0].flatten()
        inputVars = network.inputVars[0].flatten()

        # from maraboupy import Marabou

        sample, delta, j = data  # Unpack the data tuple

        # Run the first-order Taylor expansion twice to not count the precompilation time
        # as part of the verification time (the first run is for precompilation).
        # This is a bit of a hack, but it works.
        # try:
        taylor_pol_lower, taylor_pol_upper = first_order_certified_taylor_expansion(
            dynamics, sample, delta
        )

        start_time = time.time()

        taylor_pol_lower, taylor_pol_upper = first_order_certified_taylor_expansion(
            dynamics, sample, delta
        )
        # except Exception as e:
        #     print(f"Error in first_order_certified_taylor_expansion: {e}")
        #     print(f"Sample: {sample}")
        #     return SampleResultMaybe(data, 0, [data])

        # Unpack the Taylor expansion components
        # taylor_pol_lower <-- (f(c), Df(c), R_min)
        # taylor_pol_upper <-- (f(c), Df(c), R_max)
        f_c_lower = taylor_pol_lower[0]  # f(c) - function value at center
        f_c_upper = taylor_pol_upper[0]  # f(c) - function value at center
        df_c_lower = taylor_pol_lower[1]  # Df(c) - gradient at center
        df_c_upper = taylor_pol_upper[1]  # Df(c) - gradient at center
        r_lower = taylor_pol_lower[2]  # Minimum remainder term
        r_upper = taylor_pol_upper[2]  # Maximum remainder term

        A_upper = df_c_upper[j]
        b_upper = -np.dot(df_c_upper[j], sample) + f_c_upper[j] + r_upper[j]

        A_lower = df_c_lower[j]
        b_lower = -np.dot(df_c_lower[j], sample) + f_c_lower[j] + r_lower[j]

        max_gap = r_upper[j] - r_lower[j]

        # Special case if the certified bounds of the Taylor expansion are not finite
        # Then use bound_propagation to get linear bounds
        if (not np.isfinite(A_upper).all()) or (not np.isfinite(b_upper).all()) or \
           (not np.isfinite(A_lower).all()) or (not np.isfinite(b_lower).all()):
            translator = BoundPropagationTranslator()
            x = translator.to_format(sample)
            y = dynamics.compute_dynamics(x, translator)
            linear_bounds = translator.bound(y, sample, delta)

            A_lower = linear_bounds.lower[0]
            b_lower = linear_bounds.lower[1]

            A_upper = linear_bounds.upper[0]
            b_upper = linear_bounds.upper[1]

            A_gap = A_upper - A_lower
            b_gap = b_upper - b_lower
            lbp_gap = LinearBounds(linear_bounds.region, None, (A_gap, b_gap))
            interval_gap = lbp_gap.concretize()  # Turn linear bounds into interval bounds

            # Select the j-th output dimension
            max_gap = interval_gap.upper[0, j].item()

            A_lower = A_lower[j].numpy()
            b_lower = b_lower[j].numpy()

            A_upper = A_upper[j].numpy()
            b_upper = b_upper[j].numpy()

        
        # Setup Marabou options using a dynamic timeout based on the size of the region
        timeout = min(120, max(1, np.log((min(delta)))/np.log(min(data.min_radius))*max_timeout))
        options = Marabou.createOptions(verbosity=0, timeoutInSeconds=int(timeout), lpSolver="native")

        # Check if we need to split based on remainder bounds
        if max_gap > epsilon:
            split_dim = data.nextsplitdim(lambda x: mean_linear_bound(x, A_lower, b_lower, A_upper, b_upper), dynamics)
            if delta[split_dim] > data.min_radius[split_dim]:
                sample_left, sample_right = split_sample(data, delta, split_dim)
                end_time = time.time()
                return SampleResultMaybe(data, end_time - start_time, [sample_left, sample_right])

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
            end_time = time.time()
            if split_dim is None:
                # Min split radius is reached
                return SampleResultUNSAT(data, end_time - start_time, [cex])
            else:
                sample_left, sample_right = split_sample(data, delta, split_dim)
                return SampleResultMaybe(data, end_time - start_time, [sample_left, sample_right])
            
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
                end_time = time.time()
                return SampleResultMaybe(data, end_time - start_time, [sample_left, sample_right])
            #else:
            #    print("Counterexample found |N(cex) - f(cex)! > epsilon")

            end_time = time.time()
            return SampleResultUNSAT(data, end_time - start_time, [cex])

        # Reset the query
        network.additionalEquList.clear()

        # x df_c - nn_output <= -epsilon + c df_c - f(c) - r_lower
        equation_LE = MarabouUtils.Equation(MarabouCore.Equation.LE)
        for i, inputVar in enumerate(inputVars):
            # j is the output dimension, i is the input dimension, thus df_c[j, i] is the partial derivative of the j-th output with respect to the i-th input
            equation_LE.addAddend(A_lower[i], inputVar)
        equation_LE.addAddend(-1, outputVar)
        equation_LE.setScalar(-epsilon - b_lower)
        network.addEquation(equation_LE, isProperty=True)

        # Find a counterexample for lower bound
        res, vals, stats = network.solve(verbose=False, options=options)
        if stats.hasTimedOut():
            split_dim = data.nextsplitdim(lambda x: mean_linear_bound(x, A_lower, b_lower, A_upper, b_upper), dynamics)
            sample_left, sample_right = split_sample(data, delta, split_dim)
            end_time = time.time()
            return SampleResultMaybe(data, end_time - start_time, [sample_left, sample_right])

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
                sample_left, sample_right = split_sample(data, delta, split_dim)
                end_time = time.time()
                return SampleResultMaybe(data, end_time - start_time, [sample_left, sample_right])
            #else:
            #    print("Counterexample found |N(cex) - f(cex)! > epsilon")

            end_time = time.time()
            return SampleResultUNSAT(data, end_time - start_time, [cex])

        end_time = time.time()
        return SampleResultSAT(data, end_time - start_time)   # No counterexample found, return the original sample
