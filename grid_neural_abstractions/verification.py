from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from maraboupy import Marabou, MarabouCore, MarabouUtils

from .taylor_expansion import first_order_certified_taylor_expansion, prepare_taylor_expansion
from .certification_results import SampleResultSAT, SampleResultUNSAT, SampleResultMaybe, Region

def split_sample(data, delta, split_dim):
    split_radius = delta[split_dim] / 2
    sample_left = deepcopy(data)
    sample_left.center[split_dim] -= split_radius
    sample_left.radius[split_dim] = split_radius

    sample_right = deepcopy(data)
    sample_right.center[split_dim] += split_radius
    sample_right.radius[split_dim] = split_radius
    return sample_left, sample_right

def taylor_approximation(x, taylor_pol, c):
    """
    Evaluate the Taylor approximation at a given point x.

    :param taylor_pol: The Taylor polynomial coefficients.
    :param x: The point at which to evaluate the polynomial.
    :return: The value of the polynomial at x.
    """
    # Unpack the Taylor polynomial components
    f_c, df_c, r = taylor_pol
    return f_c + np.dot(df_c, (x-c)) + r


class VerificationStrategy(ABC):
    @abstractmethod
    def verify(self, network, dynamics, data: Region, epsilon, precision=1e-6):
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


class MarabouLipschitzStrategy(VerificationStrategy):
    def verify(self, network, dynamics, data, epsilon, precision=1e-6):
        outputVars = network.outputVars[0].flatten()
        inputVars = network.inputVars[0].flatten()
        options = Marabou.createOptions(verbosity=0)

        sample, delta = data  # Unpack the data tuple
        dynamics_value = dynamics(sample).flatten()

        L_max = dynamics.max_gradient_norm(sample, delta)
        # That we sum over delta comes from the Lagrange remainder term
        # in the 1st order multivariate Taylor expansion.
        # (Bound the higher order derivate + bound the norm in a closed region)
        # https://en.wikipedia.org/wiki/Taylor%27s_theorem#Taylor's_theorem_for_multivariate_functions

        # delta * L 
        L_step = np.matmul(L_max, delta)

        if np.any(L_step > epsilon):
            # consider the largest term of L_step and the delta that affects this, this is the delta we need to reduce.
            split_dim = np.argmax(L_max[np.argmax(L_step), :] * delta)
            sample_left, sample_right = split_sample(data, delta, split_dim)

            return SampleResultMaybe(data, [sample_left, sample_right])

        # Set the input variables to the sampled point
        for i, inputVar in enumerate(inputVars):
            network.setLowerBound(inputVar, sample[i] - delta[i])
            network.setUpperBound(inputVar, sample[i] + delta[i])

        # We need to verify that for all x: |nn_output - f| < delta * L
        # To find a counterexample, we look for x where: |nn_output - f| >= delta * L
        # Which means nn_output - f >= delta * L OR nn_output - f <= delta * L
        for j, outputVar in enumerate(outputVars):
            # Reset the query
            network.additionalEquList.clear()

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
                
                network.additionalEquList.clear()
                nn_cex = network.evaluateWithMarabou([cex])[0].flatten()
                f_cex = dynamics(cex).flatten()
                if np.all(np.abs(nn_cex - f_cex) < epsilon):
                    split_dim = np.argmax(L_max[j, :] * delta)
                    sample_left, sample_right = split_sample(data, delta, split_dim)
                    return SampleResultMaybe(data, [sample_left, sample_right])
                
                return SampleResultUNSAT(data, [cex])

            # Reset the query
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

                network.additionalEquList.clear()
                nn_cex = network.evaluateWithMarabou([cex])[0]
                f_cex = dynamics(cex).flatten()
                if np.all(np.abs(nn_cex - f_cex) < epsilon):
                    split_dim = np.argmax(L_max[j, :] * delta)
                    sample_left, sample_right = split_sample(data, delta, split_dim)
                    return SampleResultMaybe(data, [sample_left, sample_right])

                return SampleResultUNSAT(data, [cex])

            return SampleResultSAT(data)  # No counterexample found, return the original sample


class MarabouTaylorStrategy(VerificationStrategy):
    def __init__(self, dynamics):
        prepare_taylor_expansion(dynamics.input_dim)

    def verify(self, network, dynamics, data, epsilon, precision=1e-6):
        min_delta = 1e-3
        splitting_threshold = 1e-3

        outputVars = network.outputVars[0].flatten()
        inputVars = network.inputVars[0].flatten()
        options = Marabou.createOptions(verbosity=0)

        sample, delta = data  # Unpack the data tuple

        taylor_pol_lower, taylor_pol_upper = first_order_certified_taylor_expansion(
            dynamics, sample, delta
        )

        # Unpack the Taylor expansion components
        # taylor_pol_lower <-- (f(c), Df(c), R_min)
        # taylor_pol_upper <-- (f(c), Df(c), R_max)
        f_c_lower = taylor_pol_lower[0]  # f(c) - function value at center
        f_c_upper = taylor_pol_upper[0]  # f(c) - function value at center
        df_c_lower = taylor_pol_lower[1]  # Df(c) - gradient at center
        df_c_upper= taylor_pol_upper[1]  # Df(c) - gradient at center
        r_lower = taylor_pol_lower[2]  # Minimum remainder term
        r_upper = taylor_pol_upper[2]  # Maximum remainder term

        # Check if we need to split based on remainder bounds
        max_step = np.matmul(np.abs(df_c_lower), delta)
        if np.any(max_step > epsilon):
            # Find the dimension that contributes most to the remainder
            max_output_dim = np.argmax(max_step)
            split_dimensions = np.argsort(-np.abs(df_c_lower[max_output_dim]) * delta)  # Sort in descending order
            split_dim = [
                sd for sd in split_dimensions
                if delta[sd] > min_delta and np.abs(df_c_upper)[max_output_dim, sd] * delta[sd] > splitting_threshold
            ]
            if split_dim:
                split_dim = split_dim[0]
                sample_left, sample_right = split_sample(data, delta, split_dim)
                return SampleResultMaybe(data, [sample_left, sample_right])

        if np.any(r_upper - r_lower > epsilon):
            # Try and see if splitting the input_dimension is helpful
            for dim, _ in enumerate(inputVars):
                delta_tmp = delta.copy()
                delta_tmp[dim] = delta_tmp[dim] / 2 
                taylor_pol_lower, taylor_pol_upper = first_order_certified_taylor_expansion(
                    dynamics, sample, delta_tmp
                )
                if np.any(np.abs(r_upper - r_lower) > np.abs(taylor_pol_lower[2] - taylor_pol_upper[2])):
                    split_dim = dim
                    if delta[split_dim] > min_delta:
                        sample_left, sample_right = split_sample(data, delta, split_dim)
                        return SampleResultMaybe(data, [sample_left, sample_right])    

        # Set the input variables to the sampled point
        for i, inputVar in enumerate(inputVars):
            network.setLowerBound(inputVar, sample[i] - delta[i])
            network.setUpperBound(inputVar, sample[i] + delta[i])

        for j, outputVar in enumerate(outputVars):
            # Reset the query
            network.additionalEquList.clear()

            # x df_c - nn_output >= epsilon + c df_c - f(c) - r_upper
            equation_GE = MarabouUtils.Equation(MarabouCore.Equation.GE)
            for i, inputVar in enumerate(inputVars):
                equation_GE.addAddend(df_c_lower[j,i].item(), inputVar)
            equation_GE.addAddend(-1, outputVar)
            equation_GE.setScalar((epsilon + np.dot(sample, df_c_lower[j]) - f_c_lower[j] - r_upper[j]).item())
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
                    np.dot(cex, df_c_lower[j]) - vals[outputVar] + precision >= epsilon + np.dot(sample, df_c_lower[j]) - f_c_lower[j] - r_upper[j]
                )

                assert (
                    violation_found
                ), "The counterexample violates the bound, this is not a valid counterexample"

                network.additionalEquList.clear()
                nn_cex = network.evaluateWithMarabou([cex])[0]
                f_cex = dynamics(cex).flatten()
                if np.all(np.abs(nn_cex - f_cex) < epsilon):
                    split_dimensions = np.argsort(-np.abs(df_c_lower)[j, :] * delta)
                    split_dim = [
                        sd for sd in split_dimensions
                        if delta[sd] > min_delta and np.abs(df_c_upper)[j, sd] * delta[sd] > splitting_threshold
                    ]
                    if split_dim:
                        split_dim = split_dim[0]
                        sample_left, sample_right = split_sample(data, delta, split_dim)
                        return SampleResultMaybe(data, [sample_left, sample_right])

                return SampleResultUNSAT(data, [cex])

            # Reset the query
            network.additionalEquList.clear()

            # x df_c - nn_output <= -epsilon + c df_c - f(c) - r_lower
            equation_LE = MarabouUtils.Equation(MarabouCore.Equation.LE)            
            for i, inputVar in enumerate(inputVars):
                # j is the output dimension, i is the input dimension, thus df_c[j,i] is the partial derivative of the j-th output with respect to the i-th input
                equation_LE.addAddend(df_c_upper[j,i].item(), inputVar)
            equation_LE.addAddend(-1, outputVar)
            equation_LE.setScalar((-epsilon + np.dot(sample, df_c_upper[j]) - f_c_upper[j] - r_lower[j]).item())
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
                    np.dot(cex, df_c_upper[j]) - vals[outputVar] - precision <= -epsilon + np.dot(sample, df_c_upper[j]) - f_c_upper[j] - r_lower[j]
                )
                assert (
                    violation_found
                ), "The counterexample violates the bound, this is not a valid counterexample"

                network.additionalEquList.clear()
                nn_cex = network.evaluateWithMarabou([cex])[0].flatten()
                f_cex = dynamics(cex).flatten()
                if np.all(np.abs(nn_cex - f_cex) < epsilon):
                    split_dimensions = np.argsort(-np.abs(df_c_upper)[j, :] * delta)
                    split_dim = [
                        sd for sd in split_dimensions
                        if delta[sd] > min_delta and np.abs(df_c_upper)[j, sd] * delta[sd] > splitting_threshold
                    ]
                    if split_dim:
                        split_dim = split_dim[0]
                        sample_left, sample_right = split_sample(data, delta, split_dim)
                        return SampleResultMaybe(data, [sample_left, sample_right])

                return SampleResultUNSAT(data, [cex])

            return SampleResultSAT(data)   # No counterexample found, return the original sample
