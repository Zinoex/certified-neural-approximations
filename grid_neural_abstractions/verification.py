from abc import ABC, abstractmethod
from copy import deepcopy
import torch
import numpy as np

from maraboupy import Marabou, MarabouCore, MarabouUtils

from translators import TorchTranslator, JuliaTranslator, NumpyTranslator
from taylor_expansion import first_order_certified_taylor_expansion


class Region:
    def __init__(self, center: np.array, radius: np.float32):
        self.center = center
        # radius in the sense of a hyperrectangle
        # {x : x[i] = c[i] + \alpha[i] r[i], \alpha \in [-1, 1]^n, i = 1..n}
        self.radius = radius

    def __iter__(self):
        return iter((self.center, self.radius))

    def calculate_size(self):
        """
        Calculate the size of the region (hypercube volume).
        """
        return np.prod(2 * self.radius).item()


def split_sample(data, delta, split_dim):
    split_radius = delta[split_dim] / 2

    sample_left = deepcopy(data)
    sample_left.center[split_dim] -= split_radius
    sample_left.radius[split_dim] = split_radius

    sample_right = deepcopy(data)
    sample_right.center[split_dim] += split_radius
    sample_right.radius[split_dim] = split_radius
    return sample_left, sample_right


class VerificationStrategy(ABC):
    @abstractmethod
    def verify(self, network, dynamics, data: Region, epsilon, precision=1e-6):
        """
        Verify the neural network against the dynamics.

        :param network: The neural network to be verified.
        :param dynamics: The dynamics of the system.
        :param data: The region of interest (center and radius).
        :return: A tuple containing two lists of new regions to verify and counterexamples respectively.
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
        L_step = torch.matmul(L_max, delta)

        if torch.any(L_step > epsilon):
            # consider the largest term of L_step and the delta that affects this, this is the delta we need to reduce.
            split_dim = np.argmax(L_max[np.argmax(L_step), :] * delta)
            sample_left, sample_right = split_sample(data, delta, split_dim)
            return [sample_left, sample_right], [], []

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
                nn_cex = network.evaluateWithMarabou([cex])[0]
                f_cex = dynamics(torch.tensor(cex)).flatten().numpy()
                if np.all(np.abs(nn_cex - f_cex) < epsilon):
                    split_dim = np.argmax(L_max[j, :] * delta)
                    sample_left, sample_right = split_sample(data, delta, split_dim)
                    return [sample_left, sample_right], [], []

                return [], [cex], []

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
                f_cex = dynamics(torch.tensor(cex)).flatten().numpy()
                if np.all(np.abs(nn_cex - f_cex) < epsilon):
                    split_dim = np.argmax(L_max[j, :] * delta)
                    sample_left, sample_right = split_sample(data, delta, split_dim)
                    return [sample_left, sample_right], [], []

                return [], [cex], []

            return [], [], [data]  # No counterexample found, return the original sample


class MarabouTaylorStrategy(VerificationStrategy):
    def verify(self, network, dynamics, data, epsilon, precision=1e-6):
        outputVars = network.outputVars[0].flatten()
        inputVars = network.inputVars[0].flatten()
        options = Marabou.createOptions(verbosity=0)

        sample, delta = data  # Unpack the data tuple
        translator = NumpyTranslator()
        dynamics_value = dynamics(sample, translator)

        # taylor_pol_lower = (a_lower, b_lower, r_lower), 
        # taylor_pol_upper = (a_upper, b_upper, r_upper)
        # where a = f(c), B = Df(c), R = remainder
        taylor_pol_lower, taylor_pol_upper = first_order_certified_taylor_expansion(
            dynamics, sample, delta
        )

        # delta * L 
        L_max = np.abs(taylor_pol_upper[1])
        L_step = np.matmul(L_max, delta)

        if np.any(L_step > epsilon):
            # consider the largest term of L_step and the delta that affects this, this is the delta we need to reduce.
            split_dim = np.argmax(L_max[np.argmax(L_step), :] * delta)
            sample_left, sample_right = split_sample(data, delta, split_dim)
            return [sample_left, sample_right], [], []

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
                nn_cex = network.evaluateWithMarabou([cex])[0]
                f_cex = dynamics(torch.tensor(cex)).flatten().numpy()
                if np.all(np.abs(nn_cex - f_cex) < epsilon):
                    split_dim = np.argmax(L_max[j, :] * delta)
                    sample_left, sample_right = split_sample(data, delta, split_dim)
                    return [sample_left, sample_right], [], []

                return [], [cex], []

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
                f_cex = dynamics(torch.tensor(cex)).flatten().numpy()
                if np.all(np.abs(nn_cex - f_cex) < epsilon):
                    split_dim = np.argmax(L_max[j, :] * delta)
                    sample_left, sample_right = split_sample(data, delta, split_dim)
                    return [sample_left, sample_right], [], []

                return [], [cex], []

            return [], [], [data]  # No counterexample found, return the original sample