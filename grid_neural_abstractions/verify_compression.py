
from abc import ABC, abstractmethod
import copy
from functools import partial
import types
from torch.multiprocessing import Pool
from queue import LifoQueue
import numpy as np
import torch
from tqdm import tqdm
from bound_propagation import HyperRectangle, BoundModelFactory

from maraboupy import Marabou, MarabouCore, MarabouUtils
from grid_neural_abstractions.certification_results import CertificationRegion, SampleResultMaybe, SampleResultSAT, SampleResultUNSAT
from grid_neural_abstractions.dynamics import NNDynamics, Quadcopter
from grid_neural_abstractions.train_nn import generate_data, load_onnx_model, load_torch_model
from grid_neural_abstractions.verification import split_sample
from grid_neural_abstractions.verify_nn import aggregate


class CompressionVerificationStrategy(ABC):
    @abstractmethod
    def verify(self, large_network_dynamics, small_network, epsilon, precision=1e-6):
        """
        Verify the compression strategy.
        """
        pass


class MarabouOnlyCompressionVerificationStrategy(CompressionVerificationStrategy):

    def verify(self, large_network_dynamics, small_network, epsilon, precision=1e-6):
        joint_network, outputVarsLarge, outputVarsSmall = self.merge_networks(large_network_dynamics.network, small_network)
        inputVars = joint_network.inputVars[0].flatten()
    
        # Add input constraints
        input_dim = large_network_dynamics.input_dim
        for i in range(input_dim):
            joint_network.setLowerBound(inputVars[i], large_network_dynamics.input_domain[i][0])
            joint_network.setUpperBound(inputVars[i], large_network_dynamics.input_domain[i][1])

        # Add output constraints
        output_dim = large_network_dynamics.output_dim
        for j in range(output_dim):
            equation1 = MarabouUtils.Equation(MarabouCore.Equation.LE)
            equation1.addAddend(1, outputVarsLarge[j])
            equation1.addAddend(-1, outputVarsSmall[j])
            equation1.setScalar(epsilon)
            joint_network.addEquation(equation1, isProperty=True)

            equation1 = MarabouUtils.Equation(MarabouCore.Equation.LE)
            equation1.addAddend(-1, outputVarsLarge[j])
            equation1.addAddend(1, outputVarsSmall[j])
            equation1.setScalar(epsilon)
            joint_network.addEquation(equation1, isProperty=True)

        options = Marabou.createOptions(verbosity=0)
        res, vals, _ = joint_network.solve(options, verbose=False)

        if res == 'sat':
            return SampleResultSAT(large_network_dynamics.input_domain)
        elif res == 'unsat':
            # This sort of query does not return a counterexample.
            # If we change the query to sat if counterexample found, we can
            # return the counterexample.
            return SampleResultUNSAT(large_network_dynamics.input_domain, [])

    def merge_networks(self, large_network, small_network):
        """
        Merge two networks into a single network for verification.
        """
        joint_network = copy.deepcopy(large_network)
        large_network_inputVars = joint_network.inputVars[0].flatten()
        large_network_outputVars = joint_network.outputVars[0].flatten()

        variable_offset = large_network.numVars

        small_network_inputVars = small_network.inputVars[0].flatten() + variable_offset
        small_network_outputVars = small_network.inputVars[0].flatten() + variable_offset

        joint_network.numVars += small_network.numVars

        for i in range(len(small_network.reluList)):
            v1, v2 = small_network.reluList[i]
            joint_network.addRelu(v1 + variable_offset, v2 + variable_offset)

        for i in range(len(small_network.leakyReluList)):
            v1, v2 = small_network.leakyReluList[i]
            joint_network.addLeakyRelu(v1 + variable_offset, v2 + variable_offset)

        for i in range(len(small_network.equList)):
            eq = copy.deepcopy(small_network.equList[i])
            for j in range(len(eq.addendList)):
                c, x = eq.addendList[j]
                eq.addendList[j] = (c, x + variable_offset)
            joint_network.addEquation(eq)

        # TODO: Add other equation types

        # Add new input variables
        joint_network.inputVars[0] = np.arange(joint_network.numVars, joint_network.numVars + small_network_inputVars.shape[0], dtype=np.int64).reshape((1, -1))
        joint_network_inputVars = joint_network.inputVars[0].flatten()
        joint_network.numVars += small_network_inputVars.shape[0]

        for i in range(small_network_inputVars.shape[0]):
            equation_large = MarabouUtils.Equation(MarabouCore.Equation.EQ)
            equation_large.addAddend(1, joint_network_inputVars[i])
            equation_large.addAddend(-1, large_network_inputVars[i])
            equation_large.setScalar(0)
            joint_network.addEquation(equation_large)

            equation_small = MarabouUtils.Equation(MarabouCore.Equation.EQ)
            equation_small.addAddend(1, joint_network_inputVars[i])
            equation_small.addAddend(-1, small_network_inputVars[i])
            equation_small.setScalar(0)
            joint_network.addEquation(equation_small)

        # Give both output variables
        joint_network.outputVars[0] = np.concatenate((large_network_outputVars, small_network_outputVars)).reshape((1, -1))

        return joint_network, large_network_outputVars, small_network_outputVars




class TaylorMarabouCompressionVerificationStrategy(CompressionVerificationStrategy):
    def __init__(self, num_workers=None):
        super().__init__()
        self.num_workers = num_workers
        self.pool = None
        self.bound_network = None

    @staticmethod
    def initialize_pool(*args):
        global _LOCAL
        _LOCAL = types.SimpleNamespace()
        _LOCAL.large_torch_model = load_torch_model(*args)

    def verify(self, large_network_dynamics, small_network, epsilon, delta, precision=1e-6, batch_size=10, plotter=None):
        self.pool = Pool(self.num_workers, initializer=self.initialize_pool, initargs=large_network_dynamics.torch_params)

        factory = BoundModelFactory()
        self.bound_network = factory.build(large_network_dynamics.network.network)

        # Generate initial set of samples
        delta = np.full(large_network_dynamics.input_dim, delta)
        X_train, _ = generate_data(large_network_dynamics.input_dim, dynamics_model.input_domain, delta=delta, grid=True, device="cpu")
        output_dim = dynamics_model.output_dim
        samples = [
            CertificationRegion(x.double().numpy(), delta, j)
            for j in range(output_dim) for x in X_train
        ]

        # Calculate the total domain size
        total_domain_size = sum(sample.lebesguemeasure() for sample in samples)
        certified_domain_size = 0.0
        uncertified_domain_size = 0.0

        queue = LifoQueue()
        for sample in samples:
            queue.put(sample)

        agg = None

        with tqdm(desc="Overall Progress", smoothing=0.1) as pbar:
            while not queue.empty():
                batch = [
                    queue.get() for _ in range(min(batch_size, queue.qsize()))
                ]

                # Execute the batches
                results = self.verify_batch(large_network_dynamics, small_network, batch, epsilon, precision)

                for result in results:
                    if result.issat():
                        # Sample was succesfully verified, no new samples to process
                        # Update certified domain size in a thread-safe manner
                        certified_domain_size += result.lebesguemeasure()
                        # Update visualization if plotter is provided
                        if plotter is not None:
                            plotter.update_figure(result)

                    if result.isunsat():
                        # Sample was not verified, add to the uncertified domain size
                        uncertified_domain_size += result.lebesguemeasure()
                        # Update visualization if plotter is provided
                        if plotter is not None:
                            plotter.update_figure(result)

                    agg = aggregate(agg, result)

                    if result.hasnewsamples():
                        # Get the new samples
                        new_samples = result.newsamples()

                        # Put the new samples back into the queue
                        for new_sample in new_samples:
                            queue.put(new_sample)

                pbar.update(len(results))
                certified_percentage = (certified_domain_size / total_domain_size) * 100
                uncertified_percentage = (uncertified_domain_size / total_domain_size) * 100

                pbar.set_description_str(
                    f"Overall Progress (remaining samples: {queue.qsize()}, certified: {certified_percentage:.2f}%, uncertified: {uncertified_percentage:.2f}%)"
                )

        return agg

    def verify_batch(self, large_network_dynamics, small_network, batch, epsilon, precision=1e-6):
        linear_bounds = self.taylor_expansion(large_network_dynamics.network, batch)

        samples = [(sample, linear_bounds[i]) for i, sample in enumerate(batch)]

        process_sample = partial(self.verify_sample, large_network_dynamics, small_network, epsilon, precision=precision)

        results = self.pool.starmap(process_sample, samples)
        # results = [process_sample(sample, linear_bounds) for sample, linear_bounds in samples]

        return results

    @staticmethod
    @torch.no_grad()
    def verify_sample(large_network_dynamics, small_network, epsilon, sample, linear_bounds, precision=1e-6):
        inputVars = small_network.inputVars[0].flatten()
        outputVars = small_network.outputVars[0].flatten()
        options = Marabou.createOptions(verbosity=0)

        global _LOCAL
        torch_model = _LOCAL.large_torch_model

        # Add input constraints
        input_dim = large_network_dynamics.input_dim
        region = linear_bounds.region
        for i in range(input_dim):
            small_network.setLowerBound(inputVars[i], region.lower[i].item())
            small_network.setUpperBound(inputVars[i], region.upper[i].item())

        # Add output constraints
        output_dim = large_network_dynamics.output_dim
        for j in range(output_dim):
            outputVar = outputVars[j]

            # Reset the query
            small_network.additionalEquList.clear()

            # A_upper @ x - nn_output >= epsilon - b_upper
            equation_GE = MarabouUtils.Equation(MarabouCore.Equation.GE)
            for i, inputVar in enumerate(inputVars):
                equation_GE.addAddend(linear_bounds.upper[0][j, i].item(), inputVar)
            equation_GE.addAddend(-1, outputVar)
            equation_GE.setScalar(epsilon - linear_bounds.upper[1][j].item())
            small_network.addEquation(equation_GE, isProperty=True)

            # Find a counterexample for upper bound
            res, vals, _ = small_network.solve(verbose=False, options=options)
            if res == "sat":
                cex = np.empty(len(inputVars))
                for i, inputVar in enumerate(inputVars):
                    cex[i] = vals[inputVar]
                    assert cex[i] + precision >= region.lower[i].item()
                    assert cex[i] - precision <= region.upper[i].item()

                violation_found = (
                    np.dot(linear_bounds.upper[0][j].numpy(), cex) - vals[outputVar] + precision >= epsilon - linear_bounds.upper[1][j].item()
                )

                assert (
                    violation_found
                ), "The counterexample violates the bound, this is not a valid counterexample"

                small_network.additionalEquList.clear()
                nn_cex = small_network.evaluateWithMarabou([cex])[0].flatten()
                f_cex = torch_model(torch.as_tensor(cex, dtype=torch.float32).view(1, -1)).flatten().numpy()
                if np.abs(nn_cex - f_cex)[j] < epsilon:
                    split_dim = sample.incrementsplitdim()
                    sample_left, sample_right = split_sample(sample, sample.radius, split_dim)
                    return SampleResultMaybe(sample, [sample_left, sample_right])

                return SampleResultUNSAT(sample, [cex])

            # Reset the query
            small_network.additionalEquList.clear()

            # A_lower @ x - nn_output <= -epsilon - b_lower
            equation_LE = MarabouUtils.Equation(MarabouCore.Equation.LE)
            for i, inputVar in enumerate(inputVars):
                equation_LE.addAddend(linear_bounds.lower[0][j, i].item(), inputVar)
            equation_LE.addAddend(-1, outputVar)
            equation_LE.setScalar(-epsilon - linear_bounds.lower[1][j].item())
            small_network.addEquation(equation_LE, isProperty=True)

            # Find a counterexample for lower bound
            res, vals, _ = small_network.solve(verbose=False, options=options)
            if res == "sat":
                cex = np.empty(len(inputVars))
                for i, inputVar in enumerate(inputVars):
                    cex[i] = vals[inputVar]
                    assert cex[i] + precision >= region.lower[i].item()
                    assert cex[i] - precision <= region.upper[i].item()

                violation_found = (
                    np.dot(linear_bounds.lower[0][j].numpy(), cex) - vals[outputVar] - precision <= -epsilon - linear_bounds.lower[1][j].item()
                )
                assert (
                    violation_found
                ), "The counterexample violates the bound, this is not a valid counterexample"

                small_network.additionalEquList.clear()
                nn_cex = small_network.evaluateWithMarabou([cex])[0].flatten()
                f_cex = torch_model(torch.as_tensor(cex, dtype=torch.float32).view(1, -1)).flatten().numpy()
                if np.abs(nn_cex - f_cex)[j] < epsilon:
                    split_dim = sample.incrementsplitdim()
                    sample_left, sample_right = split_sample(sample, sample.radius, split_dim)
                    return SampleResultMaybe(sample, [sample_left, sample_right])

                return SampleResultUNSAT(sample, [cex])

        return SampleResultSAT(sample)   # No counterexample found, return the original sample

    @torch.no_grad()
    def taylor_expansion(self, network, batch):
        lower = torch.stack([torch.as_tensor(sample.center - sample.radius, dtype=torch.float32) for sample in batch])
        upper = torch.stack([torch.as_tensor(sample.center + sample.radius, dtype=torch.float32) for sample in batch])

        input_region = HyperRectangle(lower, upper)
        linear_bounds = self.bound_network.crown(input_region)

        return linear_bounds


if __name__ == "__main__":
    dynamics_model = Quadcopter()

    # Example usage
    large_network = load_torch_model("data/compression_ground_truth.pth",
                                     input_size=dynamics_model.input_dim,
                                     hidden_sizes=[1024, 1024, 1024, 1024, 1024],
                                     output_size=dynamics_model.output_dim)
    large_network_dynamics = NNDynamics(large_network, dynamics_model.input_domain)
    large_network_dynamics.torch_params = ("data/compression_ground_truth.pth", dynamics_model.input_dim, [1024, 1024, 1024, 1024, 1024], dynamics_model.output_dim)
    small_network = load_onnx_model("data/compression_compressed.onnx")

    strategy = TaylorMarabouCompressionVerificationStrategy(num_workers=8)
    epsilon = 0.1
    delta = 0.1
    precision = 1e-6

    result = strategy.verify(large_network_dynamics, small_network, epsilon, delta, precision)
    print(result)
