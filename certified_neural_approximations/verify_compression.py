
from abc import ABC, abstractmethod
import copy
from functools import partial
import time
import types
from torch.multiprocessing import Pool
from queue import LifoQueue
import numpy as np
import torch
from tqdm import tqdm
from bound_propagation import HyperRectangle, BoundModelFactory, LinearBounds

import onnxruntime
from certified_neural_approximations.executors.stats import Statistics
from certified_neural_approximations.certification_results import CertificationRegion, SampleResultMaybe, SampleResultSAT, SampleResultUNSAT
from certified_neural_approximations.dynamics import NNDynamics
from certified_neural_approximations.train_nn import load_onnx_model, load_torch_model
from certified_neural_approximations.verification import mean_linear_bound, split_sample
from certified_neural_approximations.verify_nn import aggregate
from certified_neural_approximations.generate_data import generate_grid


class CompressionVerificationStrategy(ABC):
    @abstractmethod
    def verify(self, large_network_dynamics, small_network, epsilon, precision=1e-6):
        """
        Verify the compression strategy.
        """
        pass


class MarabouOnlyCompressionVerificationStrategy(CompressionVerificationStrategy):

    def verify(self, large_network_dynamics, small_network, epsilon, precision=1e-6):
        start_time = time.time()

        from maraboupy import Marabou, MarabouCore, MarabouUtils
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
            return SampleResultSAT(large_network_dynamics.input_domain, start_time)
        elif res == 'unsat':
            # This sort of query does not return a counterexample.
            # If we change the query to sat if counterexample found, we can
            # return the counterexample.
            return SampleResultUNSAT(large_network_dynamics.input_domain, start_time, [])

    def merge_networks(self, large_network, small_network):
        from maraboupy import Marabou, MarabouCore, MarabouUtils
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
        joint_network.inputVars[0] = np.arange(joint_network.numVars, joint_network.numVars +
                                               small_network_inputVars.shape[0], dtype=np.int64).reshape((1, -1))
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
    def initialize_pool(onnx_path, small_network_onnx_path):
        # large_network_dynamics.torch_network = load_torch_model(*args, **kwargs)

        onnx_model = load_onnx_model(onnx_path)

        input_names, output_names = onnx_model.inputNames, onnx_model.outputNames
        input_vars = onnx_model.inputVars

        sess = onnxruntime.InferenceSession(onnx_path)

        def evaluate(x):
            input_dict = dict()
            for i, input_name in enumerate(input_names):

                # Try to cast input to correct type
                onnx_type = sess.get_inputs()[i].type
                if 'float' in onnx_type:
                    input_type = 'float32'
                else:
                    raise NotImplementedError("Inputs to network expected to be of type 'float', not %s" % onnx_type)
                input_dict[input_name] = x[i].reshape(input_vars[i].shape).astype(input_type)
            return sess.run(output_names, input_dict)

        global _LOCAL
        _LOCAL = types.SimpleNamespace()
        _LOCAL.onnx_model = evaluate
        _LOCAL.small_network = load_onnx_model(small_network_onnx_path)

    def verify(self, large_network_dynamics, small_network_onnx_path, precision=1e-6, batch_size=20, plotter=None):
        self.pool = Pool(self.num_workers, initializer=self.initialize_pool, initargs=(
            large_network_dynamics.onnx_path, small_network_onnx_path))

        factory = BoundModelFactory()
        self.bound_network = factory.build(large_network_dynamics.network.network)

        # Generate initial set of samples
        delta = np.full(large_network_dynamics.input_dim, large_network_dynamics.delta)
        X_train, _ = generate_grid(large_network_dynamics.input_dim, large_network_dynamics.input_domain, delta=delta)
        output_dim = large_network_dynamics.output_dim
        samples = [
            CertificationRegion(x, delta, j)
            for j in range(output_dim) for x in X_train
        ]

        statistics = Statistics(samples)

        queue = LifoQueue()
        for sample in samples:
            queue.put(sample)

        awaiting_results = []

        agg = None
        start_time = None

        with tqdm(desc="Overall Progress", smoothing=0.1) as pbar:
            while not queue.empty() or awaiting_results:
                if not queue.empty() and len(awaiting_results) <= 10:
                    batch = [
                        queue.get() for _ in range(min(batch_size, queue.qsize()))
                    ]

                    # Execute the batches
                    async_result = self.verify_batch(large_network_dynamics, batch, precision)
                    awaiting_results.append(async_result)

                for async_result in awaiting_results:
                    if async_result.ready():
                        results = async_result.get()
                        awaiting_results.remove(async_result)

                        for result in results:
                            if start_time is None:
                                start_time = result.start_time
                            else:
                                start_time = min(start_time, result.start_time)

                            # Update statistics
                            statistics.add_sample(result)

                            # Update visualization if plotter is provided
                            if result.isleaf() and plotter is not None:
                                plotter.update_figure(result)

                            # Store results however caller wants
                            agg = aggregate(agg, result)

                            # Add new results to the queue
                            if result.hasnewsamples():
                                # Get the new samples
                                new_samples = result.newsamples()

                                # Put the new samples back into the queue
                                for new_sample in new_samples:
                                    queue.put(new_sample)

                        pbar.update(len(results))
                        pbar.set_description_str(
                            f"Overall Progress (remaining samples: {queue.qsize()}, "
                            f"certified: {statistics.get_certified_percentage():.4f}%, "
                            f"uncertified: {statistics.get_uncertified_percentage():.4f}%)"
                        )

        end_time = time.time()
        computation_time = end_time - start_time

        return agg, statistics.get_certified_percentage(), statistics.get_uncertified_percentage(), computation_time

    def verify_batch(self, large_network_dynamics, batch, precision=1e-6):
        linear_bounds = self.taylor_expansion(large_network_dynamics.network, batch,
                                              device=large_network_dynamics.network.device)

        A_gap = linear_bounds.upper[0] - linear_bounds.lower[0]
        b_gap = linear_bounds.upper[1] - linear_bounds.lower[1]
        lbp_gap = LinearBounds(linear_bounds.region, None, (A_gap, b_gap))
        interval_gap = lbp_gap.concretize()  # Turn linear bounds into interval bounds

        linear_bounds = LinearBounds(
            HyperRectangle(linear_bounds.region.lower.cpu().numpy(), linear_bounds.region.upper.cpu().numpy()),
            (linear_bounds.lower[0].cpu().numpy(), linear_bounds.lower[1].cpu().numpy()),
            (linear_bounds.upper[0].cpu().numpy(), linear_bounds.upper[1].cpu().numpy()),
        )

        max_gap = interval_gap.upper.cpu().numpy()

        samples = [(sample, linear_bounds[i], max_gap[i]) for i, sample in enumerate(batch)]

        process_sample = partial(self.verify_sample, large_network_dynamics.epsilon,
                                 large_network_dynamics.input_dim, precision=precision)

        results = self.pool.starmap_async(process_sample, samples)
        # results = [process_sample(sample, linear_bounds, max_gap) for sample, linear_bounds, max_gap in samples]

        return results

    @staticmethod
    @torch.no_grad()
    def verify_sample(epsilon, input_dim, sample: CertificationRegion, linear_bounds, max_gap, precision=1e-6):
        from maraboupy import Marabou, MarabouCore, MarabouUtils
        start_time = time.time()

        global _LOCAL
        onnx_model = _LOCAL.onnx_model
        small_network = _LOCAL.small_network

        inputVars = small_network.inputVars[0].flatten()
        outputVars = small_network.outputVars[0].flatten()
        options = Marabou.createOptions(verbosity=0, timeoutInSeconds=5, lpSolver="native")
        _, delta, j = sample  # Unpack the data tuple
        epsilon = epsilon

        def numpy_model(x):
            y = onnx_model([x])[0].flatten()
            return y

        A_lower = linear_bounds.lower[0][j]
        b_lower = linear_bounds.lower[1][j]
        A_upper = linear_bounds.upper[0][j]
        b_upper = linear_bounds.upper[1][j]

        max_gap = max_gap[j].item()

        mlb = mean_linear_bound(A_lower, b_lower, A_upper, b_upper)

        # Check if we need to split based on remainder bounds
        if max_gap > epsilon:
            # Try and see if splitting the input_dimension is helpful
            split_dim = sample.nextsplitdim(mlb, numpy_model)
            if split_dim is not None:
                sample_left, sample_right = split_sample(sample, delta, split_dim)
                end_time = time.time()
                return SampleResultMaybe(sample, end_time - start_time, [sample_left, sample_right])

        # Add input constraints
        input_dim = input_dim
        region = linear_bounds.region
        for i in range(input_dim):
            small_network.setLowerBound(inputVars[i], region.lower[i].item())
            small_network.setUpperBound(inputVars[i], region.upper[i].item())

        # Add output constraints
        outputVar = outputVars[j]

        # Reset the query
        small_network.additionalEquList.clear()

        # A_upper @ x - nn_output >= epsilon - b_upper
        equation_GE = MarabouUtils.Equation(MarabouCore.Equation.GE)
        for i, inputVar in enumerate(inputVars):
            equation_GE.addAddend(A_upper[i].item(), inputVar)
        equation_GE.addAddend(-1, outputVar)
        equation_GE.setScalar(epsilon - b_upper.item())
        small_network.addEquation(equation_GE, isProperty=True)

        # Find a counterexample for upper bound
        res, vals, stats = small_network.solve(verbose=False, options=options)
        if stats.hasTimedOut():
            split_dim = sample.nextsplitdim(mlb, numpy_model)
            if split_dim is not None:
                sample_left, sample_right = split_sample(sample, delta, split_dim)
                end_time = time.time()
                return SampleResultMaybe(sample, end_time - start_time, [sample_left, sample_right])
            else:
                print("No split dimension found, returning UNSAT")
                end_time = time.time()
                return SampleResultUNSAT(sample, end_time - start_time, [])

        if res == "sat":
            cex = np.empty(len(inputVars))
            for i, inputVar in enumerate(inputVars):
                cex[i] = vals[inputVar]
                assert cex[i] + precision >= region.lower[i].item()
                assert cex[i] - precision <= region.upper[i].item()

            violation_found = (
                np.dot(A_upper, cex) - vals[outputVar] + precision >= epsilon - b_upper.item()
            )

            assert (
                violation_found
            ), "The counterexample violates the bound, this is not a valid counterexample"

            small_network.additionalEquList.clear()
            nn_cex = small_network.evaluateWithMarabou([cex])[0].flatten()
            f_cex = onnx_model([cex])[0].flatten()
            if np.abs(nn_cex - f_cex)[j] < epsilon:
                split_dim = sample.nextsplitdim(mlb, numpy_model)
                sample_left, sample_right = split_sample(sample, sample.radius, split_dim)
                end_time = time.time()
                return SampleResultMaybe(sample, end_time - start_time, [sample_left, sample_right])

            end_time = time.time()
            return SampleResultUNSAT(sample, end_time - start_time, [cex])

        # Reset the query
        small_network.additionalEquList.clear()

        # A_lower @ x - nn_output <= -epsilon - b_lower
        equation_LE = MarabouUtils.Equation(MarabouCore.Equation.LE)
        for i, inputVar in enumerate(inputVars):
            equation_LE.addAddend(A_lower[i].item(), inputVar)
        equation_LE.addAddend(-1, outputVar)
        equation_LE.setScalar(-epsilon - b_lower.item())
        small_network.addEquation(equation_LE, isProperty=True)

        # Find a counterexample for lower bound
        res, vals, stats = small_network.solve(verbose=False, options=options)
        if stats.hasTimedOut():
            split_dim = sample.nextsplitdim(mlb, numpy_model)
            if split_dim is not None:
                sample_left, sample_right = split_sample(sample, delta, split_dim)
                end_time = time.time()
                return SampleResultMaybe(sample, end_time - start_time, [sample_left, sample_right])
            else:
                print("No split dimension found, returning UNSAT")
                end_time = time.time()
                return SampleResultUNSAT(sample, end_time - start_time, [])

        if res == "sat":
            cex = np.empty(len(inputVars))
            for i, inputVar in enumerate(inputVars):
                cex[i] = vals[inputVar]
                assert cex[i] + precision >= region.lower[i].item()
                assert cex[i] - precision <= region.upper[i].item()

            violation_found = (
                np.dot(A_lower, cex) - vals[outputVar] - precision <= -epsilon - b_lower.item()
            )
            assert (
                violation_found
            ), "The counterexample violates the bound, this is not a valid counterexample"

            small_network.additionalEquList.clear()
            nn_cex = small_network.evaluateWithMarabou([cex])[0].flatten()
            f_cex = onnx_model([cex])[0].flatten()
            if np.abs(nn_cex - f_cex)[j] < epsilon:
                split_dim = sample.nextsplitdim(mlb, numpy_model)
                sample_left, sample_right = split_sample(sample, sample.radius, split_dim)
                end_time = time.time()
                return SampleResultMaybe(sample, end_time - start_time, [sample_left, sample_right])

            end_time = time.time()
            return SampleResultUNSAT(sample, end_time - start_time, [cex])

        end_time = time.time()
        return SampleResultSAT(sample, end_time - start_time)   # No counterexample found, return the original sample

    @torch.no_grad()
    def taylor_expansion(self, network, batch, device=None):
        lower = torch.stack([torch.as_tensor(sample.center - sample.radius,
                            dtype=torch.float32, device=device) for sample in batch])
        upper = torch.stack([torch.as_tensor(sample.center + sample.radius,
                            dtype=torch.float32, device=device) for sample in batch])

        input_region = HyperRectangle(lower, upper)
        linear_bounds = self.bound_network.crown(input_region)

        return linear_bounds


def verify_compression(dynamics_model, num_workers=8):
    large_network = load_torch_model("data/compression_ground_truth.pth",
                                     input_size=dynamics_model.input_dim,
                                     hidden_sizes=[1024, 1024, 1024, 1024, 1024],
                                     output_size=dynamics_model.output_dim)
    large_network_dynamics = NNDynamics(large_network, dynamics_model.input_domain)
    large_network_dynamics.onnx_path = "data/compression_ground_truth.onnx"
    large_network_dynamics.delta = dynamics_model.delta
    large_network_dynamics.epsilon = dynamics_model.epsilon
    small_network = "data/compression_compressed.onnx"

    strategy = TaylorMarabouCompressionVerificationStrategy(num_workers=num_workers)
    precision = 1e-6

    cex_list, certified_percentage, uncertified_percentage, computation_time = \
        strategy.verify(large_network_dynamics, small_network, precision)

    num_cex = len(cex_list) if cex_list else 0

    print(f"Number of counterexamples found: {num_cex}")
    print(
        f"Certified percentage: {certified_percentage:.4f}%, "
        f"uncertified percentage: {uncertified_percentage:.4f}%, "
        f"computation time: {computation_time:.2f} seconds"
    )
