
from abc import ABC, abstractmethod
import copy
import numpy as np

from maraboupy import Marabou, MarabouCore, MarabouUtils
from grid_neural_abstractions.dynamics import NNDynamics, Quadcopter


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

        options = Marabou.createOptions(verbosity=2)
        res, vals, _ = joint_network.solve(options, verbose=False)

        return res == 'sat'


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
        joint_network.inputVars[0] = [
            np.arange(joint_network.numVars, joint_network.numVars + small_network_inputVars[0].shape[0], dtype=np.int64).view(1, -1)
        ]
        joint_network_inputVars = joint_network.inputVars[0].flatten()
        joint_network.numVars += small_network_inputVars[0].shape[0]

        for i in range(small_network_inputVars[0].shape[0]):
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
        joint_network.outputVars[0] = [
            np.concatenate((large_network_outputVars, small_network_outputVars)).view(1, -1)
        ]


        return joint_network, large_network_outputVars, small_network_outputVars


if __name__ == "__main__":
    dynamics_model = Quadcopter()

    # Example usage
    large_network = Marabou.read_onnx("data/compression_ground_truth.onnx")
    large_network_dynamics = NNDynamics(large_network, dynamics_model.input_domain)
    small_network = Marabou.read_onnx("data/compression_compressed.onnx")

    strategy = MarabouOnlyCompressionVerificationStrategy()
    epsilon = 0.01
    precision = 1e-6

    result = strategy.verify(large_network_dynamics, small_network, epsilon, precision)
    print(result)