import torch
from certified_neural_approximations.dynamics import DynamicalSystem
from .translators import TorchTranslator
import numpy as np

class NNDynamics(DynamicalSystem):
    """A class representing the dynamics of a neural network."""

    def __init__(self, network, input_domain):
        super().__init__()
        self.network = network
        self.input_dim = len(input_domain)
        self.output_dim = self.input_dim
        self.input_domain = input_domain
        self.system_name = "NNDynamics"

    @torch.no_grad()
    def compute_dynamics(self, x, translator):
        """
        Compute the dynamics for the neural network.
        
        Args:
            x: The state tensor with shape [input_dim, batch_size]
            
        Returns:
            The derivatives of the system with shape [output_dim, batch_size]
        """
        assert isinstance(translator, TorchTranslator), "NNDynamics only supports TorchTranslator"
        
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        # Forward pass through the neural network
        return self.network(x.T).T