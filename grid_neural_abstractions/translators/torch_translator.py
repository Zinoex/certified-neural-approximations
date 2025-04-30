import torch

class TorchTranslator:
    def __init__(self, device=None):
        self.device = device

    def matrix_vector(self, a, b):
        """
        Matrix-vector multiplication

        :param a: torch.tensor of floats [n, m]
        :param b: torch.tensor of floats [m]

        :return: torch.tensor of floats [n]
        """
        return torch.matmul(a, b.unsqueeze(1)).squeeze(1)
    
    def sin(self, a):
        """
        Element-wise sine

        :param a: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return torch.sin(a)
    
    def cos(self, a):
        """
        Element-wise cosine

        :param a: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return torch.cos(a)
    
    def tan(self, a):
        """
        Element-wise tangent

        :param a: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return torch.tan(a)
    
    def exp(self, a):
        """
        Element-wise exponential

        :param a: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return torch.exp(a)
    
    def log(self, a):
        """
        Element-wise logarithm

        :param a: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return torch.log(a)

    def sqrt(self, a):
        """
        Element-wise square root

        :param a: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return torch.sqrt(a)

    def pow(self, a, b):
        """
        Element-wise power

        :param a: torch.tensor of floats
        :param b: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return torch.pow(a, b)

    def min(self, a):
        """
        Return the minimum value of a torch tensor

        :param a: torch.tensor of floats

        :return: float
        """
        return torch.min(a)

    def max(self, a):
        """
        Return the maximum value of a torch tensor

        :param a: torch.tensor of floats

        :return: float
        """
        return torch.max(a)

    def stack(self, a):
        """
        Stack a list of torch tensors vertically

        :param a: list of torch tensors

        :return: torch.tensor
        """
        return torch.stack(a)

    def hstack(self, a):
        """
        Stack a list of torch tensors horizontally

        :param a: list of torch tensors

        :return: torch.tensor
        """
        return torch.stack(a, dim=1)

    def to_numpy(self, a):
        """
        Convert a tensor to a numpy array

        :param a: torch.tensor

        :return: np.array
        """
        return a.detach().cpu().numpy()

    def to_format(self, a):
        """
        Convert a numpy array to a tensor

        :param a: np.array

        :return: torch.tensor
        """
        return torch.as_tensor(a, dtype=torch.float32, device=self.device)
