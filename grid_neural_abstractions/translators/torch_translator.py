class TorchTranslator:
    def __init__(self, device=None):
        self.device = device
        import torch as torch
        self.torch = torch

    def matrix_vector(self, a, b):
        """
        Matrix-vector multiplication

        :param a: self.torch.tensor of floats [n, m]
        :param b: self.torch.tensor of floats [m]

        :return: self.torch.tensor of floats [n]
        """
        return self.torch.matmul(a, b.unsqueeze(1)).squeeze(1)
    
    def sin(self, a):
        """
        Element-wise sine

        :param a: self.torch.tensor of floats

        :return: self.torch.tensor of floats
        """
        return self.torch.sin(a)
    
    def cos(self, a):
        """
        Element-wise cosine

        :param a: self.torch.tensor of floats

        :return: self.torch.tensor of floats
        """
        return self.torch.cos(a)
    
    def tan(self, a):
        """
        Element-wise tangent

        :param a: self.torch.tensor of floats

        :return: self.torch.tensor of floats
        """
        return self.torch.tan(a)
    
    def exp(self, a):
        """
        Element-wise exponential

        :param a: self.torch.tensor of floats

        :return: self.torch.tensor of floats
        """
        return self.torch.exp(a)
    
    def log(self, a):
        """
        Element-wise logarithm

        :param a: self.torch.tensor of floats

        :return: self.torch.tensor of floats
        """
        return self.torch.log(a)

    def sqrt(self, a):
        """
        Element-wise square root

        :param a: self.torch.tensor of floats

        :return: self.torch.tensor of floats
        """
        return self.torch.sqrt(a)
    
    def cbrt(self, a):
        """
        Element-wise cube root

        :param a: self.torch.tensor of floats

        :return: self.torch.tensor of floats
        """
        return self.torch.pow(a, 1/3)

    def pow(self, a, b):
        """
        Element-wise power

        :param a: self.torch.tensor of floats
        :param b: self.torch.tensor of floats

        :return: self.torch.tensor of floats
        """
        return self.torch.pow(a, b)

    def min(self, a):
        """
        Return the minimum value of a torch tensor

        :param a: self.torch.tensor of floats

        :return: float
        """
        return self.torch.min(a)

    def max(self, a):
        """
        Return the maximum value of a torch tensor

        :param a: self.torch.tensor of floats

        :return: float
        """
        return self.torch.max(a)

    def cat(self, a):
        """
        Stack a list of torch tensors vertically

        :param a: list of torch tensors

        :return: self.torch.tensor
        """
        return self.torch.cat(a)

    def stack(self, a):
        """
        Stack a list of torch tensors vertically

        :param a: list of torch tensors

        :return: self.torch.tensor
        """
        return self.torch.stack(a)

    def hstack(self, a):
        """
        Stack a list of torch tensors horizontally

        :param a: list of torch tensors

        :return: self.torch.tensor
        """
        return self.torch.stack(a, dim=1)

    def to_numpy(self, a):
        """
        Convert a tensor to a numpy array

        :param a: self.torch.tensor

        :return: np.array
        """
        return a.detach().cpu().numpy()

    def to_format(self, a):
        """
        Convert a numpy array to a tensor

        :param a: np.array

        :return: self.torch.tensor
        """
        return self.torch.as_tensor(a, dtype=self.torch.float32, device=self.device)
