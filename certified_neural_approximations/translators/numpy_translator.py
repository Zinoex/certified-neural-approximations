import numpy as np


class NumpyTranslator:
    def __init__(self):
        pass

    def matrix_vector(self, a, b):
        """
        Matrix-vector multiplication

        :param a: np.ndarray of floats [n, m]
        :param b: np.ndarray of floats [m]

        :return: np.ndarray of floats [n]
        """
        return np.matmul(a, b)
    
    def sin(self, a):
        """
        Element-wise sine

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return np.sin(a)
    
    def cos(self, a):
        """
        Element-wise cosine

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return np.cos(a)
    
    def tan(self, a):
        """
        Element-wise tangent

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return np.tan(a)
    
    def exp(self, a):
        """
        Element-wise exponential

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return np.exp(a)
    
    def log(self, a):
        """
        Element-wise logarithm

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return np.log(a)

    def sqrt(self, a):
        """
        Element-wise square root

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return np.sqrt(a)
    
    def cbrt(self, a):
        """
        Element-wise cube root

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return np.power(a, 1/3)

    def pow(self, a, b):
        """
        Element-wise power

        :param a: np.ndarray of floats
        :param b: np.ndarray or float

        :return: np.ndarray of floats
        """
        return np.power(a, b)

    def min(self, a):
        """
        Return the minimum value of a numpy array

        :param a: np.ndarray of floats

        :return: float
        """
        return np.min(a)

    def max(self, a):
        """
        Return the maximum value of a numpy array

        :param a: np.ndarray of floats

        :return: float
        """
        return np.max(a)

    def stack(self, a):
        """
        Stack a list of numpy arrays vertically

        :param a: list of np.ndarray

        :return: np.ndarray
        """
        return np.stack(a)

    def hstack(self, a):
        """
        Stack a list of numpy arrays horizontally

        :param a: list of np.ndarray

        :return: np.ndarray
        """
        return np.column_stack(a)

    def to_numpy(self, a):
        """
        Convert to a numpy array (no-op since already numpy)

        :param a: np.ndarray

        :return: np.ndarray
        """
        return a

    def to_format(self, a):
        """
        Convert to numpy format (no-op if already numpy, otherwise convert)

        :param a: array-like

        :return: np.ndarray
        """
        return np.asarray(a, dtype=np.float32)
