class JuliaTranslator:
    def __init__(self):
        from juliacall import Main as jl
        self.jl = jl

    def matrix_vector(self, a, b):
        """
        Matrix-vector multiplication

        :param a: np.ndarray of floats [n, m]
        :param b: np.ndarray of floats [m]

        :return: np.ndarray of floats [n]
        """
        return self.jl.seval("(*)")(a, b)

    def sin(self, a):
        """
        Element-wise sine

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return self.jl.sin(a)

    def cos(self, a):
        """
        Element-wise cosine

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return self.jl.cos(a)

    def tan(self, a):
        """
        Element-wise tangent

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return self.jl.tan(a)

    def exp(self, a):
        """
        Element-wise exponential

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return self.jl.exp(a)

    def log(self, a):
        """
        Element-wise logarithm

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return self.jl.log(a)

    def sqrt(self, a):
        """
        Element-wise square root

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return self.jl.sqrt(a)
    
    def cbrt(self, a):
        """
        Element-wise cube root

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return self.pow(a, 1/3)

    def pow(self, a, b):
        """
        Element-wise power

        :param a: np.ndarray of floats
        :param b: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return self.jl.seval("(^)")(a, b)

    def min(self, a):
        """
        Return the minimum value of a numpy array

        :param a: np.ndarray of floats

        :return: float
        """
        return self.jl.minimum(a)

    def max(self, a):
        """
        Return the maximum value of a numpy array

        :param a: np.ndarray of floats

        :return: float
        """
        return self.jl.maximum(a)

    def cat(self, a):
        """
        Stack a list of numpy arrays vertically

        :param a: list of numpy arrays

        :return: np.ndarray
        """
        return self.stack(a)

    def stack(self, a):
        """
        Stack a list of numpy arrays vertically

        :param a: list of numpy arrays

        :return: np.ndarray
        """
        return self.jl.reduce(self.jl.vcat, a)

    def hstack(self, a):
        """
        Stack a list of numpy arrays horizontally

        :param a: list of numpy arrays

        :return: np.ndarray
        """
        return self.jl.reduce(self.jl.hcat, a)
