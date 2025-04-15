from juliacall import Main as jl


class JuliaTranslator:
    def matrix_vector(self, a, b):
        """
        Matrix-vector multiplication

        :param a: np.ndarray of floats [n, m]
        :param b: np.ndarray of floats [m]

        :return: np.ndarray of floats [n]
        """
        return jl.seval("(*)")(a, b)

    def sin(self, a):
        """
        Element-wise sine

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return jl.sin(a)

    def cos(self, a):
        """
        Element-wise cosine

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return jl.cos(a)

    def tan(self, a):
        """
        Element-wise tangent

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return jl.tan(a)

    def exp(self, a):
        """
        Element-wise exponential

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return jl.exp(a)

    def log(self, a):
        """
        Element-wise logarithm

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return jl.log(a)

    def sqrt(self, a):
        """
        Element-wise square root

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return jl.sqrt(a)

    def pow(self, a, b):
        """
        Element-wise power

        :param a: np.ndarray of floats
        :param b: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return jl.seval("(^)")(a, b)

    def min(self, a):
        """
        Return the minimum value of a numpy array

        :param a: np.ndarray of floats

        :return: float
        """
        return jl.minimum(a)

    def max(self, a):
        """
        Return the maximum value of a numpy array

        :param a: np.ndarray of floats

        :return: float
        """
        return jl.maximum(a)

    def stack(self, a):
        """
        Stack a list of numpy arrays vertically

        :param a: list of numpy arrays

        :return: np.ndarray
        """
        return jl.reduce(jl.vcat, a)

    def hstack(self, a):
        """
        Stack a list of numpy arrays horizontally

        :param a: list of numpy arrays

        :return: np.ndarray
        """
        return jl.reduce(jl.hcat, a)

    def abs(self, a):
        """
        Element-wise absolute value

        :param a: np.ndarray of floats

        :return: np.ndarray of floats
        """
        return jl.abs(a)
