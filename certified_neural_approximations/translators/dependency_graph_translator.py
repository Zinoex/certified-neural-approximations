import numpy as np

class Dependency:
    """
    Represents a bipartite graph for input/output dependencies including whether the inputs enter non-linear terms. 
    The latter is an important point as in multi-linear terms each input enters linearly, but the terms are non-linear.
    In that case, the dependency is marked as "nonlin" as it requires a linear relaxation.
    """
    
    def __init__(self, dependencies):
        self.dependencies = dependencies

    @property
    def num_inputs(self):
        return self.dependencies.shape[1]
    
    @property
    def num_outputs(self):
        return self.dependencies.shape[0]

    @staticmethod
    def identity(num_inputs):
        dependencies = np.full((num_inputs, num_inputs), None, dtype=object)
        np.fill_diagonal(dependencies, "lin")

        return Dependency(dependencies)

    def additive_dependency(self, other):
        if isinstance(other, Dependency):
            assert self.dependencies.shape == other.dependencies.shape
            
            dependencies = np.full_like(self.dependencies, None)
            dependencies[np.logical_or(self.dependencies == "lin", other.dependencies == "lin")] = "lin"
            dependencies[np.logical_or(self.dependencies == "nonlin", other.dependencies == "nonlin")] = "nonlin"

            return Dependency(dependencies)
        elif isinstance(other, (int, float, np.ndarray, list)):
            return Dependency(self.dependencies)

    def __add__(self, other):
        return self.additive_dependency(other)

    def __radd__(self, other):
        return self.additive_dependency(other)

    def __sub__(self, other):
        return self.additive_dependency(other)

    def __rsub__(self, other):
        return self.additive_dependency(other)
    
    def __neg__(self):
        return Dependency(self.dependencies)
    
    def multiplication_dependency(self, other):
        if isinstance(other, Dependency):
            assert self.dependencies.shape == other.dependencies.shape

            dependencies = np.full_like(self.dependencies, None)
            dependencies[np.logical_or(
                np.logical_or(self.dependencies == "lin", self.dependencies == "nonlin"),
                np.logical_or(other.dependencies == "lin", other.dependencies == "nonlin")
            )] = "nonlin"

            return Dependency(dependencies)
        elif isinstance(other, (int, float, np.ndarray)):
            return Dependency(self.dependencies)
        else:
            print(f"Unsupported type for multiplication: {type(other)}")
            raise ValueError("Unsupported type for multiplication. Must be Dependency, int, float, or np.ndarray.")

    def __mul__(self, other):
        return self.multiplication_dependency(other)

    def __rmul__(self, other):
        return self.multiplication_dependency(other)

    def __truediv__(self, other):
        return self.multiplication_dependency(other)

    def __rtruediv__(self, other):
        return self.multiplication_dependency(other)

    def __getitem__(self, item):
        return Dependency(self.dependencies[item])
    

class DependencyGraphTranslator:

    def matrix_vector(self, a, b):
        """
        Matrix-vector multiplication

        :param a: np.ndarray of floats [n, m]
        :param b: Dependency

        :return: Dependency
        """
        dependencies = np.full((a.shape[0], b.num_inputs), None, dtype=object)

        for i in range(a.shape[0]):
            for j in range(b.num_inputs):
                res = None
                for k in range(a.shape[1]):
                    if a[i, k] != 0.0:
                        if b.dependencies[k, j] == "nonlin":
                            res = "nonlin"
                            break
                        elif b.dependencies[k, j] == "lin":
                            res = "lin"
                dependencies[i, j] = res

        return Dependency(dependencies)
    
    def nonlin_elementwise_dependency(self, a):
        """
        Create an element-wise dependency for a numpy array

        :param a: Dependency

        :return: Dependency
        """
        dependencies = np.full_like(a.dependencies, None)
        dependencies[np.logical_or(a.dependencies == "lin", a.dependencies == "nonlin")] = "nonlin"

        return Dependency(dependencies)
    
    def sin(self, a):
        """
        Element-wise sine

        :param a: Dependency

        :return: Dependency
        """
        return self.nonlin_elementwise_dependency(a)
    
    def cos(self, a):
        """
        Element-wise cosine

        :param a: Dependency

        :return: Dependency
        """
        return self.nonlin_elementwise_dependency(a)
    
    def tan(self, a):
        """
        Element-wise tangent

        :param a: Dependency

        :return: Dependency
        """
        return self.nonlin_elementwise_dependency(a)
    
    def exp(self, a):
        """
        Element-wise exponential

        :param a: Dependency

        :return: Dependency
        """
        return self.nonlin_elementwise_dependency(a)
    
    def log(self, a):
        """
        Element-wise logarithm

        :param a: Dependency

        :return: Dependency
        """
        return self.nonlin_elementwise_dependency(a)

    def sqrt(self, a):
        """
        Element-wise square root

        :param a: Dependency

        :return: Dependency
        """
        return self.nonlin_elementwise_dependency(a)
    
    def cbrt(self, a):
        """
        Element-wise cube root

        :param a: Dependency

        :return: Dependency
        """
        return self.nonlin_elementwise_dependency(a)

    def pow(self, a, b):
        """
        Element-wise power

        :param a: Dependency
        :param b: np.ndarray or float

        :return: Dependency
        """
        dependencies = np.full_like(a.dependencies, None)

        if isinstance(b, (np.ndarray, list)):
            for i in range(a.num_outputs):
                if b[i] == 1.0 or b[i] == -1.0:
                    dependencies[i, a.dependencies[i] == "lin"] = "lin"
                    dependencies[i, a.dependencies[i] == "nonlin"] = "nonlin"
                elif b[i] == 0.0:
                    dependencies[i, :] = None
                else:
                    dependencies[i, np.logical_or(a.dependencies[i] == "lin", a.dependencies[i] == "nonlin")] = "nonlin"
        elif isinstance(b, (int, float)):
            if b == 1.0 or b == -1.0:
                dependencies[a.dependencies == "lin"] = "lin"
                dependencies[a.dependencies == "nonlin"] = "nonlin"
            elif b == 0.0:
                dependencies[:, :] = None
            else:
                dependencies[np.logical_or(a.dependencies == "lin", a.dependencies == "nonlin")] = "nonlin"
        else:
            raise ValueError("Unsupported type for exponentiation. Must be np.ndarray, list, int, or float.")

        return Dependency(dependencies)

    def stack(self, a):
        """
        Stack a list of numpy arrays vertically

        :param a: list of np.ndarray

        :return: np.ndarray
        """
        dependencies = [dep.dependencies for dep in a]
        dependencies = np.stack(dependencies)

        return Dependency(dependencies)

    def cat(self, a):
        """
        Concatenate a list of numpy arrays along the first axis
        :param a: list of np.ndarray
        :return: np.ndarray
        """
        dependencies = [dep.dependencies for dep in a]
        dependencies = np.concatenate(dependencies)

        return Dependency(dependencies)

    def identity(self, num_inputs):
        """
        Create an identity dependency

        :param num_inputs: int

        :return: Dependency
        """
        return Dependency.identity(num_inputs)