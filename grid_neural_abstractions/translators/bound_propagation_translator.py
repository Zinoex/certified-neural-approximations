from torch import nn
import bound_propagation
import torch


def Constant(a):
    return bound_propagation.FixedLinear(
        torch.zeros(a.shape[0], a.shape[0]),
        a,
    )


class WrappedBPOperation:
    def __init__(self, op, x=None):
        if isinstance(x, WrappedBPOperation):
            op = nn.Sequential(
                x.op,
                op,
            )

        self.op = op

    def __add__(self, other):
        if isinstance(other, WrappedBPOperation):
            return WrappedBPOperation(
                bound_propagation.Add(self.op, other.op),
            )
        elif isinstance(other, WrappedBPVector):
            return WrappedBPOperation(
                bound_propagation.Add(self.op, nn.Identity()),
            )
        else:
            if torch.is_tensor(other):
                assert other.shape == (self.op.output_dim,)
            elif isinstance(other, (int, float)):
                other = torch.full(
                    (self.op.output_dim,),
                    other,
                )
            elif isinstance(other, list):
                other = torch.as_tensor(other, dtype=torch.float32)
            else:
                raise ValueError("Unsupported type for addition")

            return WrappedBPOperation(
                bound_propagation.Add(self.op, Constant(other))
            )

    def __radd__(self, other):
        self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, WrappedBPOperation):
            return WrappedBPOperation(
                bound_propagation.Sub(self.op, other.op),
            )
        elif isinstance(other, WrappedBPVector):
            return WrappedBPOperation(
                bound_propagation.Sub(self.op, nn.Identity()),
            )
        else:
            if torch.is_tensor(other):
                assert other.shape == (self.op.output_dim,)
            elif isinstance(other, (int, float)):
                other = torch.full(
                    (self.op.output_dim,),
                    other,
                )
            elif isinstance(other, list):
                other = torch.as_tensor(other, dtype=torch.float32)
            else:
                raise ValueError("Unsupported type for subtraction")

            return WrappedBPOperation(
                bound_propagation.Sub(self.op, Constant(other))
            )
        
    def __rsub__(self, other):
        if isinstance(other, WrappedBPOperation):
            return WrappedBPOperation(
                bound_propagation.Sub(other.op, self.op),
            )
        elif isinstance(other, WrappedBPVector):
            return WrappedBPOperation(
                bound_propagation.Sub(nn.Identity(), self.op),
            )
        else:
            if torch.is_tensor(other):
                assert other.shape == (self.op.output_dim,)
            elif isinstance(other, (int, float)):
                other = torch.full(
                    (self.op.output_dim,),
                    other,
                )
            elif isinstance(other, list):
                other = torch.as_tensor(other, dtype=torch.float32)
            else:
                raise ValueError("Unsupported type for subtraction")

            return WrappedBPOperation(
                bound_propagation.Sub(Constant(other), self.op)
            )


class WrappedBPVector:
    def __init__(self, indices):
        self.indices = indices

    def __getitem__(self, item):
        if isinstance(item, int):
            item = [item]

        return WrappedBPOperation(
            bound_propagation.Select(self.indices[item])
        )

    def __add__(self, other):
        if isinstance(other, WrappedBPOperation):
            return WrappedBPOperation(
                bound_propagation.Add(nn.Identity(), other.op)
            )
        elif isinstance(other, WrappedBPVector):
            return WrappedBPOperation(
                bound_propagation.Add(nn.Identity(), nn.Identity())
            )
        else:
            if torch.is_tensor(other):
                assert other.shape == (len(self.indices),)
            elif isinstance(other, (int, float)):
                other = torch.full(
                    (len(self.indices),),
                    other,
                )
            elif isinstance(other, list):
                other = torch.as_tensor(other, dtype=torch.float32)
            else:
                raise ValueError("Unsupported type for addition")

            return WrappedBPOperation(
                bound_propagation.Add(nn.Identity(), Constant(other))
            )

    def __radd__(self, other):
        self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, WrappedBPOperation):
            return WrappedBPOperation(
                bound_propagation.Sub(nn.Identity(), other.op)
            )
        elif isinstance(other, WrappedBPVector):
            return WrappedBPOperation(
                bound_propagation.Sub(nn.Identity(), nn.Identity())
            )
        else:
            if torch.is_tensor(other):
                assert other.shape == (len(self.indices),)
            elif isinstance(other, (int, float)):
                other = torch.full(
                    (len(self.indices),),
                    other,
                )
            elif isinstance(other, list):
                other = torch.as_tensor(other, dtype=torch.float32)
            else:
                raise ValueError("Unsupported type for subtraction")

            return WrappedBPOperation(
                bound_propagation.Sub(nn.Identity(), Constant(other))
            )

    def __rsub__(self, other):
        if isinstance(other, WrappedBPOperation):
            return WrappedBPOperation(
                bound_propagation.Sub(other.op, nn.Identity())
            )
        elif isinstance(other, WrappedBPVector):
            return WrappedBPOperation(
                bound_propagation.Sub(nn.Identity(), nn.Identity())
            )
        else:
            if torch.is_tensor(other):
                assert other.shape == (len(self.indices),)
            elif isinstance(other, (int, float)):
                other = torch.full(
                    (len(self.indices),),
                    other,
                )
            elif isinstance(other, list):
                other = torch.as_tensor(other, dtype=torch.float32)
            else:
                raise ValueError("Unsupported type for subtraction")

            return WrappedBPOperation(
                bound_propagation.Sub(Constant(other), nn.Identity())
            )


class BoundPropagationTranslator:
    def __init__(self):
        self.factory = bound_propagation.BoundModelFactory()

    def wrap_vector(self, x):
        """
        Wrap a vector in a BoundPropagation object

        :param x: torch.tensor of floats [n]

        :return: WrappedBPVector
        """
        return WrappedBPVector(list(range(x.shape[0])))

    def matrix_vector(self, a, b):
        """
        Matrix-vector multiplication

        :param a: torch.tensor of floats [n, m]
        :param b: torch.tensor of floats [m]

        :return: torch.tensor of floats [n]
        """
        return WrappedBPOperation(bound_propagation.FixedLinear(a, None), b)

    def sin(self, a):
        """
        Element-wise sine

        :param a: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return WrappedBPOperation(bound_propagation.Sin(), a)

    def cos(self, a):
        """
        Element-wise cosine

        :param a: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return WrappedBPOperation(bound_propagation.Cos(), a)
    
    def exp(self, a):
        """
        Element-wise exponential

        :param a: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return WrappedBPOperation(bound_propagation.Exp(), a)
    
    def log(self, a):
        """
        Element-wise logarithm

        :param a: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return WrappedBPOperation(bound_propagation.Log(), a)

    def sqrt(self, a):
        """
        Element-wise square root

        :param a: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return WrappedBPOperation(bound_propagation.Sqrt(), a)

    def cbrt(self, a):
        """
        Element-wise cube root

        :param a: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return WrappedBPOperation(bound_propagation.Cbrt(), a)

    def pow(self, a, b):
        """
        Element-wise power

        :param a: torch.tensor of floats
        :param b: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return WrappedBPOperation(bound_propagation.Pow(b), a)  # a^b

    def stack(self, a):
        """
        Stack a list of torch tensors vertically

        :param a: list of torch tensors

        :return: torch.tensor
        """
        return WrappedBPOperation(bound_propagation.Parallel(), *a)
