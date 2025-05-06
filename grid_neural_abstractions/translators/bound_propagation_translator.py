def Constant(a):
    import torch
    import bound_propagation
    return bound_propagation.FixedLinear(
        torch.zeros(a.shape[0], a.shape[0]),
        a,
    )


class WrappedBPOperation:
    def __init__(self, op, x=None):
        from torch import nn
        if isinstance(x, WrappedBPOperation):
            op = nn.Sequential(
                x.op,
                op,
            )

        self.op = op
        import bound_propagation as bp
        self.bp = bp

    def __add__(self, other):
        if isinstance(other, WrappedBPOperation):
            return WrappedBPOperation(
                self.bp.Add(self.op, other.op),
            )
        else:
            import torch
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
                self.bp.Add(self.op, Constant(other))
            )

    def __radd__(self, other):
        self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, WrappedBPOperation):
            return WrappedBPOperation(
                self.bp.Sub(self.op, other.op),
            )
        else:
            import torch
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
                self.bp.Sub(self.op, Constant(other))
            )

    def __rsub__(self, other):
        if isinstance(other, WrappedBPOperation):
            return WrappedBPOperation(
                self.bp.Sub(other.op, self.op),
            )
        else:
            import torch
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
                self.bp.Sub(Constant(other), self.op)
            )


class WrappedBPVector(WrappedBPOperation):
    def __init__(self, length):
        from torch import nn
        super().__init__(nn.Identity())
        self.length = length

    def __getitem__(self, item):
        if isinstance(item, int):
            assert 0 <= item < self.length
        elif isinstance(item, slice):
            assert item.start is not None and item.stop is not None
            assert 0 <= item.start < self.length
            assert 0 <= item.stop <= self.length
        elif isinstance(item, list):
            for i in item:
                assert 0 <= i < self.length
        else:
            raise ValueError("Unsupported type for indexing")

        return WrappedBPOperation(
            self.bp.Select(item)
        )


def bound_propgation_unwrap(x):
    """
    Unwrap a BoundPropagation object to a numpy array
    :param x: BoundPropagation object
    """
    if isinstance(x, WrappedBPOperation):
        return x.op
    else:
        return x


class BoundPropagationTranslator:
    def matrix_vector(self, a, b):
        """
        Matrix-vector multiplication

        :param a: torch.tensor of floats [n, m]
        :param b: torch.tensor of floats [m]

        :return: torch.tensor of floats [n]
        """
        return WrappedBPOperation(self.bp.FixedLinear(a, None), b)

    def sin(self, a):
        """
        Element-wise sine

        :param a: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return WrappedBPOperation(self.bp.Sin(), a)

    def cos(self, a):
        """
        Element-wise cosine

        :param a: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return WrappedBPOperation(self.bp.Cos(), a)
    
    def exp(self, a):
        """
        Element-wise exponential

        :param a: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return WrappedBPOperation(self.bp.Exp(), a)
    
    def log(self, a):
        """
        Element-wise logarithm

        :param a: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return WrappedBPOperation(self.bp.Log(), a)

    def sqrt(self, a):
        """
        Element-wise square root

        :param a: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return WrappedBPOperation(self.bp.Sqrt(), a)

    def cbrt(self, a):
        """
        Element-wise cube root

        :param a: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return WrappedBPOperation(self.bp.Cbrt(), a)

    def pow(self, a, b):
        """
        Element-wise power

        :param a: torch.tensor of floats
        :param b: torch.tensor of floats

        :return: torch.tensor of floats
        """
        return WrappedBPOperation(self.bp.Pow(b), a)  # a^b

    def stack(self, xs):
        """
        Stack a list of torch tensors vertically

        :param a: list of torch tensors

        :return: torch.tensor
        """
        return WrappedBPOperation(self.bp.Parallel(*[bound_propgation_unwrap(x) for x in xs]))

    def to_format(self, x):
        """
        Convert a numpy array to a BoundPropagation object
        :param x: numpy array of floats
        """
        return WrappedBPVector(x.shape[0])

    def bound(self, wrapped_op, x, epsilon):
        """
        Convert a BoundPropagation object to a numpy array
        :param x: BoundPropagation object
        """
        import torch
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)

        if not torch.is_tensor(epsilon):
            epsilon = torch.as_tensor(epsilon, dtype=torch.float32)

        input_bounds = self.bp.HyperRectangle.from_eps(x.unsqueeze(0), epsilon.unsqueeze(0))

        factory = self.bp.BoundModelFactory()
        module = wrapped_op.op
        bound_module = factory.build(module)

        linear_bounds = bound_module.crown(input_bounds)
        linear_bounds = self.bp.LinearBounds(linear_bounds.region, 
                                     (linear_bounds.lower[0][0], linear_bounds.lower[1][0]),
                                     (linear_bounds.upper[0][0], linear_bounds.upper[1][0]))

        return linear_bounds
