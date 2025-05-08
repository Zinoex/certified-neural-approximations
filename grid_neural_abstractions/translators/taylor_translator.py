import numpy as np


class CertifiedFirstOrderTaylorExpansion:
    def __init__(self, expansion_point, domain, linear_approximation, remainder):
        self.expansion_point = expansion_point
        self.domain = domain
        self.linear_approximation = linear_approximation
        self.remainder = remainder

    def __add__(self, other):
        if isinstance(other, CertifiedFirstOrderTaylorExpansion):
            assert self.expansion_point == other.expansion_point
            assert self.domain == other.domain

            return CertifiedFirstOrderTaylorExpansion(
                expansion_point=self.expansion_point,
                domain=self.domain,
                linear_approximation=self.linear_approximation + other.linear_approximation,
                remainder=(self.remainder[0] + other.remainder[0], self.remainder[1] + other.remainder[1])
            )
        elif isinstance(other, (int, float)):
            return CertifiedFirstOrderTaylorExpansion(
                expansion_point=self.expansion_point,
                domain=self.domain,
                linear_approximation=(self.linear_approximation[0], self.linear_approximation[1] + other),
                remainder=self.remainder
            )
        else:
            raise ValueError("Unsupported type for addition")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, CertifiedFirstOrderTaylorExpansion):
            assert self.expansion_point == other.expansion_point
            assert self.domain == other.domain

            return CertifiedFirstOrderTaylorExpansion(
                expansion_point=self.expansion_point,
                domain=self.domain,
                linear_approximation=self.linear_approximation - other.linear_approximation,
                remainder=(self.remainder[0] - other.remainder[1], self.remainder[1] - other.remainder[0])
            )
        elif isinstance(other, (int, float)):
            return CertifiedFirstOrderTaylorExpansion(
                expansion_point=self.expansion_point,
                domain=self.domain,
                linear_approximation=(self.linear_approximation[0], self.linear_approximation[1] - other),
                remainder=self.remainder
            )
        else:
            raise ValueError("Unsupported type for subtraction")

    def __rsub__(self, other):
        if isinstance(other, CertifiedFirstOrderTaylorExpansion):
            assert self.expansion_point == other.expansion_point
            assert self.domain == other.domain

            return CertifiedFirstOrderTaylorExpansion(
                expansion_point=self.expansion_point,
                domain=self.domain,
                linear_approximation=other.linear_approximation - self.linear_approximation,
                remainder=(other.remainder[0] - self.remainder[1], other.remainder[1] - self.remainder[0])
            )
        elif isinstance(other, (int, float)):
            return CertifiedFirstOrderTaylorExpansion(
                expansion_point=self.expansion_point,
                domain=self.domain,
                linear_approximation=(self.linear_approximation[0], other - self.linear_approximation[1]),
                remainder=(-self.remainder[1], -self.remainder[0])
            )
        else:
            raise ValueError("Unsupported type for subtraction")
        
    def range(self, x):
        lower = self.linear_approximation[0] @ (self.domain[0] - self.linear_approximation[1]) + self.remainder[0]
        upper = self.linear_approximation[0] @ (self.domain[1] - self.linear_approximation[1]) + self.remainder[1]

        assert np.all(lower <= upper)

        return (lower, upper)


class TaylorTranslator:
    def matrix_vector(self, a, b: CertifiedFirstOrderTaylorExpansion):
        """
        Matrix-vector multiplication

        :param a: np.ndarray of floats [n, m]
        :param b: Taylor model of size [m]

        :return: torch.tensor of floats [n]
        """
        linear_term = a @ b.linear_approximation[0]
        constant_term = a @ b.linear_approximation[1]

        new_linear = (linear_term, constant_term)

        remainder_lower, remainder_upper = b.remainder
        remainder1, remainder2 = a @ remainder_lower, a @ remainder_upper
        new_remainder = np.minimum(remainder1, remainder2), np.maximum(remainder1, remainder2)

        return CertifiedFirstOrderTaylorExpansion(
            b.expansion_point,
            b.domain,
            new_linear,
            new_remainder
        )

    def sin(self, a: CertifiedFirstOrderTaylorExpansion):
        """
        Element-wise sine

        :param a: CertifiedFirstOrderTaylorExpansion

        :return: CertifiedFirstOrderTaylorExpansion
        """

        # Incoming Taylor approximation: y = f(x0) + Df(x0) @ (x - x0) + R
        # Local Taylor expansion of sin(y): sin(y0) + cos(y0) .* (y - y0) + T where y0 = f(x0)
        # Substituting: sin(y0) + cos(y0) .* (y0 + Df(x0) @ (x - x0) + R - y0) + T
        # Rearranging: sin(y0) + cos(y0) .* Df(x0) @ (x - x0) + cos(y0) .* R + T

        y0 = a.linear_approximation[1]
        sin_y0 = np.sin(y0)
        grad_y0 = np.cos(y0)

        linear_term = grad_y0.reshape(grad_y0.shape[0], 1) * a.linear_approximation[0]

        # Use Lagrange bound
        range = a.range()
        # M = max(|-sin(y)|) for y in range
        M = max_abs_sin(range)
        local_remainder = (M / 2) * np.max(np.abs(range[0] - sin_y0), np.abs(range[1] - sin_y0)) ** 2

        remainder1, remainder2 = grad_y0 * a.remainder[0], grad_y0 * a.remainder[1]
        remainder = np.minimum(remainder1, remainder2) - local_remainder, np.maximum(remainder1, remainder2) + local_remainder

        return CertifiedFirstOrderTaylorExpansion(
            a.expansion_point,
            a.domain,
            (linear_term, sin_y0),
            remainder
        )

    def cos(self, a):
        """
        Element-wise cosine

        :param a: CertifiedFirstOrderTaylorExpansion

        :return: CertifiedFirstOrderTaylorExpansion
        """

        # Incoming Taylor approximation: y = f(x0) + Df(x0) @ (x - x0) + R
        # Local Taylor expansion of cos(y): cos(y0) - sin(y0) .* (y - y0) + T where y0 = f(x0)
        # Substituting: cos(y0) - sin(y0) .* (y0 + Df(x0) @ (x - x0) + R - y0) + T
        # Rearranging: cos(y0) - sin(y0) .* Df(x0) @ (x - x0) - sin(y0) .* R + T

        y0 = a.linear_approximation[1]
        cos_y0 = np.cos(y0)
        grad_y0 = -np.sin(y0)

        linear_term = grad_y0.reshape(grad_y0.shape[0], 1) * a.linear_approximation[0]

        # Use Lagrange bound
        range = a.range()
        # M = max(|-cos(y)|) for y in range
        M = max_abs_cos(range)
        local_remainder = (M / 2) * np.max(np.abs(range[0] - cos_y0), np.abs(range[1] - cos_y0)) ** 2

        remainder1, remainder2 = grad_y0 * a.remainder[0], grad_y0 * a.remainder[1]
        remainder = np.minimum(remainder1, remainder2) - local_remainder, np.maximum(remainder1, remainder2) + local_remainder

        return CertifiedFirstOrderTaylorExpansion(
            a.expansion_point,
            a.domain,
            (linear_term, cos_y0),
            remainder
        )

    def exp(self, a):
        """
        Element-wise exponential

        :param a: CertifiedFirstOrderTaylorExpansion

        :return: CertifiedFirstOrderTaylorExpansion
        """
        # Incoming Taylor approximation: y = f(x0) + Df(x0) @ (x - x0) + R
        # Local Taylor expansion of exp(y): exp(y0) + exp(y0) .* (y - y0) + T where y0 = f(x0)
        # Substituting: exp(y0) + exp(y0) .* (y0 + Df(x0) @ (x - x0) + R - y0) + T
        # Rearranging:  exp(y0) + exp(y0) .* Df(x0) @ (x - x0) + exp(y0) .* R + T

        y0 = a.linear_approximation[1]
        exp_y0 = np.exp(y0)

        linear_term = exp_y0.reshape(exp_y0.shape[0], 1) * a.linear_approximation[0]

        # Use monotonicity of exp
        range = a.range()
        local_remainder = (
            0.0,
            np.maximum(
                np.exp(range[0]) - (exp_y0 + exp_y0.reshape(exp_y0.shape[0], 1) * (range[0] - y0)),
                np.exp(range[1]) - (exp_y0 + exp_y0.reshape(exp_y0.shape[0], 1) * (range[1] - y0))
            )
        )

        remainder1, remainder2 = exp_y0 * a.remainder[0], exp_y0 * a.remainder[1]
        remainder = np.minimum(remainder1, remainder2) + local_remainder[0], np.maximum(remainder1, remainder2) + local_remainder[1]

        return CertifiedFirstOrderTaylorExpansion(
            a.expansion_point,
            a.domain,
            (linear_term, exp_y0),
            remainder
        )

    def log(self, a):
        """
        Element-wise logarithm

        :param a: CertifiedFirstOrderTaylorExpansion

        :return: CertifiedFirstOrderTaylorExpansion
        """
        # Incoming Taylor approximation: y = f(x0) + Df(x0) @ (x - x0) + R
        # Local Taylor expansion of log(y): log(y0) + 1/y0 .* (y - y0) + T where y0 = f(x0)
        # Substituting: log(y0) + 1/y0 .* (y0 + Df(x0) @ (x - x0) + R - y0) + T
        # Rearranging:  log(y0) + 1/y0 .* Df(x0) @ (x - x0) + 1/y0 .* R + T

        y0 = a.linear_approximation[1]
        log_y0 = np.log(y0)
        grad_y0 = 1 / y0

        linear_term = grad_y0.reshape(grad_y0.shape[0], 1) * a.linear_approximation[0]

        # Use monotonicity of log
        range = a.range()
        if np.any(range[0] <= 0):
            raise ValueError("Logarithm domain error: range[0] must be greater than 0")

        local_remainder = (
            np.minimum(
                np.log(range[0]) - (log_y0 + grad_y0.reshape(grad_y0.shape[0], 1) * (range[0] - y0)),
                np.log(range[1]) - (log_y0 + grad_y0.reshape(grad_y0.shape[0], 1) * (range[1] - y0))
            ),
            0.0
        )

        remainder1, remainder2 = grad_y0 * a.remainder[0], grad_y0 * a.remainder[1]
        remainder = np.minimum(remainder1, remainder2) + local_remainder[0], np.maximum(remainder1, remainder2) + local_remainder[1]

        return CertifiedFirstOrderTaylorExpansion(
            a.expansion_point,
            a.domain,
            (linear_term, log_y0),
            remainder
        )

    def sqrt(self, a):
        """
        Element-wise square root

        :param a: CertifiedFirstOrderTaylorExpansion

        :return: CertifiedFirstOrderTaylorExpansion
        """
        # Incoming Taylor approximation: y = f(x0) + Df(x0) @ (x - x0) + R
        # Local Taylor expansion of sqrt(y): sqrt(y0) + 1/2*sqrt(y0) .* (y - y0) + T where y0 = f(x0)
        # Substituting: sqrt(y0) + 1/2*sqrt(y0) .* (y0 + Df(x0) @ (x - x0) + R - y0) + T
        # Rearranging:  sqrt(y0) + 1/2*sqrt(y0) .* Df(x0) @ (x - x0) + 1/2*sqrt(y0) .* R + T

        y0 = a.linear_approximation[1]
        sqrt_y0 = np.sqrt(y0)
        grad_y0 = 0.5 / y0

        linear_term = grad_y0.reshape(grad_y0.shape[0], 1) * a.linear_approximation[0]

        # Use monotonicity of sqrt
        range = a.range()
        if np.any(range[0] < 0):
            raise ValueError("Square root domain error: range[0] must be non-negative")

        local_remainder = (
            np.minimum(
                np.sqrt(range[0]) - (sqrt_y0 + grad_y0.reshape(grad_y0.shape[0], 1) * (range[0] - y0)),
                np.sqrt(range[1]) - (sqrt_y0 + grad_y0.reshape(grad_y0.shape[0], 1) * (range[1] - y0))
            ),
            0.0
        )

        remainder1, remainder2 = grad_y0 * a.remainder[0], grad_y0 * a.remainder[1]
        remainder = np.minimum(remainder1, remainder2) + local_remainder[0], np.maximum(remainder1, remainder2) + local_remainder[1]

        return CertifiedFirstOrderTaylorExpansion(
            a.expansion_point,
            a.domain,
            (linear_term, sqrt_y0),
            remainder
        )

    def cbrt(self, a):
        """
        Element-wise cube root

        :param a: CertifiedFirstOrderTaylorExpansion

        :return: CertifiedFirstOrderTaylorExpansion
        """
        # Incoming Taylor approximation: y = f(x0) + Df(x0) @ (x - x0) + R
        # Local Taylor expansion of cbrt(y): cbrt(y0) + (1/3) / (x^(2/3)) .* (y - y0) + T where y0 = f(x0)
        # Substituting: log(y0) + 1/y0 .* (y0 + Df(x0) @ (x - x0) + R - y0) + T
        # Rearranging:  log(y0) + 1/y0 .* Df(x0) @ (x - x0) + 1/y0 .* R + T

        y0 = a.linear_approximation[1]
        cbrt_y0 = np.pow(y0, 1/3)
        grad_y0 = (1/3) * np.pow(y0, -2/3)

        linear_term = grad_y0.reshape(grad_y0.shape[0], 1) * a.linear_approximation[0]

        # Use monotonicity of cbrt in accepted domains (i.e., no crossing zero)
        range = a.range()
        if np.any(range[0] < 0 & range[1] > 0):
            raise ValueError("Cube root domain error: range[0] and range[1] cross zero, resulting in infinite gradients")

        local_remainder = (
            np.minmum(0.0, np.minimum(
                np.cbrt(range[0]) - (cbrt_y0 + grad_y0.reshape(grad_y0.shape[0], 1) * (range[0] - y0)),
                np.cbrt(range[1]) - (cbrt_y0 + grad_y0.reshape(grad_y0.shape[0], 1) * (range[1] - y0))
            )),
            np.maximum(0.0, np.maximum(
                np.cbrt(range[0]) - (cbrt_y0 + grad_y0.reshape(grad_y0.shape[0], 1) * (range[0] - y0)),
                np.cbrt(range[1]) - (cbrt_y0 + grad_y0.reshape(grad_y0.shape[0], 1) * (range[1] - y0))
            ))
        )

        remainder1, remainder2 = grad_y0 * a.remainder[0], grad_y0 * a.remainder[1]
        remainder = np.minimum(remainder1, remainder2) + local_remainder[0], np.maximum(remainder1, remainder2) + local_remainder[1]

        return CertifiedFirstOrderTaylorExpansion(
            a.expansion_point,
            a.domain,
            (linear_term, cbrt_y0),
            remainder
        )

    def pow(self, a, b):
        """
        Element-wise power

        :param a: CertifiedFirstOrderTaylorExpansion
        :param b: CertifiedFirstOrderTaylorExpansion

        :return: CertifiedFirstOrderTaylorExpansion
        """
        assert isinstance(b, int), "Exponent must be an integer"

        # Incoming Taylor approximation: y = f(x0) + Df(x0) @ (x - x0) + R
        # Local Taylor expansion of y^b: y0^b + b * y0^(b-1) .* (y - y0) + T where y0 = f(x0)
        # Substituting: y0^b + b * y0^(b-1) .* (y0 + Df(x0) @ (x - x0) + R - y0) + T
        # Rearranging:  y0^b + b * y0^(b-1) .* Df(x0) @ (x - x0) + b * y0^(b-1) .* R + T

        y0 = a.linear_approximation[1]
        pow_y0 = np.pow(y0, b)
        grad_y0 = b * np.pow(y0, b - 1)

        linear_term = grad_y0.reshape(grad_y0.shape[0], 1) * a.linear_approximation[0]

        # Use Lagrange bound
        range = a.range()
        # M = max(|d^2/dy^2 y^b|) = max(|b*(b-1)*y^(b - 2)) for y in range
        M = max_monomial_vectorized(b * (b - 1), b - 2, range)
        local_remainder = (M / 2) * np.max(np.abs(range[0] - pow_y0), np.abs(range[1] - pow_y0)) ** 2

        remainder1, remainder2 = grad_y0 * a.remainder[0], grad_y0 * a.remainder[1]
        remainder = np.minimum(remainder1, remainder2) + local_remainder[0], np.maximum(remainder1, remainder2) + local_remainder[1]

        return CertifiedFirstOrderTaylorExpansion(
            a.expansion_point,
            a.domain,
            (linear_term, pow_y0),
            remainder
        )

    def stack(self, xs):
        """
        Stack a list of certified taylor expansions vertically

        :param xs: list of certified taylor expansions

        :return: CertifiedFirstOrderTaylorExpansion
        """
        return CertifiedFirstOrderTaylorExpansion(
            xs[0].expansion_point,
            xs[0].domain,
            (np.concatenate([x.linear_approximation[0] for x in xs]), np.concatenate([x.linear_approximation[1] for x in xs])),
            (np.concatenate([x.remainder[0] for x in xs]), np.concatenate([x.remainder[1] for x in xs]))
        )

    def to_format(self, point, lower, upper):
        """
        Initialize the computation of a certified first-order Taylor expansion with 
        the trivial Taylor expansion of f(x) = x, the identity; with a given
        expansion point and domain, and a linear approximation of the identity.
        The remainder is set to zero.

        :param a: torch.tensor of floats

        :return: torch.tensor of floats
        """

        # for f(x) = x, the Taylor expansion is c + (x - c) \oplus R where R = 0
        return CertifiedFirstOrderTaylorExpansion(
            expansion_point=point,
            domain=(lower, upper),
            linear_approximation=(np.eye(point.shape[0]), point),
            remainder=(np.zeros(point.shape[0]), np.zeros(point.shape[0]))
        )


def max_abs_sin(intervals):
    """
    Find the maximum of |sin(x)| for x in the given intervals.

    :param intervals: tuple(np.ndarray, np.ndarray) both of shape (n,).
    :return: np.ndarray of shape (n,) containing the maximum value of |sin(x)| for each interval.
    """
    a, b = intervals

    # Evaluate |sin(x)| at the endpoints
    endpoint_values = np.maximum(np.abs(np.sin(a)), np.abs(np.sin(b)))

    # Find critical points where cos(x) = 0 (x = pi/2 + k*pi)
    critical_points = np.pi / 2 + np.pi * np.arange(-1, 2)  # Covers one period around [a, b]
    critical_points = critical_points[(critical_points >= a[:, None]) & (critical_points <= b[:, None])]

    # Evaluate |sin(x)| at critical points
    critical_values = np.abs(np.sin(critical_points))

    # Combine endpoint and critical values to find the maximum
    max_values = np.maximum.reduce([endpoint_values, critical_values], axis=0)

    return max_values


def max_abs_cos(intervals):
    """
    Find the maximum of |cos(x)| for x in the given intervals.

    :param intervals: tuple(np.ndarray, np.ndarray) both of shape (n,).
    :return: np.ndarray of shape (n,) containing the maximum value of |cos(x)| for each interval.
    """
    a, b = intervals

    # Evaluate |cos(x)| at the endpoints
    endpoint_values = np.maximum(np.abs(np.cos(a)), np.abs(np.cos(b)))

    # Find critical points where sin(x) = 0 (x = k*pi)
    critical_points = np.pi * np.arange(-1, 2)  # Covers one period around [a, b]
    critical_points = critical_points[(critical_points >= a[:, None]) & (critical_points <= b[:, None])]

    # Evaluate |cos(x)| at critical points
    critical_values = np.abs(np.cos(critical_points))

    # Combine endpoint and critical values to find the maximum
    max_values = np.maximum.reduce([endpoint_values, critical_values], axis=0)

    return max_values


def max_monomial_vectorized(c, n, intervals):
    """
    Bound the value of a univariate monomial f(x) = c * x^n over multiple intervals.

    :param c: Coefficient of the monomial (scalar or array-like of shape (m,)).
    :param n: Degree of the monomial (scalar or array-like of shape (m,)).
    :param intervals: tuple(np.ndarray, np.ndarray), both of shape (m,).
    :return: np.ndarray of shape (m,).
    """
    a, b = intervals

    # Evaluate f(x) at the endpoints
    f_a = c * np.power(a, n)
    f_b = c * np.power(b, n)

    # Initialize bounds with endpoint values
    min_values = np.minimum(f_a, f_b)
    max_values = np.maximum(f_a, f_b)

    # Check critical point at x = 0 (only if 0 is in the interval)
    zero_in_interval = (a <= 0) & (b >= 0)
    if np.any(zero_in_interval):
        f_0 = c * np.power(0, n)  # f(0) = 0
        min_values[zero_in_interval] = np.minimum(min_values[zero_in_interval], f_0)
        max_values[zero_in_interval] = np.maximum(max_values[zero_in_interval], f_0)

    # Combine min and max values into a single array
    return np.maximum(np.abs(min_values), np.abs(max_values))
