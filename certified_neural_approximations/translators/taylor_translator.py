import numpy as np

# Helper function for TE * TE multiplication remainder calculation


def _mat_interval_vec_mul(M, v_low, v_high):
    M_pos = np.maximum(M, 0)
    M_neg = np.minimum(M, 0)
    res_low = M_pos @ v_low + M_neg @ v_high
    res_high = M_pos @ v_high + M_neg @ v_low
    return res_low, res_high


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

    def __mul__(self, other):
        if isinstance(other, CertifiedFirstOrderTaylorExpansion):
            # TE * TE
            assert np.all(self.expansion_point == other.expansion_point), "Expansion points must match for TE * TE"
            assert np.all(self.domain[0] == other.domain[0]) and np.all(
                self.domain[1] == other.domain[1]), "Domains must match for TE * TE"

            y0_op1, J_op1 = self.linear_approximation
            R_op1_low, R_op1_high = self.remainder

            y0_op2, J_op2 = other.linear_approximation
            R_op2_low, R_op2_high = other.remainder

            y0_op1 = np.atleast_1d(y0_op1)
            y0_op2 = np.atleast_1d(y0_op2)

            new_const = y0_op1 * y0_op2
            new_J = y0_op1.reshape(-1, 1) * J_op2 + y0_op2.reshape(-1, 1) * J_op1

            # Propagated remainder part: y0_op1*R_op2 + y0_op2*R_op1 (interval arithmetic)
            term1_low = np.where(y0_op1 >= 0, y0_op1 * R_op2_low, y0_op1 * R_op2_high)
            term1_high = np.where(y0_op1 >= 0, y0_op1 * R_op2_high, y0_op1 * R_op2_low)

            term2_low = np.where(y0_op2 >= 0, y0_op2 * R_op1_low, y0_op2 * R_op1_high)
            term2_high = np.where(y0_op2 >= 0, y0_op2 * R_op1_high, y0_op2 * R_op1_low)

            propagated_rem_low = term1_low + term2_low
            propagated_rem_high = term1_high + term2_high

            # Higher Order Terms (HOT) from (J1 @ dx) * (J2 @ dx)
            dx_interval_low = self.domain[0] - self.expansion_point
            dx_interval_high = self.domain[1] - self.expansion_point

            u_low, u_high = _mat_interval_vec_mul(J_op1, dx_interval_low, dx_interval_high)
            v_low, v_high = _mat_interval_vec_mul(J_op2, dx_interval_low, dx_interval_high)

            u_low, u_high = np.atleast_1d(u_low), np.atleast_1d(u_high)
            v_low, v_high = np.atleast_1d(v_low), np.atleast_1d(v_high)

            uv_prods = np.array([
                u_low * v_low, u_low * v_high,
                u_high * v_low, u_high * v_high
            ])
            hot_rem_low = np.min(uv_prods, axis=0)
            hot_rem_high = np.max(uv_prods, axis=0)

            final_rem_low = propagated_rem_low + hot_rem_low
            final_rem_high = propagated_rem_high + hot_rem_high

            return CertifiedFirstOrderTaylorExpansion(
                self.expansion_point,
                self.domain,
                (new_J, new_const),
                (final_rem_low, final_rem_high)
            )
        elif isinstance(other, (int, float, np.number)):
            # TE * scalar
            new_df_c = self.linear_approximation[0] * other
            new_f_c = self.linear_approximation[1] * other
            if other >= 0:
                new_remainder = (self.remainder[0] * other, self.remainder[1] * other)
            else:
                new_remainder = (self.remainder[1] * other, self.remainder[0] * other)
            return CertifiedFirstOrderTaylorExpansion(
                expansion_point=self.expansion_point,
                domain=self.domain,
                linear_approximation=(new_df_c, new_f_c),
                remainder=new_remainder
            )
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, np.number)):
            # scalar * TE
            return self.__mul__(other)  # Multiplication is commutative for scalar
        else:
            return NotImplemented

    def _reciprocal(self) -> 'CertifiedFirstOrderTaylorExpansion':
        """
        Computes 1/self for a CertifiedFirstOrderTaylorExpansion.
        """
        y0 = self.linear_approximation[1]  # f(c)
        J_a = self.linear_approximation[0]  # Df(c)
        R_a_lower, R_a_upper = self.remainder

        a_range_lower, a_range_upper = self.range()
        if np.any((a_range_lower <= 1e-9) & (a_range_upper >= -1e-9) & (np.sign(a_range_lower) != np.sign(a_range_upper))):  # Check if zero is in the interval
            # if interval is [0,0] effectively
            if np.any(np.abs(a_range_lower) < 1e-9) and np.any(np.abs(a_range_upper) < 1e-9):
                pass  # allow if it's exactly zero, though 1/0 is problematic. This case should be rare.
            elif np.any((a_range_lower <= 0) & (a_range_upper >= 0)):
                raise ValueError("Reciprocal of a Taylor expansion whose range contains zero is undefined.")

        new_const = 1.0 / y0
        grad_g_y0 = -1.0 / (y0**2)
        new_J = grad_g_y0.reshape(-1, 1) * J_a

        prop_rem_term1 = grad_g_y0 * R_a_lower
        prop_rem_term2 = grad_g_y0 * R_a_upper
        propagated_rem_lower = np.minimum(prop_rem_term1, prop_rem_term2)
        propagated_rem_upper = np.maximum(prop_rem_term1, prop_rem_term2)

        # For g(y) = 1/y, g''(y) = 2/y^3. Lagrange remainder term involves max |g''(eta)/2!| = max |1/eta^3|.
        M_g_double_prime_factor = max_monomial_vectorized(1.0, -3, (a_range_lower, a_range_upper))

        max_abs_y_minus_y0 = np.maximum(np.abs(a_range_lower - y0), np.abs(a_range_upper - y0))
        max_sq_y_minus_y0 = max_abs_y_minus_y0 ** 2

        local_error_magnitude = M_g_double_prime_factor * max_sq_y_minus_y0

        final_rem_lower = propagated_rem_lower - local_error_magnitude
        final_rem_upper = propagated_rem_upper + local_error_magnitude

        return CertifiedFirstOrderTaylorExpansion(
            self.expansion_point,
            self.domain,
            (new_J, new_const),
            (final_rem_lower, final_rem_upper)
        )

    def __truediv__(self, other):
        if isinstance(other, (int, float, np.number)):
            if other == 0:
                raise ZeroDivisionError("Division by zero scalar.")
            return self * (1.0 / other)
        elif isinstance(other, CertifiedFirstOrderTaylorExpansion):
            return self * other._reciprocal()
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float, np.number)):
            return other * self._reciprocal()
        else:
            return NotImplemented

    def __getitem__(self, key):
        if isinstance(key, int):
            idx_slice = slice(key, key + 1)
            new_df_c = self.linear_approximation[0][idx_slice]
            new_f_c = self.linear_approximation[1][idx_slice]
            new_remainder_lower = self.remainder[0][idx_slice]
            new_remainder_upper = self.remainder[1][idx_slice]
            return CertifiedFirstOrderTaylorExpansion(
                expansion_point=self.expansion_point,
                domain=self.domain,
                linear_approximation=(new_df_c, new_f_c),
                remainder=(new_remainder_lower, new_remainder_upper)
            )
        raise TypeError(f"CertifiedFirstOrderTaylorExpansion indices must be integers, not {type(key)}")

    def range(self):
        J = self.linear_approximation[0]
        fc = self.linear_approximation[1]
        c = self.expansion_point
        domain_lower, domain_upper = self.domain
        R_lower, R_upper = self.remainder

        fc = np.atleast_1d(fc)
        c = np.atleast_1d(c)
        R_lower = np.atleast_1d(R_lower)
        R_upper = np.atleast_1d(R_upper)
        domain_lower = np.atleast_1d(domain_lower)
        domain_upper = np.atleast_1d(domain_upper)

        b_affine = fc - (J @ c)
        A_affine = J
        A_affine_pos = np.maximum(A_affine, 0)
        A_affine_neg = np.minimum(A_affine, 0)

        affine_range_lower = (A_affine_pos @ domain_lower) + (A_affine_neg @ domain_upper) + b_affine
        affine_range_upper = (A_affine_pos @ domain_upper) + (A_affine_neg @ domain_lower) + b_affine

        total_range_lower = affine_range_lower + R_lower
        total_range_upper = affine_range_upper + R_upper

        assert np.all(total_range_lower <= total_range_upper + 1e-9), \
            f"Lower bound > upper bound in range calculation: {total_range_lower} vs {total_range_upper}"

        return (total_range_lower, total_range_upper)


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

        y0 = a.linear_approximation[1]  # Center of expansion for y
        sin_y0_val = np.sin(y0)         # sin(y0)
        grad_sin_y0 = np.cos(y0)        # cos(y0), derivative of sin(y) at y0

        linear_term_jacobian = grad_sin_y0.reshape(-1, 1) * a.linear_approximation[0]

        range_of_y = a.range()
        M_lagrange = max_abs_sin(range_of_y)

        max_abs_y_minus_y0 = np.maximum(np.abs(range_of_y[0] - y0), np.abs(range_of_y[1] - y0))
        max_sq_y_minus_y0 = max_abs_y_minus_y0 ** 2

        local_error_magnitude = (M_lagrange / 2) * max_sq_y_minus_y0

        prop_rem_lower_y, prop_rem_upper_y = a.remainder

        term1_rem = grad_sin_y0 * prop_rem_lower_y
        term2_rem = grad_sin_y0 * prop_rem_upper_y

        propagated_taylor_rem_lower = np.minimum(term1_rem, term2_rem)
        propagated_taylor_rem_upper = np.maximum(term1_rem, term2_rem)

        final_rem_lower = propagated_taylor_rem_lower - local_error_magnitude
        final_rem_upper = propagated_taylor_rem_upper + local_error_magnitude

        remainder = (final_rem_lower, final_rem_upper)

        return CertifiedFirstOrderTaylorExpansion(
            a.expansion_point,
            a.domain,
            (linear_term_jacobian, sin_y0_val),
            remainder
        )

    def cos(self, a):
        """
        Element-wise cosine
        :param a: CertifiedFirstOrderTaylorExpansion
        :return: CertifiedFirstOrderTaylorExpansion
        """
        y0 = a.linear_approximation[1]  # Center of expansion for y
        cos_y0_val = np.cos(y0)        # cos(y0)
        grad_cos_y0 = -np.sin(y0)      # -sin(y0), derivative of cos(y) at y0

        linear_term_jacobian = grad_cos_y0.reshape(-1, 1) * a.linear_approximation[0]

        range_of_y = a.range()
        M_lagrange = max_abs_cos(range_of_y)

        max_abs_y_minus_y0 = np.maximum(np.abs(range_of_y[0] - y0), np.abs(range_of_y[1] - y0))
        max_sq_y_minus_y0 = max_abs_y_minus_y0 ** 2

        local_error_magnitude = (M_lagrange / 2) * max_sq_y_minus_y0

        prop_rem_lower_y, prop_rem_upper_y = a.remainder

        term1_rem = grad_cos_y0 * prop_rem_lower_y
        term2_rem = grad_cos_y0 * prop_rem_upper_y

        propagated_taylor_rem_lower = np.minimum(term1_rem, term2_rem)
        propagated_taylor_rem_upper = np.maximum(term1_rem, term2_rem)

        final_rem_lower = propagated_taylor_rem_lower - local_error_magnitude
        final_rem_upper = propagated_taylor_rem_upper + local_error_magnitude

        remainder = (final_rem_lower, final_rem_upper)

        return CertifiedFirstOrderTaylorExpansion(
            a.expansion_point,
            a.domain,
            (linear_term_jacobian, cos_y0_val),
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
        remainder = np.minimum(remainder1, remainder2) + \
            local_remainder[0], np.maximum(remainder1, remainder2) + local_remainder[1]

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
        remainder = np.minimum(remainder1, remainder2) + \
            local_remainder[0], np.maximum(remainder1, remainder2) + local_remainder[1]

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
        remainder = np.minimum(remainder1, remainder2) + \
            local_remainder[0], np.maximum(remainder1, remainder2) + local_remainder[1]

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
        remainder = np.minimum(remainder1, remainder2) + \
            local_remainder[0], np.maximum(remainder1, remainder2) + local_remainder[1]

        return CertifiedFirstOrderTaylorExpansion(
            a.expansion_point,
            a.domain,
            (linear_term, cbrt_y0),
            remainder
        )

    def pow(self, a, exponent_b):
        """
        Element-wise power
        :param a: CertifiedFirstOrderTaylorExpansion
        :param exponent_b: integer exponent
        :return: CertifiedFirstOrderTaylorExpansion
        """
        assert isinstance(exponent_b, int), "Exponent must be an integer"

        y0 = a.linear_approximation[1]
        f_y0 = np.pow(y0, exponent_b)
        grad_f_y0 = exponent_b * np.pow(y0, exponent_b - 1)

        linear_term_jacobian = grad_f_y0.reshape(-1, 1) * a.linear_approximation[0]

        range_of_y = a.range()
        coeff_f_double_prime = exponent_b * (exponent_b - 1)
        exponent_f_double_prime = exponent_b - 2
        M_lagrange = max_monomial_vectorized(coeff_f_double_prime, exponent_f_double_prime, range_of_y)

        max_abs_y_minus_y0 = np.maximum(np.abs(range_of_y[0] - y0), np.abs(range_of_y[1] - y0))
        max_sq_y_minus_y0 = max_abs_y_minus_y0 ** 2

        local_error_magnitude = (M_lagrange / 2) * max_sq_y_minus_y0

        prop_rem_lower_y, prop_rem_upper_y = a.remainder

        term1_rem = grad_f_y0 * prop_rem_lower_y
        term2_rem = grad_f_y0 * prop_rem_upper_y

        propagated_taylor_rem_lower = np.minimum(term1_rem, term2_rem)
        propagated_taylor_rem_upper = np.maximum(term1_rem, term2_rem)

        final_rem_lower = propagated_taylor_rem_lower - local_error_magnitude
        final_rem_upper = propagated_taylor_rem_upper + local_error_magnitude

        remainder = (final_rem_lower, final_rem_upper)

        return CertifiedFirstOrderTaylorExpansion(
            a.expansion_point,
            a.domain,
            (linear_term_jacobian, f_y0),
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
            linear_approximation=(np.ones(point.shape[0]), point),
            remainder=(np.zeros(point.shape[0]), np.zeros(point.shape[0]))
        )


def max_abs_sin(intervals):
    """
    Find the maximum of |sin(x)| for x in the given intervals.
    :param intervals: tuple(np.ndarray, np.ndarray) both of shape (n,).
                      intervals[0] is the array of lower bounds (a_i).
                      intervals[1] is the array of upper bounds (b_i).
    :return: np.ndarray of shape (n,) containing the maximum value of |sin(x)| for each interval.
    """
    a, b = intervals

    # Calculate |sin(x)| at the endpoints
    abs_sin_a = np.abs(np.sin(a))
    abs_sin_b = np.abs(np.sin(b))
    max_vals = np.maximum(abs_sin_a, abs_sin_b)

    # Check if any point x = k*pi + pi/2 (where sin(x) = +/-1) is within [a, b]
    # This is true if there is an integer k such that:
    # a <= k*pi + pi/2 <= b  =>  (a - pi/2)/pi <= k <= (b - pi/2)/pi
    # So, we check if floor((b - pi/2)/pi) >= ceil((a - pi/2)/pi)

    # Lower bound for k (must be an integer, so use ceil)
    k_lower_bound = np.ceil((a - np.pi/2) / np.pi)
    # Upper bound for k (must be an integer, so use floor)
    k_upper_bound = np.floor((b - np.pi/2) / np.pi)

    # If k_lower_bound <= k_upper_bound, then there is an integer k in the range,
    # meaning a peak of |sin(x)| = 1 is included in the interval.
    contains_peak = k_lower_bound <= k_upper_bound
    max_vals[contains_peak] = 1.0

    return max_vals


def max_abs_cos(intervals):
    """
    Find the maximum of |cos(x)| for x in the given intervals.
    :param intervals: tuple(np.ndarray, np.ndarray) both of shape (n,).
                      intervals[0] is the array of lower bounds (a_i).
                      intervals[1] is the array of upper bounds (b_i).
    :return: np.ndarray of shape (n,) containing the maximum value of |cos(x)| for each interval.
    """
    a, b = intervals

    # Calculate |cos(x)| at the endpoints
    abs_cos_a = np.abs(np.cos(a))
    abs_cos_b = np.abs(np.cos(b))
    max_vals = np.maximum(abs_cos_a, abs_cos_b)

    # Check if any point x = k*pi (where cos(x) = +/-1) is within [a, b]
    # This is true if there is an integer k such that:
    # a <= k*pi <= b  =>  a/pi <= k <= b/pi
    # So, we check if floor(b/pi) >= ceil(a/pi)

    # Lower bound for k (must be an integer, so use ceil)
    k_lower_bound = np.ceil(a / np.pi)
    # Upper bound for k (must be an integer, so use floor)
    k_upper_bound = np.floor(b / np.pi)

    # If k_lower_bound <= k_upper_bound, then there is an integer k in the range,
    # meaning a peak of |cos(x)| = 1 is included in the interval.
    contains_peak = k_lower_bound <= k_upper_bound
    max_vals[contains_peak] = 1.0

    return max_vals


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
