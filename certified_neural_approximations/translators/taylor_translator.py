import numpy as np

class CertifiedFirstOrderTaylorExpansion:
    """
    Represents a first-order Taylor expansion with certified error bounds.
    
    For a function f: R^n -> R^m, the Taylor expansion around point c is:
    f(x) ≈ f(c) + ∇f(c)(x - c) + R(x)
    
    where R(x) is the remainder term bounded by interval arithmetic.
    
    Attributes:
        expansion_point (np.ndarray): Point c around which the expansion is computed
        domain (tuple): (lower, upper) bounds defining the domain of validity
        linear_approximation (tuple): (Jacobian, constant) where:
            - Jacobian: ∇f(c) matrix of partial derivatives
            - constant: f(c) function value at expansion point
        remainder (tuple): (lower, upper) certified bounds on the remainder term
    """
    
    def __init__(self, expansion_point, domain, linear_approximation = None, remainder=None):
        """
        Initialize a certified first-order Taylor expansion.
        
        Args:
            expansion_point (np.ndarray): Center point of the expansion
            domain (tuple): (lower_bounds, upper_bounds) defining valid input region
            linear_approximation (tuple): (Jacobian_matrix, function_value_at_center)
            remainder (tuple): (remainder_lower_bounds, remainder_upper_bounds)
        """
        self.expansion_point = expansion_point
        self.domain = domain
        if linear_approximation is None:
            # Default f(x) = x
            self.linear_approximation = (np.eye(expansion_point.size), expansion_point)
            self.remainder = (np.zeros_like(expansion_point), np.zeros_like(expansion_point))
        else:
            self.linear_approximation = linear_approximation
            self.remainder = remainder

    def __add__(self, other):
        """
        Addition operation for Taylor expansions.
        
        For f(x) = g(x) + h(x):
        - Linear parts add: ∇f = ∇g + ∇h
        - Constants add: f(c) = g(c) + h(c)  
        - Remainders add: R_f = R_g + R_h
        
        Args:
            other: Another CertifiedFirstOrderTaylorExpansion or scalar
            
        Returns:
            CertifiedFirstOrderTaylorExpansion: Result of addition
        """
        if isinstance(other, CertifiedFirstOrderTaylorExpansion):
            # Ensure compatible expansion points and domains
            assert all(self.expansion_point == other.expansion_point)
            assert self.domain == other.domain

            # Properly add the linear approximation tuples element-wise
            new_jacobian = self.linear_approximation[0] + other.linear_approximation[0]
            new_constant = self.linear_approximation[1] + other.linear_approximation[1]
            new_linear_approximation = (new_jacobian, new_constant)

            return CertifiedFirstOrderTaylorExpansion(
                expansion_point=self.expansion_point,
                domain=self.domain,
                linear_approximation=new_linear_approximation,
                remainder=(self.remainder[0] + other.remainder[0], self.remainder[1] + other.remainder[1])
            )
        elif isinstance(other, (int, float)):
            # Adding a scalar only affects the constant term
            return CertifiedFirstOrderTaylorExpansion(
                expansion_point=self.expansion_point,
                domain=self.domain,
                linear_approximation=(self.linear_approximation[0], self.linear_approximation[1] + other),
                remainder=self.remainder
            )
        else:
            raise ValueError("Unsupported type for addition")

    def __radd__(self, other):
        """Right addition (scalar + TaylorExpansion)."""
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtraction operation for Taylor expansions.
        
        For f(x) = g(x) - h(x):
        - Linear parts subtract: ∇f = ∇g - ∇h
        - Constants subtract: f(c) = g(c) - h(c)
        - Remainders subtract with interval arithmetic: R_f = R_g - R_h
        
        Note: For interval subtraction [a,b] - [c,d] = [a-d, b-c]
        """
        if isinstance(other, CertifiedFirstOrderTaylorExpansion):
            assert all(self.expansion_point == other.expansion_point)
            assert self.domain == other.domain

            # Properly subtract the linear approximation tuples element-wise
            new_jacobian = self.linear_approximation[0] - other.linear_approximation[0]
            new_constant = self.linear_approximation[1] - other.linear_approximation[1]
            new_linear_approximation = (new_jacobian, new_constant)

            return CertifiedFirstOrderTaylorExpansion(
                expansion_point=self.expansion_point,
                domain=self.domain,
                linear_approximation=new_linear_approximation,
                # Interval subtraction: [a,b] - [c,d] = [a-d, b-c]
                remainder=(self.remainder[0] - other.remainder[1], self.remainder[1] - other.remainder[0])
            )
        elif isinstance(other, (int, float)):
            # Subtracting a scalar only affects the constant term
            return CertifiedFirstOrderTaylorExpansion(
                expansion_point=self.expansion_point,
                domain=self.domain,
                linear_approximation=(self.linear_approximation[0], self.linear_approximation[1] - other),
                remainder=self.remainder
            )
        else:
            raise ValueError("Unsupported type for subtraction")

    def __rsub__(self, other):
        """Right subtraction (scalar - TaylorExpansion)."""
        if isinstance(other, CertifiedFirstOrderTaylorExpansion):
            assert self.expansion_point == other.expansion_point
            assert self.domain == other.domain

            # Properly subtract the linear approximation tuples element-wise
            new_jacobian = other.linear_approximation[0] - self.linear_approximation[0]
            new_constant = other.linear_approximation[1] - self.linear_approximation[1]
            new_linear_approximation = (new_jacobian, new_constant)

            return CertifiedFirstOrderTaylorExpansion(
                expansion_point=self.expansion_point,
                domain=self.domain,
                linear_approximation=new_linear_approximation,
                remainder=(other.remainder[0] - self.remainder[1], other.remainder[1] - self.remainder[0])
            )
        elif isinstance(other, (int, float)):
            return CertifiedFirstOrderTaylorExpansion(
                expansion_point=self.expansion_point,
                domain=self.domain,
                linear_approximation=(-self.linear_approximation[0], other - self.linear_approximation[1]),
                # When subtracting from scalar: c - [a,b] = [c-b, c-a]
                remainder=(-self.remainder[1], -self.remainder[0])
            )
        else:
            raise ValueError("Unsupported type for subtraction")

    def __mul__(self, other):
        """
        Multiplication operation for Taylor expansions.
        
        For f(x) = g(x) * h(x), using product rule:
        f(x) = g(c)h(c) + [g(c)∇h(c) + h(c)∇g(c)](x-c) + higher_order_terms
        
        The remainder includes:
        1. Propagated remainders: g(c)*R_h + h(c)*R_g
        2. Higher-order terms from (∇g·dx)(∇h·dx)
        3. Cross terms: R_g * R_h
        """
        if isinstance(other, CertifiedFirstOrderTaylorExpansion):
            # Taylor expansion multiplication using product rule
            assert np.allclose(self.expansion_point, other.expansion_point, atol=1e-12), "Expansion points must match for TE * TE"
            assert np.allclose(self.domain[0], other.domain[0], atol=1e-12) and np.allclose(self.domain[1], other.domain[1], atol=1e-12), "Domains must match for TE * TE"

            # Extract components: (Jacobian, constant)
            J_g, y0_g = self.linear_approximation
            R_g_low, R_g_high = self.remainder

            J_h, y0_h = other.linear_approximation
            R_h_low, R_h_high = other.remainder
            
            # Ensure arrays are at least 1D for consistent operations
            y0_g = np.atleast_1d(y0_g)
            y0_h = np.atleast_1d(y0_h)

            # Product rule: f(c) = g(c) * h(c)
            new_const = y0_g * y0_h
            
            # Product rule: ∇f(c) = g(c)∇h(c) + h(c)∇g(c)
            # Handle broadcasting correctly for element-wise multiplication
            if len(y0_g.shape) == 1 and len(y0_h.shape) == 1:
                # Element-wise case: each output component is independent
                new_J = np.diag(y0_g) @ J_h + np.diag(y0_h) @ J_g
            else:
                raise NotImplementedError("General tensor multiplication not yet implemented")

            # Remainder computation with proper interval arithmetic
            # Term 1: g(c) * R_h
            term1_low = np.where(y0_g >= 0, y0_g * R_h_low, y0_g * R_h_high)
            term1_high = np.where(y0_g >= 0, y0_g * R_h_high, y0_g * R_h_low)
            
            # Term 2: h(c) * R_g  
            term2_low = np.where(y0_h >= 0, y0_h * R_g_low, y0_h * R_g_high)
            term2_high = np.where(y0_h >= 0, y0_h * R_g_high, y0_h * R_g_low)

            # Term 3: R_g * R_h (cross terms)
            cross_products = np.array([
                R_g_low * R_h_low, R_g_low * R_h_high,
                R_g_high * R_h_low, R_g_high * R_h_high
            ])
            term3_low = np.min(cross_products, axis=0)
            term3_high = np.max(cross_products, axis=0)

            # Higher Order Terms (HOT) from (∇g·dx)(∇h·dx)
            dx_low = self.domain[0] - self.expansion_point
            dx_high = self.domain[1] - self.expansion_point

            # Compute interval bounds for Jacobian times displacement
            u_low, u_high = _mat_interval_vec_mul(J_g, dx_low, dx_high)
            v_low, v_high = _mat_interval_vec_mul(J_h, dx_low, dx_high)
            
            u_low, u_high = np.atleast_1d(u_low), np.atleast_1d(u_high)
            v_low, v_high = np.atleast_1d(v_low), np.atleast_1d(v_high)

            # Interval multiplication for higher-order terms
            uv_products = np.array([
                u_low * v_low, u_low * v_high,
                u_high * v_low, u_high * v_high
            ])
            hot_low = np.min(uv_products, axis=0)
            hot_high = np.max(uv_products, axis=0)
            
            # Combine all remainder terms
            final_rem_low = term1_low + term2_low + term3_low + hot_low
            final_rem_high = term1_high + term2_high + term3_high + hot_high
            
            return CertifiedFirstOrderTaylorExpansion(
                self.expansion_point,
                self.domain,
                (new_J, new_const),
                (final_rem_low, final_rem_high)
            )
        elif isinstance(other, (int, float, np.number)):
            # Scalar multiplication: scales all terms
            new_df_c = self.linear_approximation[0] * other
            new_f_c = self.linear_approximation[1] * other
            # For interval [a,b] * c: if c≥0 then [ac,bc], if c<0 then [bc,ac]
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
        """Right multiplication (scalar * TaylorExpansion)."""
        if isinstance(other, (int, float, np.number)):
            # Multiplication is commutative for scalars
            return self.__mul__(other)
        else:
            return NotImplemented

    def _reciprocal(self) -> 'CertifiedFirstOrderTaylorExpansion':
        """
        Compute the reciprocal 1/f(x) of a Taylor expansion.
        
        For g(y) = 1/y where y = f(x):
        g(y₀) = 1/y₀
        g'(y₀) = -1/y₀²
        
        The Taylor expansion becomes:
        1/f(x) ≈ 1/f(c) - (1/f(c)²)∇f(c)(x-c) + remainder
        
        Raises:
            ValueError: If the range of f(x) contains zero
        """
        y0 = self.linear_approximation[1]  # f(c)
        J_a = self.linear_approximation[0] # ∇f(c)
        R_a_lower, R_a_upper = self.remainder

        # Check that zero is not in the range to avoid division by zero
        a_range_lower, a_range_upper = self.range()
        if np.any((a_range_lower <= 0) & (a_range_upper >= 0)):
            raise ValueError("Reciprocal of a Taylor expansion whose range contains zero is undefined.")

        # g(y₀) = 1/y₀
        new_const = 1.0 / y0
        # g'(y₀) = -1/y₀²
        grad_g_y0 = -1.0 / (y0**2)
        # Chain rule: ∇(1/f) = g'(f(c))∇f(c)
        new_J = grad_g_y0.reshape(-1, 1) * J_a
        
        # Propagate remainder through the derivative
        prop_rem_term1 = grad_g_y0 * R_a_lower
        prop_rem_term2 = grad_g_y0 * R_a_upper
        propagated_rem_lower = np.minimum(prop_rem_term1, prop_rem_term2)
        propagated_rem_upper = np.maximum(prop_rem_term1, prop_rem_term2)        # Second-order remainder: g''(y) = 2/y³, so Lagrange remainder involves max |2/η³|
        coeff_f_double_prime = 2.0
        exponent_f_double_prime = -3
        M_g_double_prime_factor = max_monomial_vectorized(coeff_f_double_prime, exponent_f_double_prime, (a_range_lower, a_range_upper))
        
        # Use proper remainder bound computation for function composition f(g(x)) = 1/g(x)
        second_derivative_bounds = (-M_g_double_prime_factor, M_g_double_prime_factor)  # Symmetric for reciprocal
        local_error_magnitude_min, local_error_magnitude_max = compute_function_composition_remainder_bound(
            self, second_derivative_bounds
        )
        
        # For reciprocal, the remainder is symmetric, so we use the absolute maximum
        local_error_magnitude = np.maximum(np.abs(local_error_magnitude_min), np.abs(local_error_magnitude_max))

        final_rem_lower = propagated_rem_lower - local_error_magnitude
        final_rem_upper = propagated_rem_upper + local_error_magnitude

        return CertifiedFirstOrderTaylorExpansion(
            self.expansion_point,
            self.domain,
            (new_J, new_const),
            (final_rem_lower, final_rem_upper)
        )

    def __truediv__(self, other):
        """Division: self / other = self * (1/other)."""
        if isinstance(other, (int, float, np.number)):
            if other == 0:
                raise ZeroDivisionError("Division by zero scalar.")
            return self * (1.0 / other)
        elif isinstance(other, CertifiedFirstOrderTaylorExpansion):
            return self * other._reciprocal()
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        """Right division: other / self = other * (1/self)."""
        if isinstance(other, (int, float, np.number)):
            return other * self._reciprocal()
        else:
            return NotImplemented

    def __getitem__(self, key):
        """
        Index into a Taylor expansion to extract specific components.
        
        Args:
            key (int or slice): Index or slice of the components to extract
            
        Returns:
            CertifiedFirstOrderTaylorExpansion: Single-component or multi-component Taylor expansion
        """
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
        elif isinstance(key, slice):
            new_df_c = self.linear_approximation[0][key]
            new_f_c = self.linear_approximation[1][key]
            new_remainder_lower = self.remainder[0][key]
            new_remainder_upper = self.remainder[1][key]
            return CertifiedFirstOrderTaylorExpansion(
                expansion_point=self.expansion_point,
                domain=self.domain,
                linear_approximation=(new_df_c, new_f_c),
                remainder=(new_remainder_lower, new_remainder_upper)
            )
        else:
            raise TypeError(f"CertifiedFirstOrderTaylorExpansion indices must be integers or slices, not {type(key)}")

    def range(self):
        """
        Compute the range (lower and upper bounds) of the Taylor expansion over its domain.
        
        The range is computed as:
        f(x) = f(c) + ∇f(c)(x-c) + R(x)
        
        For the affine part f(c) + ∇f(c)(x-c), we use interval arithmetic:
        - If ∇f(c)[i] ≥ 0: contributes ∇f(c)[i] * domain_lower[i] to lower bound
        - If ∇f(c)[i] < 0: contributes ∇f(c)[i] * domain_upper[i] to lower bound
        
        Returns:
            tuple: (lower_bounds, upper_bounds) arrays of shape matching the output dimension
        """
        J = self.linear_approximation[0]  # Jacobian ∇f(c)
        fc = self.linear_approximation[1]  # f(c)
        c = self.expansion_point
        domain_lower, domain_upper = self.domain
        R_lower, R_upper = self.remainder

        # Ensure all arrays are at least 1D for consistent operations
        fc = np.atleast_1d(fc)
        c = np.atleast_1d(c)
        R_lower = np.atleast_1d(R_lower)
        R_upper = np.atleast_1d(R_upper)
        domain_lower = np.atleast_1d(domain_lower)
        domain_upper = np.atleast_1d(domain_upper)

        # Convert to affine form: f(x) = A*x + b where A = J, b = fc - J*c
        b_affine = fc - (J @ c)
        A_affine = J
        
        # Split positive and negative parts for interval arithmetic
        A_affine_pos = np.maximum(A_affine, 0)
        A_affine_neg = np.minimum(A_affine, 0)

        # Compute bounds of affine part using interval arithmetic
        affine_range_lower = (A_affine_pos @ domain_lower) + (A_affine_neg @ domain_upper) + b_affine
        affine_range_upper = (A_affine_pos @ domain_upper) + (A_affine_neg @ domain_lower) + b_affine

        # Add remainder bounds to get total range
        total_range_lower = affine_range_lower + R_lower
        total_range_upper = affine_range_upper + R_upper

        # Sanity check: lower bounds should not exceed upper bounds
        assert np.all(total_range_lower <= total_range_upper + 1e-9), \
            f"Lower bound > upper bound in range calculation: {total_range_lower} vs {total_range_upper}"

        return (total_range_lower, total_range_upper)

    def __neg__(self):
        """
        Negation operation for Taylor expansions.
        
        Returns:
            CertifiedFirstOrderTaylorExpansion: Negated Taylor expansion
        """
        return CertifiedFirstOrderTaylorExpansion(
            expansion_point=self.expansion_point,
            domain=self.domain,
            linear_approximation=(-self.linear_approximation[0], -self.linear_approximation[1]),
            remainder=(-self.remainder[1], -self.remainder[0])
        )


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
        f_y0_val = np.sin(y0)         # sin(y0)
        grad_f_y0 = np.cos(y0)        # cos(y0), derivative of sin(y) at y0

        linear_term_jacobian = grad_f_y0.reshape(-1, 1) * a.linear_approximation[0]

        # Lagrange remainder for sin:
        domain_low, domain_high = a.range()

        # Check if any point x = k*2pi + pi/2 (where sin(x) = +1) is within [a, b]
        # This is true if there is an integer k such that:
        # a <= k*pi + pi/2 <= b  =>  (a - pi/2)/pi <= k <= (b - pi/2)/pi
        # So, we check if floor((b - pi/2)/pi) >= ceil((a - pi/2)/pi)

        # Lower bound for k (must be an integer, so use ceil)
        k_lower_bound = np.ceil((domain_low - 3*np.pi/2) / (2*np.pi))
        # Upper bound for k (must be an integer, so use floor)
        k_upper_bound = np.floor((domain_high - 3*np.pi/2) / (2*np.pi))
        contains_trough = k_lower_bound <= k_upper_bound

        # If no crest, the maximum value of |sin(x)| is at the endpoints
        M_lagrange_max = np.maximum(-np.sin(domain_low), -np.sin(domain_high))
        M_lagrange_max[contains_trough] = 1.0  # If contains trough, max is 1.0
        M_lagrange_max = np.maximum(M_lagrange_max, 0.0)  # Ensure non-negative max

        # Only need to consider the distance between peaks/troughs and the expansion point
        min_abs_y_minus_peak = np.minimum(np.abs(k_lower_bound*2*np.pi + np.pi/2 - y0), np.abs(k_upper_bound*2*np.pi + np.pi/2 - y0))
        
        max_abs_y_minus_y0 = np.maximum(np.abs(domain_low - y0), np.abs(domain_high - y0))
        max_abs_y_minus_y0 = np.minimum(max_abs_y_minus_y0, min_abs_y_minus_peak)
        max_sq_y_minus_y0 = max_abs_y_minus_y0 ** 2
        local_error_magnitude_max = (M_lagrange_max / 2) * max_sq_y_minus_y0
        
        # Lower bound for k (must be an integer, so use ceil)
        k_lower_bound = np.ceil((domain_low - np.pi/2) / (2*np.pi))
        # Upper bound for k (must be an integer, so use floor)
        k_upper_bound = np.floor((domain_high - np.pi/2) / (2*np.pi))
        contains_crest = k_lower_bound <= k_upper_bound

        # If no trough, the minimum value of |sin(x)| is at the endpoints
        M_lagrange_min = np.minimum(-np.sin(domain_low), -np.sin(domain_high))
        M_lagrange_min[contains_crest] = -1.0  # If contains crest, min is -1.0
        M_lagrange_min = np.minimum(M_lagrange_min, 0.0)  # Ensure non-negative max
        
        # Only need to consider the distance between peaks/troughs and the expansion point
        min_abs_y_minus_trough = np.minimum(np.abs(k_lower_bound*2*np.pi + 3*np.pi/2 - y0), np.abs(k_upper_bound*2*np.pi + 3*np.pi/2 - y0))
        
        max_abs_y_minus_y0 = np.maximum(np.abs(domain_low - y0), np.abs(domain_high - y0))
        max_abs_y_minus_y0 = np.minimum(max_abs_y_minus_y0, min_abs_y_minus_trough)
        max_sq_y_minus_y0 = max_abs_y_minus_y0 ** 2
        local_error_magnitude_min = (M_lagrange_min / 2) * max_sq_y_minus_y0

        prop_rem_lower_y, prop_rem_upper_y = a.remainder
        
        term1_rem = grad_f_y0 * prop_rem_lower_y
        term2_rem = grad_f_y0 * prop_rem_upper_y
        
        propagated_taylor_rem_lower = np.minimum(term1_rem, term2_rem)
        propagated_taylor_rem_upper = np.maximum(term1_rem, term2_rem)

        final_rem_lower = propagated_taylor_rem_lower + local_error_magnitude_min
        final_rem_upper = propagated_taylor_rem_upper + local_error_magnitude_max

        remainder = (final_rem_lower, final_rem_upper)

        return CertifiedFirstOrderTaylorExpansion(
            a.expansion_point,
            a.domain,
            (linear_term_jacobian, f_y0_val),
            remainder
        )

    def cos(self, a):
        """
        Element-wise cosine
        :param a: CertifiedFirstOrderTaylorExpansion
        :return: CertifiedFirstOrderTaylorExpansion
        """
        y0 = a.linear_approximation[1] # Center of expansion for y
        cos_y0_val = np.cos(y0)        # cos(y0)
        grad_cos_y0 = -np.sin(y0)      # -sin(y0), derivative of cos(y) at y0

        linear_term_jacobian = grad_cos_y0.reshape(-1, 1) * a.linear_approximation[0]

        # Lagrange remainder for cos:
        domain_low, domain_high = a.range()

        # Check if any point x = k*2pi (where cos(x) = +1) is within [a, b]
        # This is true if there is an integer k such that:
        # a <= k*2pi <= b  =>  (a)/2pi <= k <= (b)/2pi
        
        # Lower bound for k (must be an integer, so use ceil)
        k_lower_bound = np.ceil((domain_low - np.pi) / (2*np.pi))
        # Upper bound for k (must be an integer, so use floor)
        k_upper_bound = np.floor((domain_high - np.pi) / (2*np.pi))
        contains_trough = k_lower_bound <= k_upper_bound

        # If no crest, the maximum value of -cos(x) is at the endpoints
        M_lagrange_max = np.maximum(-np.cos(domain_low), -np.cos(domain_high))
        M_lagrange_max[contains_trough] = 1.0  # If contains trough, max is 1.0
        M_lagrange_max = np.maximum(M_lagrange_max, 0.0)  # Ensure non-negative max

        # Only need to consider the distance between peaks/troughs and the expansion point      
        min_abs_y_minus_peak = np.minimum(np.abs(k_lower_bound*2*np.pi - y0), np.abs(k_upper_bound*2*np.pi - y0))
                
        max_abs_y_minus_y0 = np.maximum(np.abs(domain_low - y0), np.abs(domain_high - y0))
        max_abs_y_minus_y0 = np.minimum(max_abs_y_minus_y0, min_abs_y_minus_peak)
        max_sq_y_minus_y0 = max_abs_y_minus_y0 ** 2
        local_error_magnitude_max = (M_lagrange_max / 2) * max_sq_y_minus_y0

        # Lower bound for k (must be an integer, so use ceil)
        k_lower_bound = np.ceil(domain_low / (2*np.pi))
        # Upper bound for k (must be an integer, so use floor)
        k_upper_bound = np.floor(domain_high/ (2*np.pi))
        contains_crest = k_lower_bound <= k_upper_bound

        # If no trough, the minimum value of -cos(x) is at the endpoints
        M_lagrange_min = np.minimum(-np.cos(domain_low), -np.cos(domain_high))
        M_lagrange_min[contains_crest] = -1.0  # If contains crest, min is -1.0
        M_lagrange_min = np.minimum(M_lagrange_min, 0.0)  # Ensure non-negative max
                      
        # Only need to consider the distance between peaks/troughs and the expansion point
        min_abs_y_minus_trough = np.minimum(np.abs(k_lower_bound*2*np.pi + np.pi - y0), np.abs(k_upper_bound*2*np.pi + np.pi - y0))
        
        max_abs_y_minus_y0 = np.maximum(np.abs(domain_low - y0), np.abs(domain_high - y0))
        max_abs_y_minus_y0 = np.minimum(max_abs_y_minus_y0, min_abs_y_minus_trough)
        max_sq_y_minus_y0 = max_abs_y_minus_y0 ** 2
        local_error_magnitude_min = (M_lagrange_min / 2) * max_sq_y_minus_y0

        # Propagate remainder through the derivative
        prop_rem_lower_y, prop_rem_upper_y = a.remainder
        term1_rem = grad_cos_y0 * prop_rem_lower_y
        term2_rem = grad_cos_y0 * prop_rem_upper_y

        propagated_taylor_rem_lower = np.minimum(term1_rem, term2_rem)
        propagated_taylor_rem_upper = np.maximum(term1_rem, term2_rem)

        final_rem_lower = propagated_taylor_rem_lower + local_error_magnitude_min
        final_rem_upper = propagated_taylor_rem_upper + local_error_magnitude_max

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
            np.zeros_like(np.exp(range[0])),
            np.maximum(
                np.exp(range[0]) - (exp_y0 * (1 - y0) + exp_y0 * range[0]),
                np.exp(range[1]) - (exp_y0 * (1 - y0) + exp_y0 * range[1])
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
        lower, upper = a.range()
        if np.any(lower <= 0):
            raise ValueError("Logarithm domain error: range[0] must be greater than 0")

        # Compute the second derivative bounds for log(x): f''(x) = -1/x^2
        second_derivative_lower = -1 / (lower**2)
        second_derivative_upper = -1 / (upper**2)

        # Ensure valid bounds
        second_derivative_lower = np.minimum(second_derivative_lower, 0)
        second_derivative_upper = np.maximum(second_derivative_upper, 0)

        # Compute the maximum deviation from the expansion point
        max_deviation = np.maximum(
            np.abs(lower - y0),
            np.abs(upper - y0)
        )

        # Compute the remainder bounds using the second derivative
        remainder_lower = (second_derivative_lower / 2) * max_deviation**2
        remainder_upper = (second_derivative_upper / 2) * max_deviation**2

        # Propagate the remainder through the derivative
        remainder1, remainder2 = grad_y0 * a.remainder[0], grad_y0 * a.remainder[1]
        propagated_rem_lower = np.minimum(remainder1, remainder2)
        propagated_rem_upper = np.maximum(remainder1, remainder2)

        # Combine propagated and second derivative remainders
        final_rem_lower = propagated_rem_lower + remainder_lower
        final_rem_upper = propagated_rem_upper + remainder_upper

        return CertifiedFirstOrderTaylorExpansion(
            a.expansion_point,
            a.domain,
            (linear_term, log_y0),
            (final_rem_lower, final_rem_upper)
        )

    def sqrt(self, a):
        """
        Element-wise square root
        :param a: CertifiedFirstOrderTaylorExpansion
        :return: CertifiedFirstOrderTaylorExpansion
        """
        y0 = a.linear_approximation[1]
        sqrt_y0 = np.sqrt(y0)
        grad_y0 = 0.5 / sqrt_y0

        linear_term = grad_y0.reshape(grad_y0.shape[0], 1) * a.linear_approximation[0]

        # Use monotonicity of sqrt
        range_lower, range_upper = a.range()
        if np.any(range_lower < 0):
            raise ValueError("Square root domain error: range[0] must be non-negative")
        
        # Use monotonicity: sqrt is concave, so linear approximation is always above the true function
        # This means the remainder is always non-positive
        linear_at_lower = sqrt_y0 + grad_y0 * (range_lower - y0)
        linear_at_upper = sqrt_y0 + grad_y0 * (range_upper - y0)
        
        # Handle zero values element-wise for multidimensional case
        true_at_lower = np.where(range_lower == 0, np.zeros_like(range_lower), np.sqrt(range_lower))
        true_at_upper = np.sqrt(range_upper)
        
        # Since sqrt is concave (second derivative negative), linear approximation overestimates
        # So remainder = true_value - linear_approximation ≤ 0
        remainder_at_lower = true_at_lower - linear_at_lower
        remainder_at_upper = true_at_upper - linear_at_upper
        
        # The remainder bounds are always non-positive due to concavity
        remainder_lower = np.minimum(remainder_at_lower, remainder_at_upper)
        remainder_upper = np.maximum(remainder_at_lower, remainder_at_upper)
        
        # Ensure upper bound is never positive (due to concavity)
        remainder_upper = np.maximum(remainder_upper, 0.0)

        # Propagate the remainder through the derivative
        prop_rem_lower, prop_rem_upper = a.remainder
        term1_rem = grad_y0 * prop_rem_lower
        term2_rem = grad_y0 * prop_rem_upper

        propagated_rem_lower = np.minimum(term1_rem, term2_rem)
        propagated_rem_upper = np.maximum(term1_rem, term2_rem)

        # Combine propagated and local remainders
        final_rem_lower = propagated_rem_lower + remainder_lower
        final_rem_upper = propagated_rem_upper + remainder_upper

        return CertifiedFirstOrderTaylorExpansion(
            a.expansion_point,
            a.domain,
            (linear_term, sqrt_y0),
            (final_rem_lower, final_rem_upper)
        )

    def cbrt(self, a):
        """
        Element-wise cube root
        :param a: CertifiedFirstOrderTaylorExpansion
        :return: CertifiedFirstOrderTaylorExpansion
        """
        y0 = a.linear_approximation[1]
        cbrt_y0 = np.power(y0, 1/3)
        grad_y0 = (1/3) * np.power(y0, -2/3)

        linear_term = grad_y0.reshape(grad_y0.shape[0], 1) * a.linear_approximation[0]

        # Use concavity/convexity properties of cbrt
        range_lower, range_upper = a.range()

        # Compute linear approximation at endpoints
        linear_at_lower = cbrt_y0 + grad_y0 * (range_lower - y0)
        linear_at_upper = cbrt_y0 + grad_y0 * (range_upper - y0)
        
        # Handle zero values element-wise for multidimensional case
        range_lower = np.maximum(range_lower, 0)  # Ensure non-negative for cbrt
        true_at_lower = np.power(range_lower, 1/3)
        true_at_upper = np.power(range_upper, 1/3)
        
        # Remainder = true_value - linear_approximation
        remainder_at_lower = true_at_lower - linear_at_lower
        remainder_at_upper = true_at_upper - linear_at_upper
        
        # The remainder bounds are always non-positive due to concavity
        remainder_lower = np.minimum(remainder_at_lower, remainder_at_upper)
        remainder_upper = np.maximum(remainder_at_lower, remainder_at_upper)
        
        # Ensure upper bound is never positive (due to concavity)
        remainder_upper = np.maximum(remainder_upper, 0.0)

        # Propagate the remainder through the derivative
        prop_rem_lower, prop_rem_upper = a.remainder
        term1_rem = grad_y0 * prop_rem_lower
        term2_rem = grad_y0 * prop_rem_upper

        propagated_rem_lower = np.minimum(term1_rem, term2_rem)
        propagated_rem_upper = np.maximum(term1_rem, term2_rem)

        # Combine propagated and local remainders
        final_rem_lower = propagated_rem_lower + remainder_lower
        final_rem_upper = propagated_rem_upper + remainder_upper

        remainder = (final_rem_lower, final_rem_upper)

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
        f_y0 = np.power(y0, exponent_b)
        grad_f_y0 = exponent_b * np.power(y0, exponent_b - 1)

        linear_term_jacobian = grad_f_y0.reshape(-1, 1) * a.linear_approximation[0]
        
        range_of_y = a.range()
        coeff_f_double_prime = exponent_b * (exponent_b - 1)
        exponent_f_double_prime = exponent_b - 2
        M_lagrange_max = max_monomial_vectorized(coeff_f_double_prime, exponent_f_double_prime, range_of_y)
        M_lagrange_min = min_monomial_vectorized(coeff_f_double_prime, exponent_f_double_prime, range_of_y)
        
        # Use proper remainder bound computation for function composition f(g(x)) = g(x)^exponent_b
        second_derivative_bounds = (M_lagrange_min, M_lagrange_max)
        local_error_magnitude_min, local_error_magnitude_max = compute_function_composition_remainder_bound(
            a, second_derivative_bounds
        )

        prop_rem_lower_y, prop_rem_upper_y = a.remainder
        
        term1_rem = grad_f_y0 * prop_rem_lower_y
        term2_rem = grad_f_y0 * prop_rem_upper_y
        
        propagated_taylor_rem_lower = np.minimum(term1_rem, term2_rem)
        propagated_taylor_rem_upper = np.maximum(term1_rem, term2_rem)

        final_rem_lower = propagated_taylor_rem_lower + local_error_magnitude_min
        final_rem_upper = propagated_taylor_rem_upper + local_error_magnitude_max
        
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
        # Assert that all expansion points are the same
        for i, x in enumerate(xs):
            assert np.array_equal(x.expansion_point, xs[0].expansion_point), \
                f"Expansion point mismatch at index {i}: {x.expansion_point} != {xs[0].expansion_point}"
        
        # Assert that all domains are the same
        for i, x in enumerate(xs):
            assert np.array_equal(x.domain[0], xs[0].domain[0]) and np.array_equal(x.domain[1], xs[0].domain[1]), \
                f"Domain mismatch at index {i}: {x.domain} != {xs[0].domain}"
        
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
        :param point: expansion point
        :param lower: lower bounds of domain
        :param upper: upper bounds of domain
        :return: CertifiedFirstOrderTaylorExpansion
        """

        # for f(x) = x, the Taylor expansion is c + (x - c) \oplus R where R = 0
        # The Jacobian should be the identity matrix of appropriate size
        return CertifiedFirstOrderTaylorExpansion(
            expansion_point=point,
            domain=(lower, upper),
            linear_approximation=(np.eye(point.shape[0]), point),
            remainder=(np.zeros(point.shape[0]), np.zeros(point.shape[0]))
        )


def max_monomial_vectorized(c, n, intervals):
    """
    Bound the maximum value of a univariate monomial f(x) = c * x^n over multiple intervals.
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
    max_values = np.maximum(f_a, f_b)

    # Check critical points for even powers (n > 0 and n is even)
    even_power = (n > 0) & (n % 2 == 0)
    if np.any(even_power):
        # For even powers, the maximum absolute value occurs at the endpoint with the largest magnitude
        abs_a = np.abs(a)
        abs_b = np.abs(b)
        critical_max = np.maximum(abs_a, abs_b)
        max_values[even_power] = np.maximum(max_values[even_power], (c * np.power(critical_max, n))[even_power])

    return max_values


def min_monomial_vectorized(c, n, intervals):
    """
    Bound the minimum value of a univariate monomial f(x) = c * x^n over multiple intervals.
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

    # Check critical points for even powers (n > 0 and n is even)
    even_power = (n > 0) & (n % 2 == 0)
    if np.any(even_power):
        # For even powers, the minimum absolute value occurs at the endpoint with the smallest magnitude
        abs_a = np.abs(a)
        abs_b = np.abs(b)
        critical_min = np.minimum(abs_a, abs_b)
        min_values[even_power] = np.minimum(min_values[even_power], (c * np.power(critical_min, n))[even_power])

    return min_values

def max_euclidean_distance_squared(expansion_point, domain):
    """
    Compute the maximum Euclidean distance squared from the expansion point
    to any point in the given domain.
    
    For a domain [a, b] and expansion point c, this computes:
    max_{x ∈ [a,b]} ||x - c||²
    
    The maximum occurs at one of the corners of the domain hypercube.
    
    Args:
        expansion_point (np.ndarray): Point c around which the expansion is computed
        domain (tuple): (lower_bounds, upper_bounds) defining the domain
        
    Returns:
        float: Maximum Euclidean distance squared
    """
    expansion_point = np.atleast_1d(expansion_point)
    domain_lower = np.atleast_1d(domain[0])
    domain_upper = np.atleast_1d(domain[1])
    
    # For each dimension, the maximum distance is achieved at either 
    # the lower or upper bound - whichever is farther from the expansion point
    max_dist_per_dim = np.maximum(
        np.abs(domain_lower - expansion_point),
        np.abs(domain_upper - expansion_point)
    )
    
    # The maximum Euclidean distance squared is the sum of squared distances per dimension
    return np.sum(max_dist_per_dim ** 2)

def compute_function_composition_remainder_bound(inner_taylor_expansion, second_derivative_bounds):
    """
    Compute Lagrange remainder bounds for univariate function composition f(g(x)).
    
    For a composition f(g(x)) where g: R^n -> R^m is represented by a Taylor expansion
    and f is applied element-wise, this computes the Lagrange remainder bounds.
    
    Args:
        inner_taylor_expansion: CertifiedFirstOrderTaylorExpansion representing g(x)
        second_derivative_bounds: (M_min, M_max) bounds on f''(y) over the range of g(x)
        
    Returns:
        tuple: (remainder_lower, remainder_upper) bounds for the Lagrange remainder
    """
    M_min, M_max = second_derivative_bounds
    
    # For composition f(g(x)), the Lagrange remainder is bounded by:
    # |R(x)| ≤ M/2! * |g(x) - g(c)|²
    # where M is the maximum of |f''(y)| over the range of g(x)
    
    # The challenge is to bound max_{x ∈ D} |g(x) - g(c)|²
    # For a Taylor expansion g(x) = g(c) + ∇g(c)(x-c) + R_g(x), we have:
    # |g(x) - g(c)| ≤ |∇g(c)(x-c)| + |R_g(x)|
    
    # Method 1: Use the maximum Euclidean distance in the input domain
    # This gives us max_{x ∈ D} |∇g(c)(x-c)|
    max_euclidean_dist_sq = max_euclidean_distance_squared(
        inner_taylor_expansion.expansion_point, 
        inner_taylor_expansion.domain
    )
    
    # For the linear part ∇g(c)(x-c), we need ||∇g(c)||² * ||x-c||²
    jacobian = inner_taylor_expansion.linear_approximation[0]
    
    # For each output component, compute the maximum contribution from the linear part
    if len(jacobian.shape) == 1:
        # Single output component
        max_linear_contrib_sq = (np.linalg.norm(jacobian)**2) * max_euclidean_dist_sq
    else:
        # Multiple output components
        max_linear_contrib_sq = np.array([
            (np.linalg.norm(jacobian[i])**2) * max_euclidean_dist_sq 
            for i in range(jacobian.shape[0])
        ])
      # Add the contribution from the remainder of the inner function
    # Use triangle inequality more precisely: |∇g(c)(x-c) + R_g(x)| ≤ |∇g(c)(x-c)| + |R_g(x)|
    R_g_lower, R_g_upper = inner_taylor_expansion.remainder
    max_remainder_contrib = np.maximum(np.abs(R_g_lower), np.abs(R_g_upper))
    
    # Combine using triangle inequality: |a + b|² ≤ (|a| + |b|)²
    max_linear_contrib = np.sqrt(max_linear_contrib_sq)
    max_deviation = max_linear_contrib + max_remainder_contrib
    max_deviation_sq = max_deviation ** 2
    
    # Apply the Lagrange remainder formula
    remainder_magnitude_max = np.maximum(0, (M_max / 2) * max_deviation_sq)
    remainder_magnitude_min = np.minimum(0, (M_min / 2) * max_deviation_sq)
    
    return remainder_magnitude_min, remainder_magnitude_max

# Helper function for TE * TE multiplication remainder calculation
def _mat_interval_vec_mul(M, v_low, v_high):
    M_pos = np.maximum(M, 0)
    M_neg = np.minimum(M, 0)
    res_low = M_pos @ v_low + M_neg @ v_high
    res_high = M_pos @ v_high + M_neg @ v_low
    return res_low, res_high
