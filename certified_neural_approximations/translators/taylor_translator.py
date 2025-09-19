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
        4. Cross terms: R_g * J_h(x-c) and R_h * J_g(x-c)
        """
        if isinstance(other, CertifiedFirstOrderTaylorExpansion):
            # Taylor expansion multiplication using product rule
            assert np.allclose(self.expansion_point, other.expansion_point, atol=1e-12), "Expansion points must match for TE * TE"
            assert np.allclose(self.domain[0], other.domain[0], atol=1e-12) and np.allclose(self.domain[1], other.domain[1], atol=1e-12), "Domains must match for TE * TE"

            # Extract components for self (g)
            jacobian_self, const_self = self.linear_approximation
            remainder_self_low, remainder_self_high = self.remainder

            # Extract components for other (h)
            jacobian_other, const_other = other.linear_approximation
            remainder_other_low, remainder_other_high = other.remainder
            
            # Ensure constant terms are at least 1D for consistent operations
            const_self_1d = np.atleast_1d(const_self)
            const_other_1d = np.atleast_1d(const_other)

            # Product rule for new constant: f(c) = g(c) * h(c)
            new_const = const_self_1d * const_other_1d
            
            # Product rule for new Jacobian: ∇f(c) = g(c)∇h(c) + h(c)∇g(c)
            if len(const_self_1d.shape) == 1 and len(const_other_1d.shape) == 1:
                # Element-wise case: each output component is independent
                new_jacobian = np.diag(const_self_1d) @ jacobian_other + np.diag(const_other_1d) @ jacobian_self
            else:
                raise NotImplementedError("General tensor multiplication not yet implemented for Jacobian calculation")

            # Remainder computation
            # Term: const_self * Remainder_other
            const_self_times_rem_other_low = np.where(const_self_1d >= 0, const_self_1d * remainder_other_low, const_self_1d * remainder_other_high)
            const_self_times_rem_other_high = np.where(const_self_1d >= 0, const_self_1d * remainder_other_high, const_self_1d * remainder_other_low)
            
            # Term: const_other * Remainder_self  
            const_other_times_rem_self_low = np.where(const_other_1d >= 0, const_other_1d * remainder_self_low, const_other_1d * remainder_self_high)
            const_other_times_rem_self_high = np.where(const_other_1d >= 0, const_other_1d * remainder_self_high, const_other_1d * remainder_self_low)

            # Term: Remainder_self * Remainder_other
            rem_self_times_rem_other_products = np.array([
                remainder_self_low * remainder_other_low, remainder_self_low * remainder_other_high,
                remainder_self_high * remainder_other_low, remainder_self_high * remainder_other_high
            ])
            rem_self_times_rem_other_low = np.min(rem_self_times_rem_other_products, axis=0)
            rem_self_times_rem_other_high = np.max(rem_self_times_rem_other_products, axis=0)

            # Higher Order Terms (HOT) from (Jacobian_self · dx) * (Jacobian_other · dx)
            dx_low = self.domain[0] - self.expansion_point
            dx_high = self.domain[1] - self.expansion_point

            # Compute interval bounds for Jacobian_self * dx
            j_self_times_dx_low, j_self_times_dx_high = _mat_interval_vec_mul(jacobian_self, dx_low, dx_high)
            j_self_times_dx_low_1d, j_self_times_dx_high_1d = np.atleast_1d(j_self_times_dx_low), np.atleast_1d(j_self_times_dx_high)
            
            # Compute interval bounds for Jacobian_other * dx
            j_other_times_dx_low, j_other_times_dx_high = _mat_interval_vec_mul(jacobian_other, dx_low, dx_high)
            j_other_times_dx_low_1d, j_other_times_dx_high_1d = np.atleast_1d(j_other_times_dx_low), np.atleast_1d(j_other_times_dx_high)

            # Interval multiplication for (Jacobian_self · dx) * (Jacobian_other · dx)
            hot_jdx_jdx_products = np.array([
                j_self_times_dx_low_1d * j_other_times_dx_low_1d, j_self_times_dx_low_1d * j_other_times_dx_high_1d,
                j_self_times_dx_high_1d * j_other_times_dx_low_1d, j_self_times_dx_high_1d * j_other_times_dx_high_1d
            ])
            hot_jdx_jdx_low = np.min(hot_jdx_jdx_products, axis=0)
            hot_jdx_jdx_high = np.max(hot_jdx_jdx_products, axis=0)
            
            # Term: Remainder_self * (Jacobian_other · dx)
            # Ensure Remainder_self components are at least 1D
            remainder_self_low_1d = np.atleast_1d(remainder_self_low)
            remainder_self_high_1d = np.atleast_1d(remainder_self_high)

            rem_self_times_j_other_dx_products = np.array([
                remainder_self_low_1d * j_other_times_dx_low_1d, remainder_self_low_1d * j_other_times_dx_high_1d,
                remainder_self_high_1d * j_other_times_dx_low_1d, remainder_self_high_1d * j_other_times_dx_high_1d
            ])
            rem_self_times_j_other_dx_low = np.min(rem_self_times_j_other_dx_products, axis=0)
            rem_self_times_j_other_dx_high = np.max(rem_self_times_j_other_dx_products, axis=0)

            # Term: Remainder_other * (Jacobian_self · dx)
            # Ensure Remainder_other components are at least 1D
            remainder_other_low_1d = np.atleast_1d(remainder_other_low)
            remainder_other_high_1d = np.atleast_1d(remainder_other_high)
            
            rem_other_times_j_self_dx_products = np.array([
                remainder_other_low_1d * j_self_times_dx_low_1d, remainder_other_low_1d * j_self_times_dx_high_1d,
                remainder_other_high_1d * j_self_times_dx_low_1d, remainder_other_high_1d * j_self_times_dx_high_1d
            ])
            rem_other_times_j_self_dx_low = np.min(rem_other_times_j_self_dx_products, axis=0)
            rem_other_times_j_self_dx_high = np.max(rem_other_times_j_self_dx_products, axis=0)
            
            # Combine all remainder terms
            final_remainder_low = (const_self_times_rem_other_low + 
                                   const_other_times_rem_self_low + 
                                   rem_self_times_rem_other_low + 
                                   hot_jdx_jdx_low + 
                                   rem_self_times_j_other_dx_low + 
                                   rem_other_times_j_self_dx_low)
            final_remainder_high = (const_self_times_rem_other_high + 
                                    const_other_times_rem_self_high + 
                                    rem_self_times_rem_other_high + 
                                    hot_jdx_jdx_high + 
                                    rem_self_times_j_other_dx_high + 
                                    rem_other_times_j_self_dx_high)
            
            return CertifiedFirstOrderTaylorExpansion(
                self.expansion_point,
                self.domain,
                (new_jacobian, new_const),
                (final_remainder_low, final_remainder_high)
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
        # Compute actual min and max of the second derivative 2/y^3 over the range of y
        M_min_g_double_prime = min_monomial_vectorized(coeff_f_double_prime, exponent_f_double_prime, (a_range_lower, a_range_upper))
        M_max_g_double_prime = max_monomial_vectorized(coeff_f_double_prime, exponent_f_double_prime, (a_range_lower, a_range_upper))
        
        second_derivative_bounds = (M_min_g_double_prime, M_max_g_double_prime)
        local_error_magnitude_min, local_error_magnitude_max = compute_function_composition_remainder_bound(
            self, second_derivative_bounds
        )
        
        # Combine propagated remainder with local Lagrange error
        final_rem_lower = propagated_rem_lower + local_error_magnitude_min
        final_rem_upper = propagated_rem_upper + local_error_magnitude_max

        # Apply monotonic bounds tightening for reciprocal function
        # 1/x is monotonically decreasing for x > 0 and x < 0 separately
        reciprocal_at_boundaries = (1.0 / a_range_lower, 1.0 / a_range_upper)  # Note: order matches domain order
        
        # Create temporary Taylor expansion to use the monotonic tightening helper
        temp_expansion = CertifiedFirstOrderTaylorExpansion(
            self.expansion_point,
            self.domain,
            (new_J, new_const),
            (final_rem_lower, final_rem_upper)
        )
        
        clip_rem_lower, clip_rem_upper = apply_monotonic_bounds_tightening(temp_expansion, reciprocal_at_boundaries, is_increasing=False)
        
        # Intersect the Taylor bounds with the monotonic bounds
        final_rem_lower = np.maximum(final_rem_lower, clip_rem_lower)
        final_rem_upper = np.minimum(final_rem_upper, clip_rem_upper)

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

        max_abs_y_minus_y0 = np.maximum(np.abs(domain_low - y0), np.abs(domain_high - y0))
        local_error_magnitude_max = (M_lagrange_max / 2) * max_abs_y_minus_y0 ** 2
        
        # Lower bound for k (must be an integer, so use ceil)
        k_lower_bound = np.ceil((domain_low - np.pi/2) / (2*np.pi))
        # Upper bound for k (must be an integer, so use floor)
        k_upper_bound = np.floor((domain_high - np.pi/2) / (2*np.pi))
        contains_crest = k_lower_bound <= k_upper_bound

        # If no trough, the minimum value of |sin(x)| is at the endpoints
        M_lagrange_min = np.minimum(-np.sin(domain_low), -np.sin(domain_high))
        M_lagrange_min[contains_crest] = -1.0  # If contains crest, min is -1.0
        M_lagrange_min = np.minimum(M_lagrange_min, 0.0)  # Ensure non-negative max

        max_abs_y_minus_y0 = np.maximum(np.abs(domain_low - y0), np.abs(domain_high - y0))
        local_error_magnitude_min = (M_lagrange_min / 2) * max_abs_y_minus_y0 ** 2

        prop_rem_lower_y, prop_rem_upper_y = a.remainder
        term1_rem = grad_f_y0 * prop_rem_lower_y
        term2_rem = grad_f_y0 * prop_rem_upper_y
        
        propagated_taylor_rem_lower = np.minimum(term1_rem, term2_rem)
        propagated_taylor_rem_upper = np.maximum(term1_rem, term2_rem)

        # --- Final summation (applies to all elements) ---
        final_rem_lower = propagated_taylor_rem_lower + local_error_magnitude_min
        final_rem_upper = propagated_taylor_rem_upper + local_error_magnitude_max

        # Apply global bounds tightening for sin function (range [-1, 1])
        temp_expansion = CertifiedFirstOrderTaylorExpansion(
            a.expansion_point,
            a.domain,
            (linear_term_jacobian, f_y0_val),
            (final_rem_lower, final_rem_upper)
        )
        
        clip_rem_lower, clip_rem_upper = apply_global_bounds_tightening(temp_expansion, -1.0, 1.0)
        
        # Intersect the Taylor bounds with the global range bounds
        final_rem_lower = np.maximum(final_rem_lower, clip_rem_lower)
        final_rem_upper = np.minimum(final_rem_upper, clip_rem_upper)

        remainder = (final_rem_lower, final_rem_upper)

        return CertifiedFirstOrderTaylorExpansion(
            a.expansion_point,
            a.domain,
            (linear_term_jacobian, f_y0_val),
            remainder
        )

    def cos(self, a: CertifiedFirstOrderTaylorExpansion):
        """
        Element-wise cosine
        :param a: CertifiedFirstOrderTaylorExpansion
        :return: CertifiedFirstOrderTaylorExpansion
        """
        y0 = a.linear_approximation[1]  # Center of expansion for y
        f_y0_val = np.cos(y0)         # cos(y0)
        grad_cos_y0 = -np.sin(y0)       # -sin(y0), derivative of cos(y) at y0

        linear_term_jacobian = grad_cos_y0.reshape(-1, 1) * a.linear_approximation[0]
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

        max_abs_y_minus_y0 = np.maximum(np.abs(domain_low - y0), np.abs(domain_high - y0))
        local_error_magnitude_max = (M_lagrange_max / 2) * max_abs_y_minus_y0 ** 2

        # Lower bound for k (must be an integer, so use ceil)
        k_lower_bound = np.ceil(domain_low / (2*np.pi))
        # Upper bound for k (must be an integer, so use floor)
        k_upper_bound = np.floor(domain_high/ (2*np.pi))
        contains_crest = k_lower_bound <= k_upper_bound

        # If no trough, the minimum value of -cos(x) is at the endpoints
        M_lagrange_min = np.minimum(-np.cos(domain_low), -np.cos(domain_high))
        M_lagrange_min[contains_crest] = -1.0  # If contains crest, min is -1.0
        M_lagrange_min = np.minimum(M_lagrange_min, 0.0)  # Ensure non-negative max

        max_abs_y_minus_y0 = np.maximum(np.abs(domain_low - y0), np.abs(domain_high - y0))
        local_error_magnitude_min = (M_lagrange_min / 2) * max_abs_y_minus_y0 ** 2

        # Propagate remainder through the derivative
        prop_rem_lower_y, prop_rem_upper_y = a.remainder
        term1_rem = grad_cos_y0 * prop_rem_lower_y
        term2_rem = grad_cos_y0 * prop_rem_upper_y

        propagated_taylor_rem_lower = np.minimum(term1_rem, term2_rem)
        propagated_taylor_rem_upper = np.maximum(term1_rem, term2_rem)

        # --- Final summation (applies to all elements) ---
        final_rem_lower = propagated_taylor_rem_lower + local_error_magnitude_min
        final_rem_upper = propagated_taylor_rem_upper + local_error_magnitude_max

        # Apply global bounds tightening for cos function (range [-1, 1])
        temp_expansion = CertifiedFirstOrderTaylorExpansion(
            a.expansion_point,
            a.domain,
            (linear_term_jacobian, f_y0_val),
            (final_rem_lower, final_rem_upper)
        )
        
        clip_rem_lower, clip_rem_upper = apply_global_bounds_tightening(temp_expansion, -1.0, 1.0)
        
        # Intersect the Taylor bounds with the global range bounds
        final_rem_lower = np.maximum(final_rem_lower, clip_rem_lower)
        final_rem_upper = np.minimum(final_rem_upper, clip_rem_upper)

        remainder = (final_rem_lower, final_rem_upper)

        return CertifiedFirstOrderTaylorExpansion(
            a.expansion_point,
            a.domain,
            (linear_term_jacobian, f_y0_val),
            remainder
        )

    def exp(self, a: CertifiedFirstOrderTaylorExpansion):
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

    def log(self, a: CertifiedFirstOrderTaylorExpansion):
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

        max_deviation = np.maximum(
            np.abs(lower - y0),
            np.abs(upper - y0)
        )

        # Compute the remainder bounds using the second derivative
        remainder_lower = (second_derivative_lower / 2) * max_deviation**2
        remainder_upper = (second_derivative_upper / 2) * max_deviation**2

        # We can add a clip to ensure non-positivity just in case of float error
        remainder_upper = np.minimum(remainder_upper, 0.0)
        remainder_lower = np.minimum(remainder_lower, remainder_upper)

        # Propagate the remainder through the derivative
        remainder1, remainder2 = grad_y0 * a.remainder[0], grad_y0 * a.remainder[1]
        propagated_rem_lower = np.minimum(remainder1, remainder2)
        propagated_rem_upper = np.maximum(remainder1, remainder2)

        # Combine propagated and second derivative remainders
        final_rem_lower = propagated_rem_lower + remainder_lower
        final_rem_upper = propagated_rem_upper + remainder_upper

        # Apply monotonic bounds tightening for log function
        # log is monotonically increasing, so use boundary values as global bounds
        log_at_boundaries = (np.log(lower), np.log(upper))
        
        # Create temporary Taylor expansion to use the monotonic tightening helper
        temp_expansion = CertifiedFirstOrderTaylorExpansion(
            a.expansion_point,
            a.domain,
            (linear_term, log_y0),
            (final_rem_lower, final_rem_upper)
        )
        
        clip_rem_lower, clip_rem_upper = apply_monotonic_bounds_tightening(temp_expansion, log_at_boundaries, is_increasing=True)
        
        # Intersect the Taylor bounds with the monotonic bounds
        final_rem_lower = np.maximum(final_rem_lower, clip_rem_lower)
        final_rem_upper = np.minimum(final_rem_upper, clip_rem_upper)

        return CertifiedFirstOrderTaylorExpansion(
            a.expansion_point,
            a.domain,
            (linear_term, log_y0),
            (final_rem_lower, final_rem_upper)
        )

    def sqrt(self, a: CertifiedFirstOrderTaylorExpansion):
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

        # Apply monotonic bounds tightening for sqrt function
        # sqrt is monotonically increasing and defined only for non-negative values
        sqrt_at_boundaries = (np.sqrt(range_lower), np.sqrt(range_upper))
        
        # Create temporary Taylor expansion to use the monotonic tightening helper
        temp_expansion = CertifiedFirstOrderTaylorExpansion(
            a.expansion_point,
            a.domain,
            (linear_term, sqrt_y0),
            (final_rem_lower, final_rem_upper)
        )
        
        clip_rem_lower, clip_rem_upper = apply_monotonic_bounds_tightening(temp_expansion, sqrt_at_boundaries, is_increasing=True)
        
        # Intersect the Taylor bounds with the monotonic bounds
        final_rem_lower = np.maximum(final_rem_lower, clip_rem_lower)
        final_rem_upper = np.minimum(final_rem_upper, clip_rem_upper)

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

        # Handle trivial cases
        if exponent_b == 0:
            # f(y) = 1, f'(y) = 0
            f_y0 = np.ones_like(a.linear_approximation[1])
            grad_f_y0 = np.zeros_like(a.linear_approximation[1])
            linear_term_jacobian = np.zeros_like(a.linear_approximation[0])
            local_error_magnitude_min = 0.0
            local_error_magnitude_max = 0.0
            remainder = (local_error_magnitude_min * np.ones_like(f_y0), local_error_magnitude_max * np.ones_like(f_y0))
            return CertifiedFirstOrderTaylorExpansion(
                a.expansion_point,
                a.domain,
                (linear_term_jacobian, f_y0),
                remainder
            )
        
        elif exponent_b == 1:
            # f(y) = y, f'(y) = 1
            return a

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
        
        # Apply monotonic bounds tightening for specific power functions
        range_of_y = a.range()
        if exponent_b > 0:
            # For positive exponents, x^n is monotonically increasing for x > 0
            if np.all(range_of_y[0] > 0):  # All values in range are positive
                power_at_boundaries = (np.power(range_of_y[0], exponent_b), np.power(range_of_y[1], exponent_b))
                
                # Create temporary Taylor expansion to use the monotonic tightening helper
                temp_expansion = CertifiedFirstOrderTaylorExpansion(
                    a.expansion_point,
                    a.domain,
                    (linear_term_jacobian, f_y0),
                    (final_rem_lower, final_rem_upper)
                )
                
                clip_rem_lower, clip_rem_upper = apply_monotonic_bounds_tightening(temp_expansion, power_at_boundaries, is_increasing=True)
                
                # Intersect the Taylor bounds with the monotonic bounds
                final_rem_lower = np.maximum(final_rem_lower, clip_rem_lower)
                final_rem_upper = np.minimum(final_rem_upper, clip_rem_upper)
        elif exponent_b < 0:
            # For negative exponents, x^n is monotonically decreasing for x > 0 (same as 1/x^|n|)
            if np.all(range_of_y[0] > 0):  # All values in range are positive
                power_at_boundaries = (np.power(range_of_y[0], exponent_b), np.power(range_of_y[1], exponent_b))
                
                # Create temporary Taylor expansion to use the monotonic tightening helper
                temp_expansion = CertifiedFirstOrderTaylorExpansion(
                    a.expansion_point,
                    a.domain,
                    (linear_term_jacobian, f_y0),
                    (final_rem_lower, final_rem_upper)
                )
                
                clip_rem_lower, clip_rem_upper = apply_monotonic_bounds_tightening(temp_expansion, power_at_boundaries, is_increasing=False)
                
                # Intersect the Taylor bounds with the monotonic bounds
                final_rem_lower = np.maximum(final_rem_lower, clip_rem_lower)
                final_rem_upper = np.minimum(final_rem_upper, clip_rem_upper)
        
        remainder = (final_rem_lower, final_rem_upper)

        return CertifiedFirstOrderTaylorExpansion(
            a.expansion_point,
            a.domain,
            (linear_term_jacobian, f_y0),
            remainder
        )

    def cat(self, a):
        """
        Stack a list of certified taylor expansions vertically

        :param a: list of certified taylor expansions

        :return: CertifiedFirstOrderTaylorExpansion
        """
        return self.stack(a)

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
        second_derivative_bounds: (M_min, M_max) bounds on f''(y) over the range of g(x).
                                  M_min and M_max are expected to be np.ndarray of shape (m,),
                                  where m is the output dimension of g(x).
        
    Returns:
        tuple: (remainder_lower, remainder_upper) bounds for the Lagrange remainder, np.ndarray of shape (m,).
    """
    M_min_f_double_prime, M_max_f_double_prime = second_derivative_bounds

    # g(x) is inner_taylor_expansion
    # g(x) approx g_c + J_g_c * (x - x_c) + R_g(x)
    # We need to bound S(x) = g(x) - g_c = J_g_c * (x - x_c) + R_g(x)
    
    J_g_c = inner_taylor_expansion.linear_approximation[0]  # Jacobian of g at center x_c
    R_g_lower, R_g_upper = inner_taylor_expansion.remainder # Remainder of g

    x_c = inner_taylor_expansion.expansion_point
    domain_lower_x, domain_upper_x = inner_taylor_expansion.domain

    # delta_x = x - x_c
    delta_x_lower = domain_lower_x - x_c
    delta_x_upper = domain_upper_x - x_c

    # Calculate interval for L(x) = J_g_c * delta_x
    # J_g_c has shape (m, n), delta_x_lower/upper have shape (n,)
    # L_lower/upper will have shape (m,)
    
    if J_g_c.ndim == 1: # If g is scalar-input, scalar-output, J_g_c might be scalar
        J_g_c = J_g_c.reshape(1, -1) if J_g_c.shape == delta_x_lower.shape else J_g_c.reshape(-1, 1)
        if J_g_c.shape[1] != delta_x_lower.shape[0]: # try to fix if it was (1,) and delta_x is (1,)
             J_g_c = J_g_c.T


    num_outputs_g = J_g_c.shape[0]
    L_lower = np.zeros(num_outputs_g)
    L_upper = np.zeros(num_outputs_g)

    for i in range(num_outputs_g):
        J_row_i = J_g_c[i, :]
        # Interval of J_row_i * delta_x_component
        # L_i_lower = sum_k (A_k * X_lower_k if A_k > 0 else A_k * X_upper_k)
        # L_i_upper = sum_k (A_k * X_upper_k if A_k > 0 else A_k * X_lower_k)
        L_lower[i] = np.sum(np.where(J_row_i > 0, J_row_i * delta_x_lower, J_row_i * delta_x_upper))
        L_upper[i] = np.sum(np.where(J_row_i > 0, J_row_i * delta_x_upper, J_row_i * delta_x_lower))

    # Interval for S(x) = g(x) - g_c = L(x) + R_g(x)
    S_min = L_lower + R_g_lower
    S_max = L_upper + R_g_upper

    # Interval for K(x) = S(x)^2 = (g(x) - g_c)^2
    # K_lower is 0 if S_min and S_max have different signs (or one is zero)
    # otherwise K_lower is min(S_min^2, S_max^2)
    K_lower = np.where(S_min * S_max <= 0, 0, np.minimum(S_min**2, S_max**2))
    K_upper = np.maximum(S_min**2, S_max**2)

    # Local error E_local = (1/2) * [M_min_f'', M_max_f''] * [K_lower, K_upper]
    # This is an interval product. For [a,b] * [c,d] where c,d >=0:
    # result is [min(ac,ad), max(bc,bd)] if a,b have same sign
    # or more generally [min(ac,ad,bc,bd), max(ac,ad,bc,bd)]
    
    # Ensure M_min_f_double_prime, M_max_f_double_prime, K_lower, K_upper are arrays
    M_min_f_double_prime = np.asarray(M_min_f_double_prime)
    M_max_f_double_prime = np.asarray(M_max_f_double_prime)
    K_lower = np.asarray(K_lower)
    K_upper = np.asarray(K_upper)

    # Products for interval multiplication:
    # M_min * K_lower, M_min * K_upper, M_max * K_lower, M_max * K_upper
    product_terms = np.array([
        M_min_f_double_prime * K_lower,
        M_min_f_double_prime * K_upper,
        M_max_f_double_prime * K_lower,
        M_max_f_double_prime * K_upper
    ])

    local_error_min = 0.5 * np.min(product_terms, axis=0)
    local_error_max = 0.5 * np.max(product_terms, axis=0)
    
    return local_error_min, local_error_max


def apply_global_bounds_tightening(taylor_expansion, global_lower_bound, global_upper_bound):
    """
    Apply post-processing step to tighten remainder bounds for functions with known global ranges.
    
    For functions f(y) with known bounds f(y) ∈ [L, U], we can clip the remainder R_f(y) 
    to the domain:
    R_f(y) ∈ [L - f_L(y), U - f_L(y)] ⊆ [L - max_y f_L(y), U - min_y f_L(y)]
    
    where f_L(y) is the linear approximation part of the Taylor expansion.
    
    Args:
        taylor_expansion (CertifiedFirstOrderTaylorExpansion): The Taylor expansion to tighten
        global_lower_bound (float): Known global lower bound L for the function
        global_upper_bound (float): Known global upper bound U for the function
        
    Returns:
        tuple: (tightened_remainder_lower, tightened_remainder_upper)
    """
    # Extract linear approximation components
    linear_jacobian, f_y0_val = taylor_expansion.linear_approximation
    
    # Get domain bounds
    x_domain_low, x_domain_high = taylor_expansion.domain
    x_0 = taylor_expansion.expansion_point
    
    # Find the interval for (x - x_0)
    delta_x_low = x_domain_low - x_0
    delta_x_high = x_domain_high - x_0

    # Compute range of f_L(x) using interval arithmetic (center-radius form)
    delta_x_center = (delta_x_low + delta_x_high) / 2.0
    delta_x_radius = (delta_x_high - delta_x_low) / 2.0
    
    # Range of J_f @ (x - x_0) = (J_f @ center) +/- (|J_f| @ radius)
    f_L_center_offset = linear_jacobian @ delta_x_center
    f_L_radius = np.abs(linear_jacobian) @ delta_x_radius
    
    # Total range of f_L(x) = f_0 + range(J_f @ (x - x_0))
    f_L_min = (f_y0_val + f_L_center_offset) - f_L_radius
    f_L_max = (f_y0_val + f_L_center_offset) + f_L_radius
    
    # Compute the new remainder bounds implied by the global range
    # R_f(x) >= global_lower_bound - f_L(x)
    # The constant lower bound must be <= the minimum of the right side:
    # R_min' <= min_x(global_lower_bound - f_L(x)) = global_lower_bound - max_x(f_L(x))
    clip_rem_lower = global_lower_bound - f_L_max
    
    # R_f(x) <= global_upper_bound - f_L(x)
    # The constant upper bound must be >= the maximum of the right side:
    # R_max' >= max_x(global_upper_bound - f_L(x)) = global_upper_bound - min_x(f_L(x))
    clip_rem_upper = global_upper_bound - f_L_min
    
    return clip_rem_lower, clip_rem_upper


def apply_monotonic_bounds_tightening(taylor_expansion, domain_range, is_increasing=True):
    """
    Apply post-processing step to tighten remainder bounds for monotonic functions.
    
    For monotonic functions, we can use the fact that extreme values occur at boundaries.
    We compute the actual function values at the boundaries and use them as global bounds.
    
    Args:
        taylor_expansion (CertifiedFirstOrderTaylorExpansion): The Taylor expansion to tighten
        domain_range (tuple): (f(domain_low), f(domain_high)) - function values at domain boundaries
        is_increasing (bool): True if function is monotonically increasing, False if decreasing
        
    Returns:
        tuple: (tightened_remainder_lower, tightened_remainder_upper)
    """
    f_at_low, f_at_high = domain_range
    
    if is_increasing:
        # For increasing functions: min = f(domain_low), max = f(domain_high)
        global_lower = f_at_low
        global_upper = f_at_high
    else:
        # For decreasing functions: min = f(domain_high), max = f(domain_low)
        global_lower = f_at_high
        global_upper = f_at_low
    
    return apply_global_bounds_tightening(taylor_expansion, global_lower, global_upper)


# Helper function for TE * TE multiplication remainder calculation
def _mat_interval_vec_mul(M, v_low, v_high):
    M_pos = np.maximum(M, 0)
    M_neg = np.minimum(M, 0)
    res_low = M_pos @ v_low + M_neg @ v_high
    res_high = M_pos @ v_high + M_neg @ v_low
    return res_low, res_high
