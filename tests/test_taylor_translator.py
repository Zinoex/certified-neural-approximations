import pytest
import numpy as np
import sys
import os

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from certified_neural_approximations.translators.taylor_translator import (
    CertifiedFirstOrderTaylorExpansion,
    TaylorTranslator,
    max_abs_sin,
    max_abs_cos,
    max_monomial_vectorized
)

class TestCertifiedFirstOrderTaylorExpansion:
    def setup_method(self):
        """Set up test fixtures."""
        self.translator = TaylorTranslator()
        self.expansion_point = np.array([1.0, 2.0])
        self.domain = (np.array([0.5, 1.5]), np.array([1.5, 2.5]))
        self.linear_approx = (np.array([[1.0, 0.5], [0.0, 1.0]]), np.array([1.0, 2.0]))
        self.remainder = (np.array([-0.1, -0.2]), np.array([0.1, 0.2]))
        
        self.te = CertifiedFirstOrderTaylorExpansion(
            self.expansion_point, self.domain, self.linear_approx, self.remainder
        )

    def test_initialization(self):
        """Test proper initialization of TaylorExpansion."""
        assert np.array_equal(self.te.expansion_point, self.expansion_point)
        assert np.array_equal(self.te.domain[0], self.domain[0])
        assert np.array_equal(self.te.domain[1], self.domain[1])
        assert np.array_equal(self.te.linear_approximation[0], self.linear_approx[0])
        assert np.array_equal(self.te.linear_approximation[1], self.linear_approx[1])
        assert np.array_equal(self.te.remainder[0], self.remainder[0])
        assert np.array_equal(self.te.remainder[1], self.remainder[1])

    def test_addition(self):
        """Test addition of two TaylorExpansions."""
        other = CertifiedFirstOrderTaylorExpansion(
            self.expansion_point, self.domain,
            (np.array([[0.5, 1.0], [1.0, 0.5]]), np.array([0.5, 1.0])),
            (np.array([-0.05, -0.1]), np.array([0.05, 0.1]))
        )
        
        result = self.te + other
        expected_linear = (
            self.linear_approx[0] + other.linear_approximation[0],
            self.linear_approx[1] + other.linear_approximation[1]
        )
        expected_remainder = (
            self.remainder[0] + other.remainder[0],
            self.remainder[1] + other.remainder[1]
        )
        
        assert np.allclose(result.linear_approximation[0], expected_linear[0])
        assert np.allclose(result.linear_approximation[1], expected_linear[1])
        assert np.allclose(result.remainder[0], expected_remainder[0])
        assert np.allclose(result.remainder[1], expected_remainder[1])
    
        # Test addition with scalar
        scalar = 5.0
        result = self.te + scalar
        
        assert np.allclose(result.linear_approximation[0], self.linear_approx[0])
        assert np.allclose(result.linear_approximation[1], self.linear_approx[1] + scalar)
        assert np.allclose(result.remainder[0], self.remainder[0])
        assert np.allclose(result.remainder[1], self.remainder[1])

    def test_subtraction(self):
        """Test subtraction of two TaylorExpansions."""
        other = CertifiedFirstOrderTaylorExpansion(
            self.expansion_point, self.domain,
            (np.array([[0.5, 1.0], [1.0, 0.5]]), np.array([0.5, 1.0])),
            (np.array([-0.05, -0.1]), np.array([0.05, 0.1]))
        )
        
        result = self.te - other
        expected_linear = (
            self.linear_approx[0] - other.linear_approximation[0],
            self.linear_approx[1] - other.linear_approximation[1]
        )
        expected_remainder = (
            self.remainder[0] - other.remainder[1],
            self.remainder[1] - other.remainder[0]
        )
        
        assert np.allclose(result.linear_approximation[0], expected_linear[0])
        assert np.allclose(result.linear_approximation[1], expected_linear[1])
        assert np.allclose(result.remainder[0], expected_remainder[0])
        assert np.allclose(result.remainder[1], expected_remainder[1])

        # Test subtraction with positive and negative scalar."""
        scalar = 2.0
        result = self.te * scalar
        
        assert np.allclose(result.linear_approximation[0], self.linear_approx[0] * scalar)
        assert np.allclose(result.linear_approximation[1], self.linear_approx[1] * scalar)
        assert np.allclose(result.remainder[0], self.remainder[0] * scalar)
        assert np.allclose(result.remainder[1], self.remainder[1] * scalar)

        # Test with negative scalar
        scalar = -2.0
        result = self.te * scalar
        
        assert np.allclose(result.linear_approximation[0], self.linear_approx[0] * scalar)
        assert np.allclose(result.linear_approximation[1], self.linear_approx[1] * scalar)
        assert np.allclose(result.remainder[0], self.remainder[1] * scalar)
        assert np.allclose(result.remainder[1], self.remainder[0] * scalar)

    def test_multiplication(self):
        """Test multiplication of two TaylorExpansions using product rule."""
        # Test multiplication with scalar (positive)
        scalar = 2.0
        result = self.te * scalar
        
        assert np.allclose(result.linear_approximation[0], self.linear_approx[0] * scalar)
        assert np.allclose(result.linear_approximation[1], self.linear_approx[1] * scalar)
        assert np.allclose(result.remainder[0], self.remainder[0] * scalar)
        assert np.allclose(result.remainder[1], self.remainder[1] * scalar)

        # Test multiplication with negative scalar
        scalar = -2.0
        result = self.te * scalar
        
        assert np.allclose(result.linear_approximation[0], self.linear_approx[0] * scalar)
        assert np.allclose(result.linear_approximation[1], self.linear_approx[1] * scalar)
        assert np.allclose(result.remainder[0], self.remainder[1] * scalar)
        assert np.allclose(result.remainder[1], self.remainder[0] * scalar)

        # Test multiplication of two TaylorExpansions
        other = CertifiedFirstOrderTaylorExpansion(
            self.expansion_point, self.domain,
            (np.array([[0.5, 1.0], [1.0, 0.5]]), np.array([0.5, 1.0])),
            (np.array([-0.05, -0.1]), np.array([0.05, 0.1]))
        )
        
        result = self.te * other
        
        # Expected values using product rule: f*g = f(c)*g(c) + [f(c)*∇g + g(c)*∇f]*(x-c) + HOT
        y0_self = self.linear_approx[1]  # f(c)
        J_self = self.linear_approx[0]   # ∇f(c)
        y0_other = other.linear_approximation[1]  # g(c)
        J_other = other.linear_approximation[0]   # ∇g(c)
        
        # Product rule for constant: f(c) * g(c)
        expected_constant = y0_self * y0_other
        
        # Product rule for Jacobian: f(c)*∇g(c) + g(c)*∇f(c)
        expected_jacobian = (y0_self.reshape(-1, 1) * J_other + 
                           y0_other.reshape(-1, 1) * J_self)
        
        assert np.allclose(result.linear_approximation[1], expected_constant)
        assert np.allclose(result.linear_approximation[0], expected_jacobian)
        
        # Check that expansion point and domain are preserved
        assert np.array_equal(result.expansion_point, self.expansion_point)
        assert np.array_equal(result.domain[0], self.domain[0])
        assert np.array_equal(result.domain[1], self.domain[1])
        
        # Check that remainders are properly computed (should be non-zero due to propagation and HOT)
        assert result.remainder[0] is not None
        assert result.remainder[1] is not None
        assert len(result.remainder[0]) == len(self.expansion_point)
        assert len(result.remainder[1]) == len(self.expansion_point)

    def test_composite_functions(self):
        """Test composition of functions using chain rule."""
        # Create an inner function: g(x) = 2x + 1
        inner_te = CertifiedFirstOrderTaylorExpansion(
            self.expansion_point, self.domain,
            (np.array([[2.0, 0.0], [0.0, 2.0]]), np.array([3.0, 5.0])),  # g(c) = 2*[1,2] + 1 = [3,5]
            (np.array([-0.02, -0.03]), np.array([0.02, 0.03]))
        )
        
        # Apply outer function: f(g(x)) = sin(g(x))
        # Chain rule: d/dx[sin(g(x))] = cos(g(x)) * g'(x)
        result = self.translator.sin(inner_te)
        
        # Manual chain rule calculation
        g_c = inner_te.linear_approximation[1]  # g(c) = [3, 5]
        g_prime_c = inner_te.linear_approximation[0]  # g'(c) = [[2,0],[0,2]]
        
        # Expected constant: sin(g(c))
        expected_constant = np.sin(g_c)
        
        # Expected Jacobian: cos(g(c)) * g'(c)
        expected_jacobian = np.cos(g_c).reshape(-1, 1) * g_prime_c
        
        # Verify chain rule application
        assert np.allclose(result.linear_approximation[1], expected_constant)
        assert np.allclose(result.linear_approximation[0], expected_jacobian)
        
        # Test another composition: h(x) = exp(sin(g(x)))
        # This tests nested composition
        intermediate = self.translator.sin(inner_te)
        final_result = self.translator.exp(intermediate)
        
        # Manual calculation for h(x) = exp(sin(g(x)))
        # h'(x) = exp(sin(g(x))) * cos(g(x)) * g'(x)
        sin_g_c = np.sin(g_c)
        cos_g_c = np.cos(g_c)
        
        expected_final_constant = np.exp(sin_g_c)
        expected_final_jacobian = (np.exp(sin_g_c) * cos_g_c).reshape(-1, 1) * g_prime_c
        
        assert np.allclose(final_result.linear_approximation[1], expected_final_constant)
        assert np.allclose(final_result.linear_approximation[0], expected_final_jacobian)
        
        # Verify structure preservation through composition
        assert np.array_equal(final_result.expansion_point, self.expansion_point)
        assert np.array_equal(final_result.domain[0], self.domain[0])
        assert np.array_equal(final_result.domain[1], self.domain[1])
        
        # Check that remainders accumulate properly through composition
        assert final_result.remainder[0] is not None
        assert final_result.remainder[1] is not None
        assert len(final_result.remainder[0]) == len(self.expansion_point)
        assert len(final_result.remainder[1]) == len(self.expansion_point)
        assert np.all(final_result.remainder[0] <= final_result.remainder[1] + 1e-9)

    def test_division(self):
        """Test division by scalar."""
        scalar = 2.0
        result = self.te / scalar
        
        assert np.allclose(result.linear_approximation[0], self.linear_approx[0] / scalar)
        assert np.allclose(result.linear_approximation[1], self.linear_approx[1] / scalar)
        assert np.allclose(result.remainder[0], self.remainder[0] / scalar)
        assert np.allclose(result.remainder[1], self.remainder[1] / scalar)

        # Test division by zero
        with pytest.raises(ZeroDivisionError):
            self.te / 0

        # Test division of two TaylorExpansions
        #    - Create a positive TaylorExpansion for the denominator to avoid domain issues
        #    - We need to ensure the range of the denominator doesn't contain zero to avoid
        #    - division by zero errors in the reciprocal computation
        denominator = CertifiedFirstOrderTaylorExpansion(
            self.expansion_point, self.domain,
            (np.array([[0.5, 0.2], [0.1, 0.8]]), np.array([2.0, 3.0])),  # Positive constants
            (np.array([-0.05, -0.1]), np.array([0.05, 0.1]))  # Small remainders
        )
        
        # Perform division: self.te / denominator
        # Mathematically: f(x) / g(x) = f(x) * (1/g(x))
        result = self.te / denominator
        
        # Test 1: Division is implemented as multiplication by reciprocal
        # Verify this by computing the expected result manually
        expected_result_reciprocal = self.te * denominator._reciprocal()
        
        # Check that the linear approximation (Jacobian and constant) match
        # For division f/g, the result should be identical to f * (1/g)
        assert np.allclose(result.linear_approximation[0], expected_result_reciprocal.linear_approximation[0])
        assert np.allclose(result.linear_approximation[1], expected_result_reciprocal.linear_approximation[1])
        
        # Check that the remainder bounds are correctly propagated
        # The remainder captures higher-order terms and error bounds
        assert np.allclose(result.remainder[0], expected_result_reciprocal.remainder[0])
        assert np.allclose(result.remainder[1], expected_result_reciprocal.remainder[1])
        
        # Test 2: Verify using manual product rule calculation
        # For division f/g = f * (1/g), we manually compute the product rule
        J_f, y0_f = self.te.linear_approximation  # f components
        J_g, y0_g = denominator.linear_approximation  # g components
        
        # Reciprocal components: 1/g(c) and ∇(1/g)(c) = -1/g(c)² * ∇g(c)
        recip_constant = 1.0 / y0_g  # 1/g(c)
        recip_jacobian = -(1.0 / (y0_g**2)).reshape(-1, 1) * J_g  # ∇(1/g)(c)
        
        # Product rule: f * (1/g) = f(c)*(1/g(c)) + [f(c)*∇(1/g)(c) + (1/g(c))*∇f(c)]
        expected_constant_manual = y0_f * recip_constant
        expected_jacobian_manual = (y0_f.reshape(-1, 1) * recip_jacobian + 
                                   recip_constant.reshape(-1, 1) * J_f)
        
        # Verify manual calculation matches the result
        assert np.allclose(result.linear_approximation[1], expected_constant_manual)
        assert np.allclose(result.linear_approximation[0], expected_jacobian_manual)
        
        # Verify that the mathematical structure is preserved
        # Expansion point and domain should remain unchanged
        assert np.array_equal(result.expansion_point, self.expansion_point)
        assert np.array_equal(result.domain[0], self.domain[0])
        assert np.array_equal(result.domain[1], self.domain[1])

    def test_reciprocal(self):
        """Test reciprocal operation."""
        # Create a positive TaylorExpansion to avoid domain issues
        te_pos = CertifiedFirstOrderTaylorExpansion(
            np.array([2.0]), (np.array([1.5]), np.array([2.5])),
            (np.array([[1.0]]), np.array([2.0])),
            (np.array([-0.1]), np.array([0.1]))
        )
        
        result = te_pos._reciprocal()
        assert result.linear_approximation[1] == pytest.approx(0.5)  # 1/2
        assert result.linear_approximation[0] == pytest.approx(-0.25)  # -1/(2^2)

    def test_indexing(self):
        """Test indexing operation."""
        result = self.te[0]
        
        assert np.allclose(result.linear_approximation[0], self.linear_approx[0][0:1])
        assert np.allclose(result.linear_approximation[1], self.linear_approx[1][0:1])
        assert np.allclose(result.remainder[0], self.remainder[0][0:1])
        assert np.allclose(result.remainder[1], self.remainder[1][0:1])

    def test_range_computation(self):
        """Test range computation."""
        lower, upper = self.te.range()
        
        # Should return valid bounds
        assert len(lower) == len(self.expansion_point)
        assert len(upper) == len(self.expansion_point)
        assert np.all(lower <= upper + 1e-9)

    def test_monotonic_functions(self):
        """Test bounds for monotonic functions using second derivative evaluation at endpoints."""
        # Test a monotonically increasing function: f(x) = x^3
        exponent = 3
        te_increasing = CertifiedFirstOrderTaylorExpansion(
            self.expansion_point, self.domain,
            (np.array([[3.0, 0.0], [0.0, 3.0]]), np.array([1.0, 8.0])),  # f(c) = [1^3, 2^3]
            (np.array([-0.1, -0.2]), np.array([0.1, 0.2]))
        )
        
        # Compute the second derivative bounds for f(x) = x^3
        # f''(x) = 6x, so evaluate at the endpoints of the domain
        lower_bound = self.domain[0]  # x_min
        upper_bound = self.domain[1]  # x_max
        second_derivative_lower = 6 * lower_bound
        second_derivative_upper = 6 * upper_bound
        
        # If f''(x) is increasing, max is at x_max
        max_second_derivative = second_derivative_upper
        min_second_derivative = second_derivative_lower
        
        # Verify that the remainder bounds are consistent with the second derivative
        max_deviation = np.maximum(np.abs(self.domain[0] - self.expansion_point), 
                                   np.abs(self.domain[1] - self.expansion_point))
        expected_remainder_bound_upper = (max_second_derivative / 2) * max_deviation**2
        expected_remainder_bound_lower = (min_second_derivative / 2) * max_deviation**2

        assert np.all(te_increasing.remainder[1] == expected_remainder_bound_upper)
        assert np.all(te_increasing.remainder[0] == expected_remainder_bound_lower)

        # Test a monotonically decreasing function: f(x) = -x^3
        te_decreasing = CertifiedFirstOrderTaylorExpansion(
            self.expansion_point, self.domain,
            (np.array([[-3.0, 0.0], [0.0, -3.0]]), np.array([-1.0, -8.0])),  # f(c) = -[1^3, 2^3]
            (np.array([-0.1, -0.2]), np.array([0.1, 0.2]))
        )
        
        # Compute the second derivative bounds for f(x) = -x^3
        # f''(x) = -6x, so evaluate at the endpoints of the domain
        second_derivative_lower = -6 * lower_bound
        second_derivative_upper = -6 * upper_bound
        
        # If f''(x) is decreasing, min is at x_max (negative max)
        min_second_derivative = second_derivative_upper
        max_second_derivative = second_derivative_lower
        
        # Verify that the remainder bounds are consistent with the second derivative
        max_deviation = np.maximum(np.abs(self.domain[0] - self.expansion_point), 
                                   np.abs(self.domain[1] - self.expansion_point))
        expected_remainder_bound_upper = (max_second_derivative / 2) * max_deviation**2
        expected_remainder_bound_lower = (min_second_derivative / 2) * max_deviation**2

        assert np.all(te_increasing.remainder[1] == expected_remainder_bound_upper)
        assert np.all(te_increasing.remainder[0] == expected_remainder_bound_lower)

class TestTaylorTranslator:
    def setup_method(self):
        """Set up test fixtures."""
        self.translator = TaylorTranslator()
        self.expansion_point = np.array([1.0])
        self.domain = (np.array([0.5]), np.array([1.5]))
        self.te = CertifiedFirstOrderTaylorExpansion(
            self.expansion_point, self.domain,
            (np.array([[1.0]]), np.array([1.0])),
            (np.array([-0.1]), np.array([0.1]))
        )

    def test_to_format(self):
        """Test initialization of identity TaylorExpansion."""
        point = np.array([1.0, 2.0])
        lower = np.array([0.5, 1.5])
        upper = np.array([1.5, 2.5])
        
        result = self.translator.to_format(point, lower, upper)
        
        assert np.array_equal(result.expansion_point, point)
        assert np.array_equal(result.domain[0], lower)
        assert np.array_equal(result.domain[1], upper)
        # For identity function f(x) = x, the Jacobian should be the identity matrix
        assert np.array_equal(result.linear_approximation[0], np.eye(point.shape[0]))
        assert np.array_equal(result.linear_approximation[1], point)
        assert np.array_equal(result.remainder[0], np.zeros(point.shape[0]))
        assert np.array_equal(result.remainder[1], np.zeros(point.shape[0]))

    def test_matrix_vector(self):
        """Test matrix-vector multiplication."""
        matrix = np.array([[2.0, 1.0], [1.0, 3.0]])
        te_2d = CertifiedFirstOrderTaylorExpansion(
            np.array([1.0, 2.0]), (np.array([0.5, 1.5]), np.array([1.5, 2.5])),
            (np.array([[1.0, 0.5], [0.0, 1.0]]), np.array([1.0, 2.0])),
            (np.array([-0.1, -0.2]), np.array([0.1, 0.2]))
        )
        
        result = self.translator.matrix_vector(matrix, te_2d)
        
        expected_linear = matrix @ te_2d.linear_approximation[0]
        expected_constant = matrix @ te_2d.linear_approximation[1]
        
        assert np.allclose(result.linear_approximation[0], expected_linear)
        assert np.allclose(result.linear_approximation[1], expected_constant)

    def test_sin(self):
        """Test sine function."""
        result = self.translator.sin(self.te)
        
        expected_constant = np.sin(self.expansion_point)
        expected_jacobian = np.cos(self.expansion_point).reshape(-1, 1) * self.te.linear_approximation[0]
        
        assert np.allclose(result.linear_approximation[1], expected_constant)
        assert np.allclose(result.linear_approximation[0], expected_jacobian)

    def test_cos(self):
        """Test cosine function."""
        result = self.translator.cos(self.te)
        
        expected_constant = np.cos(self.expansion_point)
        expected_jacobian = -np.sin(self.expansion_point).reshape(-1, 1) * self.te.linear_approximation[0]
        
        assert np.allclose(result.linear_approximation[1], expected_constant)
        assert np.allclose(result.linear_approximation[0], expected_jacobian)

    def test_exp(self):
        """Test exponential function."""
        result = self.translator.exp(self.te)
        
        expected_constant = np.exp(self.expansion_point)
        expected_jacobian = np.exp(self.expansion_point).reshape(-1, 1) * self.te.linear_approximation[0]
        
        assert np.allclose(result.linear_approximation[1], expected_constant)
        assert np.allclose(result.linear_approximation[0], expected_jacobian)

    def test_log(self):
        """Test logarithm function."""
        # Use positive values to avoid domain issues
        te_pos = CertifiedFirstOrderTaylorExpansion(
            np.array([2.0]), (np.array([1.5]), np.array([2.5])),
            (np.array([[1.0]]), np.array([2.0])),
            (np.array([-0.1]), np.array([0.1]))
        )
        
        result = self.translator.log(te_pos)
        
        expected_constant = np.log(np.array([2.0]))
        expected_jacobian = (1.0 / np.array([2.0])).reshape(-1, 1) * te_pos.linear_approximation[0]
        
        assert np.allclose(result.linear_approximation[1], expected_constant)
        assert np.allclose(result.linear_approximation[0], expected_jacobian)

    def test_log_domain_error(self):
        """Test logarithm with invalid domain."""
        # Create TaylorExpansion with range including negative values
        te_invalid = CertifiedFirstOrderTaylorExpansion(
            np.array([0.5]), (np.array([-1.0]), np.array([1.0])),
            (np.array([[1.0]]), np.array([0.5])),
            (np.array([-1.0]), np.array([1.0]))
        )
        
        with pytest.raises(ValueError, match="Logarithm domain error"):
            self.translator.log(te_invalid)

    def test_sqrt(self):
        """Test square root function."""
        te_pos = CertifiedFirstOrderTaylorExpansion(
            np.array([4.0]), (np.array([1.0]), np.array([9.0])),
            (np.array([[1.0]]), np.array([4.0])),
            (np.array([-0.1]), np.array([0.1]))
        )
        
        result = self.translator.sqrt(te_pos)
        
        expected_constant = np.sqrt(np.array([4.0]))
        expected_jacobian = (0.5 / np.array([4.0])).reshape(-1, 1) * te_pos.linear_approximation[0]
        
        assert np.allclose(result.linear_approximation[1], expected_constant)
        assert np.allclose(result.linear_approximation[0], expected_jacobian)

    def test_sqrt_domain_error(self):
        """Test square root with invalid domain."""
        te_invalid = CertifiedFirstOrderTaylorExpansion(
            np.array([0.5]), (np.array([-1.0]), np.array([1.0])),
            (np.array([[1.0]]), np.array([0.5])),
            (np.array([-1.0]), np.array([1.0]))
        )
        
        with pytest.raises(ValueError, match="Square root domain error"):
            self.translator.sqrt(te_invalid)

    def test_pow(self):
        """Test power function."""
        exponent = 3
        result = self.translator.pow(self.te, exponent)
        
        expected_constant = np.pow(self.expansion_point, exponent)
        expected_jacobian = (exponent * np.pow(self.expansion_point, exponent - 1)).reshape(-1, 1) * self.te.linear_approximation[0]
        
        assert np.allclose(result.linear_approximation[1], expected_constant)
        assert np.allclose(result.linear_approximation[0], expected_jacobian)

    def test_pow_invalid_exponent(self):
        """Test power function with non-integer exponent."""
        with pytest.raises(AssertionError):
            self.translator.pow(self.te, 2.5)

    def test_stack(self):
        """Test stacking multiple TaylorExpansions."""
        te1 = CertifiedFirstOrderTaylorExpansion(
            self.expansion_point, self.domain,
            (np.array([[1.0]]), np.array([1.0])),
            (np.array([-0.1]), np.array([0.1]))
        )
        te2 = CertifiedFirstOrderTaylorExpansion(
            self.expansion_point, self.domain,
            (np.array([[2.0]]), np.array([2.0])),
            (np.array([-0.2]), np.array([0.2]))
        )
        
        result = self.translator.stack([te1, te2])
        
        expected_jacobian = np.concatenate([te1.linear_approximation[0], te2.linear_approximation[0]])
        expected_constant = np.concatenate([te1.linear_approximation[1], te2.linear_approximation[1]])
        expected_remainder_low = np.concatenate([te1.remainder[0], te2.remainder[0]])
        expected_remainder_high = np.concatenate([te1.remainder[1], te2.remainder[1]])
        
        assert np.allclose(result.linear_approximation[0], expected_jacobian)
        assert np.allclose(result.linear_approximation[1], expected_constant)
        assert np.allclose(result.remainder[0], expected_remainder_low)
        assert np.allclose(result.remainder[1], expected_remainder_high)


class TestHelperFunctions:
    def test_max_abs_sin(self):
        """Test maximum absolute sine function."""
        # Test intervals without peaks
        intervals = (np.array([0.1, 2.0]), np.array([0.5, 2.5]))
        result = max_abs_sin(intervals)
        
        assert len(result) == 2
        assert all(0 <= val <= 1 for val in result)
        
        # Test interval containing peak
        intervals_peak = (np.array([0.0]), np.array([np.pi]))
        result_peak = max_abs_sin(intervals_peak)
        assert result_peak[0] == pytest.approx(1.0)

    def test_max_abs_cos(self):
        """Test maximum absolute cosine function."""
        # Test intervals without peaks
        intervals = (np.array([0.5, 2.0]), np.array([1.0, 2.5]))
        result = max_abs_cos(intervals)
        
        assert len(result) == 2
        assert all(0 <= val <= 1 for val in result)
        
        # Test interval containing peak
        intervals_peak = (np.array([-0.5]), np.array([0.5]))
        result_peak = max_abs_cos(intervals_peak)
        assert result_peak[0] == pytest.approx(1.0)

    def test_max_monomial_vectorized(self):
        """Test maximum monomial function."""
        c = np.array([1.0, -2.0])
        n = np.array([2, 3])
        intervals = (np.array([-1.0, -2.0]), np.array([1.0, 2.0]))
        
        result = max_monomial_vectorized(c, n, intervals)
        
        assert len(result) == 2
        assert all(val >= 0 for val in result)
        
        # For x^2 on [-1, 1], max |f(x)| = 1
        assert result[0] == pytest.approx(1.0)
        # For -2*x^3 on [-2, 2], max |f(x)| = 16
        assert result[1] == pytest.approx(16.0)


if __name__ == "__main__":
    """
    Run tests directly with:
    python tests/test_taylor_translator.py
    
    Or run with pytest:
    pytest tests/test_taylor_translator.py
    pytest tests/test_taylor_translator.py -v  # verbose output
    pytest tests/test_taylor_translator.py::TestCertifiedFirstOrderTaylorExpansion  # specific class
    pytest tests/test_taylor_translator.py::TestCertifiedFirstOrderTaylorExpansion::test_initialization  # specific test
    
    Or run all tests in the tests directory:
    pytest tests/
    """
    pytest.main([__file__, "-v"])
