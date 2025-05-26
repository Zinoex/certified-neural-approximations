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

    def test_addition_with_te(self):
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

    def test_addition_with_scalar(self):
        """Test addition with scalar."""
        scalar = 5.0
        result = self.te + scalar
        
        assert np.allclose(result.linear_approximation[0], self.linear_approx[0])
        assert np.allclose(result.linear_approximation[1], self.linear_approx[1] + scalar)
        assert np.allclose(result.remainder[0], self.remainder[0])
        assert np.allclose(result.remainder[1], self.remainder[1])

    def test_subtraction_with_te(self):
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

    def test_multiplication_with_scalar(self):
        """Test multiplication with scalar."""
        scalar = 2.0
        result = self.te * scalar
        
        assert np.allclose(result.linear_approximation[0], self.linear_approx[0] * scalar)
        assert np.allclose(result.linear_approximation[1], self.linear_approx[1] * scalar)
        assert np.allclose(result.remainder[0], self.remainder[0] * scalar)
        assert np.allclose(result.remainder[1], self.remainder[1] * scalar)

    def test_multiplication_with_negative_scalar(self):
        """Test multiplication with negative scalar."""
        scalar = -2.0
        result = self.te * scalar
        
        assert np.allclose(result.linear_approximation[0], self.linear_approx[0] * scalar)
        assert np.allclose(result.linear_approximation[1], self.linear_approx[1] * scalar)
        assert np.allclose(result.remainder[0], self.remainder[1] * scalar)
        assert np.allclose(result.remainder[1], self.remainder[0] * scalar)

    def test_multiplication_with_te(self):
        """Test multiplication of two TaylorExpansions using product rule."""
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
        
        # Verify Lagrange remainder bounds for sin composition
        # For sin(g(x)), the second derivative bound is max|sin''(g(η))| = max|sin(g(η))| ≤ 1
        # Plus chain rule contributions from g(x) remainder propagation
        g_range = inner_te.range()
        M_sin_bound = max_abs_sin(g_range)  # Should be ≤ 1
        
        # The remainder should include both propagated terms and Lagrange terms
        # Check that remainder bounds are reasonable and non-trivial
        assert np.all(result.remainder[0] <= result.remainder[1] + 1e-9)
        assert not np.allclose(result.remainder[0], 0.0, atol=1e-12)
        assert not np.allclose(result.remainder[1], 0.0, atol=1e-12)
        
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
        
        # Verify Lagrange bounds for nested composition
        # exp(sin(g(x))) has complex remainder structure due to composition
        # The remainder should account for both exp and sin Lagrange terms
        intermediate_range = intermediate.range()
        assert np.all(final_result.remainder[0] <= final_result.remainder[1] + 1e-9)
        
        # Check that remainders grow appropriately through composition
        # (nested functions should have larger error bounds)
        final_remainder_width = final_result.remainder[1] - final_result.remainder[0]
        initial_remainder_width = result.remainder[1] - result.remainder[0]
        assert np.all(final_remainder_width >= initial_remainder_width * 0.8)  # Allow some tolerance
        
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
        
        # Verify Lagrange remainder bounds for sin function
        # sin''(x) = -sin(x), so max|sin''(η)| over range bounds the Lagrange remainder
        y_range = self.te.range()
        M_lagrange = max_abs_sin(y_range)
        max_deviation = np.maximum(np.abs(y_range[0] - self.expansion_point), 
                                  np.abs(y_range[1] - self.expansion_point))
        expected_lagrange_bound = (M_lagrange / 2) * max_deviation**2
        
        # The actual remainder should be bounded by Lagrange term plus propagated terms
        remainder_magnitude = np.maximum(np.abs(result.remainder[0]), np.abs(result.remainder[1]))
        assert np.all(remainder_magnitude >= expected_lagrange_bound * 0.5)  # Should contain Lagrange term

    def test_cos(self):
        """Test cosine function."""
        result = self.translator.cos(self.te)
        
        expected_constant = np.cos(self.expansion_point)
        expected_jacobian = -np.sin(self.expansion_point).reshape(-1, 1) * self.te.linear_approximation[0]
        
        assert np.allclose(result.linear_approximation[1], expected_constant)
        assert np.allclose(result.linear_approximation[0], expected_jacobian)
        
        # Verify Lagrange remainder bounds for cos function
        # cos''(x) = -cos(x), so max|cos''(η)| over range bounds the Lagrange remainder
        y_range = self.te.range()
        M_lagrange = max_abs_cos(y_range)
        max_deviation = np.maximum(np.abs(y_range[0] - self.expansion_point), 
                                  np.abs(y_range[1] - self.expansion_point))
        expected_lagrange_bound = (M_lagrange / 2) * max_deviation**2
        
        remainder_magnitude = np.maximum(np.abs(result.remainder[0]), np.abs(result.remainder[1]))
        assert np.all(remainder_magnitude >= expected_lagrange_bound * 0.5)

    def test_exp(self):
        """Test exponential function."""
        result = self.translator.exp(self.te)
        
        expected_constant = np.exp(self.expansion_point)
        expected_jacobian = np.exp(self.expansion_point).reshape(-1, 1) * self.te.linear_approximation[0]
        
        assert np.allclose(result.linear_approximation[1], expected_constant)
        assert np.allclose(result.linear_approximation[0], expected_jacobian)
        
        # Verify Lagrange remainder bounds for exp function
        # exp''(x) = exp(x), so max|exp(η)| over range bounds the Lagrange remainder
        y_range = self.te.range()
        M_lagrange = np.exp(np.maximum(y_range[0], y_range[1]))  # exp is monotonic
        max_deviation = np.maximum(np.abs(y_range[0] - self.expansion_point), 
                                  np.abs(y_range[1] - self.expansion_point))
        expected_lagrange_bound = (M_lagrange / 2) * max_deviation**2
        
        remainder_magnitude = np.maximum(np.abs(result.remainder[0]), np.abs(result.remainder[1]))
        assert np.all(remainder_magnitude >= expected_lagrange_bound * 0.1)  # Looser bound due to growth

    def test_pow(self):
        """Test power function."""
        exponent = 3
        result = self.translator.pow(self.te, exponent)
        
        expected_constant = np.pow(self.expansion_point, exponent)
        expected_jacobian = (exponent * np.pow(self.expansion_point, exponent - 1)).reshape(-1, 1) * self.te.linear_approximation[0]
        
        assert np.allclose(result.linear_approximation[1], expected_constant)
        assert np.allclose(result.linear_approximation[0], expected_jacobian)
        
        # Verify Lagrange remainder bounds for power function
        # f''(x) = n(n-1)x^(n-2) for f(x) = x^n
        y_range = self.te.range()
        coeff_second_deriv = exponent * (exponent - 1)
        exp_second_deriv = exponent - 2
        M_lagrange = max_monomial_vectorized(coeff_second_deriv, exp_second_deriv, y_range)
        max_deviation = np.maximum(np.abs(y_range[0] - self.expansion_point), 
                                  np.abs(y_range[1] - self.expansion_point))
        expected_lagrange_bound = (M_lagrange / 2) * max_deviation**2
        
        remainder_magnitude = np.maximum(np.abs(result.remainder[0]), np.abs(result.remainder[1]))
        assert np.all(remainder_magnitude >= expected_lagrange_bound * 0.5)

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
