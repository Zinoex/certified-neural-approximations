import pytest
import numpy as np
import sys
import os

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from certified_neural_approximations.translators.taylor_translator import (
    CertifiedFirstOrderTaylorExpansion,
    TaylorTranslator
)
from certified_neural_approximations.translators.numpy_translator import NumpyTranslator

class TestCertifiedFirstOrderTaylorExpansion:
    def setup_method(self):
        """Set up test fixtures."""
        self.translator = TaylorTranslator()
        self.expansion_point = np.array([1.0, 2.0])
        self.domain = (np.array([0.5, 1.5]), np.array([1.5, 2.5]))

        # Start with f(x) = x
        self.linear_approx = (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([1.0, 2.0]))
        self.remainder = (np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        
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
        te_increasing = self.translator.pow(self.te, exponent)
        
        # Verify that the first-order approximation with the remainder term contains x^3 on the interval
        x_test = np.linspace(self.domain[0], self.domain[1], 1000)  # Test points
        pow_x = np.power(x_test, exponent)  # True power values
        approx_with_remainder_lower, approx_with_remainder_upper = self.compute_approximation_bounds(
            te_increasing, x_test, self.expansion_point
        )

        # Verify that the true function is contained within the bounds
        assert np.all(pow_x >= approx_with_remainder_lower)
        assert np.all(pow_x <= approx_with_remainder_upper)
        
        # Verify that remainder bounds are properly ordered
        assert np.all(te_increasing.remainder[0] <= te_increasing.remainder[1] + 1e-9)

        # Test a monotonically decreasing function: f(x) = -x^3
        te_decreasing = -te_increasing
        
        # Verify that the first-order approximation with the remainder term contains -x^3 on the interval
        neg_pow_x = -pow_x  # True negative power values
        approx_with_remainder_lower_neg, approx_with_remainder_upper_neg = self.compute_approximation_bounds(
            te_decreasing, x_test, self.expansion_point
        )

        # Verify that the true function is contained within the bounds
        assert np.all(neg_pow_x >= approx_with_remainder_lower_neg)
        assert np.all(neg_pow_x <= approx_with_remainder_upper_neg)
        
        # Verify that remainder bounds are properly ordered
        assert np.all(te_decreasing.remainder[0] <= te_decreasing.remainder[1])

    def test_negation(self):
        """Test negation of a Taylor expansion."""
        negated_te = -self.te

        # Check that the linear approximation is negated
        assert np.allclose(negated_te.linear_approximation[0], -self.te.linear_approximation[0])
        assert np.allclose(negated_te.linear_approximation[1], -self.te.linear_approximation[1])

        # Check that the remainder bounds are swapped and negated
        assert np.allclose(negated_te.remainder[0], -self.te.remainder[1])
        assert np.allclose(negated_te.remainder[1], -self.te.remainder[0])

        # Ensure the expansion point and domain remain unchanged
        assert np.array_equal(negated_te.expansion_point, self.te.expansion_point)
        assert np.array_equal(negated_te.domain[0], self.te.domain[0])
        assert np.array_equal(negated_te.domain[1], self.te.domain[1])

        negated_te = (-1) * self.te

        # Check that the linear approximation is negated
        assert np.allclose(negated_te.linear_approximation[0], -self.te.linear_approximation[0])
        assert np.allclose(negated_te.linear_approximation[1], -self.te.linear_approximation[1])

        # Check that the remainder bounds are swapped and negated
        assert np.allclose(negated_te.remainder[0], -self.te.remainder[1])
        assert np.allclose(negated_te.remainder[1], -self.te.remainder[0])

        # Ensure the expansion point and domain remain unchanged
        assert np.array_equal(negated_te.expansion_point, self.te.expansion_point)
        assert np.array_equal(negated_te.domain[0], self.te.domain[0])
        assert np.array_equal(negated_te.domain[1], self.te.domain[1])

        negated_te = self.te/(-1)

        # Check that the linear approximation is negated
        assert np.allclose(negated_te.linear_approximation[0], -self.te.linear_approximation[0])
        assert np.allclose(negated_te.linear_approximation[1], -self.te.linear_approximation[1])

        # Check that the remainder bounds are swapped and negated
        assert np.allclose(negated_te.remainder[0], -self.te.remainder[1])
        assert np.allclose(negated_te.remainder[1], -self.te.remainder[0])

        # Ensure the expansion point and domain remain unchanged
        assert np.array_equal(negated_te.expansion_point, self.te.expansion_point)
        assert np.array_equal(negated_te.domain[0], self.te.domain[0])
        assert np.array_equal(negated_te.domain[1], self.te.domain[1])

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
        
        intervals_list = [
            (np.array([-1]), np.array([1])),
            (np.array([0.0]), np.array([1])), 
            (np.array([np.pi / 4]), np.array([4*np.pi / 5])),
            (np.array([-np.pi]), np.array([0.0])),
            (np.array([-10]), np.array([20])),
            (np.array([-100]), np.array([100]))
        ]

        for i, interval in enumerate(intervals_list):
            # Define the expansion point as the midpoint of the interval
            expansion_point = (interval[0] + interval[1]) / 2

            # Create a Taylor expansion for sin(x) around the expansion_point
            te = CertifiedFirstOrderTaylorExpansion(
                expansion_point=expansion_point,
                domain=interval
            )

            # Compute the sine function using the translator
            translator = TaylorTranslator()
            result = translator.sin(te)
            
            expected_constant = np.sin(expansion_point)
            expected_jacobian = np.cos(expansion_point).reshape(-1, 1) * te.linear_approximation[0]
            
            assert np.allclose(result.linear_approximation[1], expected_constant)
            assert np.allclose(result.linear_approximation[0], expected_jacobian)

            # Verify that the first-order approximation with the remainder term contains sin(x) on the interval
            x_test = np.linspace(interval[0], interval[1], 1000)  # Ensure x_test is a column vector
            sin_x = np.sin(x_test) # True sine values, flattened for plotting
            approx_with_remainder_lower, approx_with_remainder_upper = self.compute_approximation_bounds(
                result, x_test, expansion_point
            )

            assert np.all(sin_x >= approx_with_remainder_lower)
            assert np.all(sin_x <= approx_with_remainder_upper)

            # Optional plotting for visualization
            PLOT_TESTS = False
            if PLOT_TESTS:
                approx_function = (
                    result.linear_approximation[1] +
                    result.linear_approximation[0].dot((x_test - expansion_point).T).T
                )
                self.plot_taylor_approximation(
                    x_test=x_test,
                    true_values=sin_x,
                    approx_function=approx_function,
                    approx_with_remainder_lower=approx_with_remainder_lower,
                    approx_with_remainder_upper=approx_with_remainder_upper,
                    expansion_point=expansion_point,
                    title=f"Sine Function and First-Order Approximation (Interval {interval})",
                    ylabel="sin(x)"
                )

        # Multidimensional test
        domain = (np.array([-np.pi, 0.0]), np.array([np.pi, np.pi / 2]))
        expansion_point = (domain[1] + domain[0]) / 2
        te_multi = CertifiedFirstOrderTaylorExpansion(
            expansion_point=expansion_point,
            domain=domain
        )

        result_multi = self.translator.sin(te_multi)

        expected_constant_multi = np.sin(expansion_point)
        expected_jacobian_multi = np.cos(expansion_point).reshape(-1, 1) * te_multi.linear_approximation[0]

        assert np.allclose(result_multi.linear_approximation[1], expected_constant_multi)
        assert np.allclose(result_multi.linear_approximation[0], expected_jacobian_multi)

        # Verify that the first-order approximation with the remainder term contains sin(x) on the interval
        n_points = 64  # Use fewer points for efficiency
        x1_range = np.linspace(result_multi.domain[0][0], result_multi.domain[1][0], n_points)
        x2_range = np.linspace(result_multi.domain[0][1], result_multi.domain[1][1], n_points)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        x_test = np.column_stack([X1.ravel(), X2.ravel()])
        sin_x = np.sin(x_test)  # True sine values
        approx_with_remainder_lower, approx_with_remainder_upper = self.compute_approximation_bounds(
            result_multi, x_test, expansion_point
        )

        assert np.all(sin_x >= approx_with_remainder_lower)
        assert np.all(sin_x <= approx_with_remainder_upper)

    def test_cos(self):
        """Test cosine function."""
        intervals_list = [
            (np.array([-1]), np.array([1])),
            (np.array([0.0]), np.array([1])),
            (np.array([np.pi / 4]), np.array([4 * np.pi / 5])),
            (np.array([-np.pi]), np.array([0.0])),
            (np.array([-10]), np.array([20])),
            (np.array([-100]), np.array([100]))
        ]

        for i, interval in enumerate(intervals_list):
            # Define the expansion point as the midpoint of the interval
            expansion_point = (interval[0] + interval[1]) / 2

            # Create a Taylor expansion for cos(x) around the expansion_point
            te = CertifiedFirstOrderTaylorExpansion(
                expansion_point=expansion_point,
                domain=interval
            )

            # Compute the cosine function using the translator
            translator = TaylorTranslator()
            result = translator.cos(te)

            expected_constant = np.cos(expansion_point)
            expected_jacobian = -np.sin(expansion_point).reshape(-1, 1) * te.linear_approximation[0]

            assert np.allclose(result.linear_approximation[1], expected_constant)
            assert np.allclose(result.linear_approximation[0], expected_jacobian)

            # Verify that the first-order approximation with the remainder term contains cos(x) on the interval
            x_test = np.linspace(interval[0], interval[1], 1000)  # Ensure x_test is a column vector
            cos_x = np.cos(x_test)  # True cosine values
            approx_with_remainder_lower, approx_with_remainder_upper = self.compute_approximation_bounds(
                result, x_test, expansion_point
            )

            assert np.all(cos_x >= approx_with_remainder_lower)
            assert np.all(cos_x <= approx_with_remainder_upper)

            # Optional plotting for visualization
            PLOT_TESTS = False
            if PLOT_TESTS:
                approx_function = (
                    result.linear_approximation[1] +
                    result.linear_approximation[0].dot((x_test - expansion_point).T).T
                )
                self.plot_taylor_approximation(
                    x_test=x_test,
                    true_values=cos_x,
                    approx_function=approx_function,
                    approx_with_remainder_lower=approx_with_remainder_lower,
                    approx_with_remainder_upper=approx_with_remainder_upper,
                    expansion_point=expansion_point,
                    title=f"Cosine Function and First-Order Approximation (Interval {interval})",
                    ylabel="cos(x)"
                )

        # Multidimensional test
        domain = (np.array([-np.pi, 0.0]), np.array([np.pi, np.pi / 2]))
        expansion_point = (domain[1] + domain[0]) / 2
        te_multi = CertifiedFirstOrderTaylorExpansion(
            expansion_point=expansion_point,
            domain=domain
        )

        result_multi = self.translator.cos(te_multi)

        expected_constant_multi = np.cos(expansion_point)
        expected_jacobian_multi = -np.sin(expansion_point).reshape(-1, 1) * te_multi.linear_approximation[0]

        assert np.allclose(result_multi.linear_approximation[1], expected_constant_multi)
        assert np.allclose(result_multi.linear_approximation[0], expected_jacobian_multi)

        # Verify that the first-order approximation with the remainder term contains cos(x) on the interval
        n_points = 64  # Use fewer points for efficiency
        x1_range = np.linspace(result_multi.domain[0][0], result_multi.domain[1][0], n_points)
        x2_range = np.linspace(result_multi.domain[0][1], result_multi.domain[1][1], n_points)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        x_test = np.column_stack([X1.ravel(), X2.ravel()])
        cos_x = np.cos(x_test)  # True cosine values
        approx_with_remainder_lower, approx_with_remainder_upper = self.compute_approximation_bounds(
            result_multi, x_test, expansion_point
        )

        assert np.all(cos_x >= approx_with_remainder_lower)
        assert np.all(cos_x <= approx_with_remainder_upper)

    def test_exp(self):
        """Test exponential function."""
        domain = (np.array([-1.0]), np.array([3.1]))
        expansion_point = (domain[1] + domain[0]) / 2
        te = CertifiedFirstOrderTaylorExpansion(
            expansion_point, domain
        )
        result = self.translator.exp(te)
        
        expected_constant = np.exp(expansion_point)
        expected_jacobian = np.exp(expansion_point).reshape(-1, 1) * te.linear_approximation[0]
        
        assert np.allclose(result.linear_approximation[1], expected_constant)
        assert np.allclose(result.linear_approximation[0], expected_jacobian)

        # Verify that the first-order approximation with the remainder term contains exp(x) on the interval
        x_test = np.linspace(domain[0], domain[1], 1000)  # Ensure x_test is a column vector
        exp_x = np.exp(x_test)  # True exponential values, flattened for comparison
        approx_with_remainder_lower, approx_with_remainder_upper = self.compute_approximation_bounds(
            result, x_test, expansion_point
        )

        assert np.all(exp_x >= approx_with_remainder_lower)
        assert np.all(exp_x <= approx_with_remainder_upper)

        # Multidimensional test
        domain = (np.array([-10, 0.0]), np.array([np.pi, 0.1]))
        expansion_point = (domain[1] + domain[0]) / 2
        te_multi = CertifiedFirstOrderTaylorExpansion(
            expansion_point=expansion_point,
            domain=domain
        )

        result_multi = self.translator.exp(te_multi)

        expected_constant_multi = np.exp(expansion_point)
        expected_jacobian_multi = np.exp(expansion_point).reshape(-1, 1) * te_multi.linear_approximation[0]

        assert np.allclose(result_multi.linear_approximation[1], expected_constant_multi)
        assert np.allclose(result_multi.linear_approximation[0], expected_jacobian_multi)

        # Verify that the first-order approximation with the remainder term contains exp(x) on the interval
        n_points = 64  # Use fewer points for efficiency
        x1_range = np.linspace(result_multi.domain[0][0], result_multi.domain[1][0], n_points)
        x2_range = np.linspace(result_multi.domain[0][1], result_multi.domain[1][1], n_points)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        x_test = np.column_stack([X1.ravel(), X2.ravel()])
        exp_x = np.exp(x_test)  # True exponential values, flattened for comparison
        approx_with_remainder_lower, approx_with_remainder_upper = self.compute_approximation_bounds(
            result_multi, x_test, expansion_point
        )

        assert np.all(exp_x >= approx_with_remainder_lower)
        assert np.all(exp_x <= approx_with_remainder_upper)

    def test_log(self):
        """Test logarithm function."""
        # Use positive values to avoid domain issues
        domain = (np.array([1.0]), np.array([9.0]))
        expansion_point = (domain[1] + domain[0]) / 2
        te_pos = CertifiedFirstOrderTaylorExpansion(
            expansion_point, domain
        )
        
        result = self.translator.log(te_pos)
        
        expected_constant = np.log(expansion_point)
        expected_jacobian = (1.0 / expansion_point).reshape(-1, 1) * te_pos.linear_approximation[0]
        
        assert np.allclose(result.linear_approximation[1], expected_constant)
        assert np.allclose(result.linear_approximation[0], expected_jacobian)

        # Verify that the first-order approximation with the remainder term contains log(x) on the interval
        x_test = np.linspace(te_pos.domain[0], te_pos.domain[1], 1000)  # Ensure x_test is a column vector
        log_x = np.log(x_test)  # True logarithm values, flattened for comparison
        approx_with_remainder_lower, approx_with_remainder_upper = self.compute_approximation_bounds(
            result, x_test, te_pos.expansion_point
        )

        assert np.all(log_x >= approx_with_remainder_lower)
        assert np.all(log_x <= approx_with_remainder_upper)

        # Optional plotting for visualization
        PLOT_TESTS = False
        if PLOT_TESTS:
            approx_function = (
                result.linear_approximation[1] +
                result.linear_approximation[0].dot((x_test - te_pos.expansion_point).T).T
            )
            self.plot_taylor_approximation(
                x_test=x_test,
                true_values=log_x,
                approx_function=approx_function,
                approx_with_remainder_lower=approx_with_remainder_lower,
                approx_with_remainder_upper=approx_with_remainder_upper,
                expansion_point=te_pos.expansion_point,
                title="Logarithm Function and First-Order Approximation",
                ylabel="log(x)"
            )

        # Multidimensional test
        domain = (np.array([1.0, 2.0]), np.array([4.0, 5.0]))
        expansion_point = (domain[1] + domain[0]) / 2
        te_multi = CertifiedFirstOrderTaylorExpansion(
            expansion_point=expansion_point,
            domain=domain
        )

        result_multi = self.translator.log(te_multi)

        expected_constant_multi = np.log(expansion_point)
        expected_jacobian_multi = (1.0 / expansion_point).reshape(-1, 1) * te_multi.linear_approximation[0]

        assert np.allclose(result_multi.linear_approximation[1], expected_constant_multi)
        assert np.allclose(result_multi.linear_approximation[0], expected_jacobian_multi)

        # Verify that the first-order approximation with the remainder term contains log(x) on the interval
        n_points = 64  # Use fewer points for efficiency
        x1_range = np.linspace(result_multi.domain[0][0], result_multi.domain[1][0], n_points)
        x2_range = np.linspace(result_multi.domain[0][1], result_multi.domain[1][1], n_points)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        x_test = np.column_stack([X1.ravel(), X2.ravel()])
        log_x = np.log(x_test)  # True logarithm values
        approx_with_remainder_lower, approx_with_remainder_upper = self.compute_approximation_bounds(
            result_multi, x_test, expansion_point
        )

        assert np.all(log_x >= approx_with_remainder_lower)
        assert np.all(log_x <= approx_with_remainder_upper)

    def test_sqrt(self):
        """Test square root function."""
        domain = (np.array([1.0]), np.array([9.0]))
        expansion_point = (domain[1] + domain[0]) / 2
        te_pos = CertifiedFirstOrderTaylorExpansion(
            expansion_point, domain
        )
        
        result = self.translator.sqrt(te_pos)
        
        expected_constant = np.sqrt(expansion_point)
        expected_jacobian = (0.5 / np.sqrt(expansion_point)).reshape(-1, 1) * te_pos.linear_approximation[0]
        
        assert np.allclose(result.linear_approximation[1], expected_constant)
        assert np.allclose(result.linear_approximation[0], expected_jacobian)

        # Verify that the first-order approximation with the remainder term contains sqrt(x) on the interval
        x_test = np.linspace(te_pos.domain[0], te_pos.domain[1], 1000)  # Multidimensional test points
        sqrt_x = np.sqrt(x_test)  # True square root values
        approx_with_remainder_lower, approx_with_remainder_upper = self.compute_approximation_bounds(
            result, x_test, te_pos.expansion_point
        )

        assert np.all(sqrt_x >= approx_with_remainder_lower)
        assert np.all(sqrt_x <= approx_with_remainder_upper)

        # Multidimensional test
        domain = (np.array([1.0, 4.0]), np.array([16.0, 25.0]))
        expansion_point = (domain[1] + domain[0]) / 2
        te_multi = CertifiedFirstOrderTaylorExpansion(
            expansion_point=expansion_point,
            domain=domain
        )

        result_multi = self.translator.sqrt(te_multi)

        expected_constant_multi = np.sqrt(expansion_point)
        expected_jacobian_multi = (0.5 / np.sqrt(expansion_point)).reshape(-1, 1) * te_multi.linear_approximation[0]

        assert np.allclose(result_multi.linear_approximation[1], expected_constant_multi)
        assert np.allclose(result_multi.linear_approximation[0], expected_jacobian_multi)

        # Verify that the first-order approximation with the remainder term contains sqrt(x) on the interval
        n_points = 64  # Use fewer points for efficiency
        x1_range = np.linspace(result_multi.domain[0][0], result_multi.domain[1][0], n_points)
        x2_range = np.linspace(result_multi.domain[0][1], result_multi.domain[1][1], n_points)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        x_test = np.column_stack([X1.ravel(), X2.ravel()])
        sqrt_x = np.sqrt(x_test)  # True square root values
        approx_with_remainder_lower, approx_with_remainder_upper = self.compute_approximation_bounds(
            result_multi, x_test, expansion_point
        )

        assert np.all(sqrt_x >= approx_with_remainder_lower)
        assert np.all(sqrt_x <= approx_with_remainder_upper)

    def test_pow(self):
        """Test power function."""
        exponent = 3
        domain = (np.array([-1.0]), np.array([6.0]))
        expansion_point = (domain[1] + domain[0]) / 2
        te = CertifiedFirstOrderTaylorExpansion(
            expansion_point, domain
        )
        result = self.translator.pow(te, exponent)
        
        expected_constant = np.pow(expansion_point, exponent)
        expected_jacobian = (exponent * np.pow(expansion_point, exponent - 1)).reshape(-1, 1) * te.linear_approximation[0]
        
        assert np.allclose(result.linear_approximation[1], expected_constant)
        assert np.allclose(result.linear_approximation[0], expected_jacobian)

        # Verify that the first-order approximation with the remainder term contains x^3 on the interval
        x_test = np.linspace(domain[0], domain[1], 1000)  # Ensure x_test is a column vector
        pow_x = np.power(x_test, exponent)  # True power values, flattened for comparison
        approx_with_remainder_lower, approx_with_remainder_upper = self.compute_approximation_bounds(
            result, x_test, expansion_point
        )

        assert np.all(pow_x >= approx_with_remainder_lower)
        assert np.all(pow_x <= approx_with_remainder_upper)

        # Optional plotting for visualization
        PLOT_TESTS = False
        if PLOT_TESTS:
            approx_function = (
                result.linear_approximation[1] +
                result.linear_approximation[0].dot((x_test - te.expansion_point).T).T
            )
            self.plot_taylor_approximation(
                x_test=x_test,
                true_values=pow_x,
                approx_function=approx_function,
                approx_with_remainder_lower=approx_with_remainder_lower,
                approx_with_remainder_upper=approx_with_remainder_upper,
                expansion_point=te.expansion_point,
                title="Logarithm Function and First-Order Approximation",
                ylabel="pow(x)"
            )

        # Multidimensional test
        exponent = 3
        domain = (np.array([1.0, 2.0]), np.array([4.0, 5.0]))
        expansion_point = (domain[1] + domain[0]) / 2
        te_multi = CertifiedFirstOrderTaylorExpansion(
            expansion_point=expansion_point,
            domain=domain
        )

        result_multi = self.translator.pow(te_multi, exponent)

        expected_constant_multi = np.power(expansion_point, exponent)
        expected_jacobian_multi = (exponent * np.power(expansion_point, exponent - 1)).reshape(-1, 1) * te_multi.linear_approximation[0]

        assert np.allclose(result_multi.linear_approximation[1], expected_constant_multi)
        assert np.allclose(result_multi.linear_approximation[0], expected_jacobian_multi)

        # Verify that the first-order approximation with the remainder term contains x^3 on the interval
        n_points = 64  # Use fewer points for efficiency
        x1_range = np.linspace(result_multi.domain[0][0], result_multi.domain[1][0], n_points)
        x2_range = np.linspace(result_multi.domain[0][1], result_multi.domain[1][1], n_points)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        x_test = np.column_stack([X1.ravel(), X2.ravel()])
        pow_x = np.power(x_test, exponent)  # True power values
        approx_with_remainder_lower, approx_with_remainder_upper = self.compute_approximation_bounds(
            result_multi, x_test, expansion_point
        )

        PLOT_TESTS = False
        if PLOT_TESTS:
            approx_function = (
                result_multi.linear_approximation[1] +
                result_multi.linear_approximation[0].dot((x_test - te_multi.expansion_point).T).T
            )
            self.plot_taylor_approximation(
                x_test=x_test,
                true_values=pow_x,
                approx_function=approx_function,
                approx_with_remainder_lower=approx_with_remainder_lower,
                approx_with_remainder_upper=approx_with_remainder_upper,
                expansion_point=te_multi.expansion_point,
                title="Logarithm Function and First-Order Approximation",
                ylabel="pow(x)"
            )

        assert np.all(pow_x >= approx_with_remainder_lower)
        assert np.all(pow_x <= approx_with_remainder_upper)

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

    def test_composite_function_exp_sin(self):
        """Test the composite function 0.2 * exp(sin(x^3) + 0.1 * x) on the domain [-0.5, 0.7] with expansion point 0.1."""
        # Define the input Taylor expansion for x
        te_x = CertifiedFirstOrderTaylorExpansion(
            expansion_point=np.array([0.1]),
            domain=(np.array([-0.5]), np.array([0.7]))
        )

        # Compute x^3
        te_x_cubed = self.translator.pow(te_x, 3)

        # Compute sin(x^3)
        te_sin_x_cubed = self.translator.sin(te_x_cubed)

        # Compute 0.1 * x
        te_0_1_x = 0.1 * te_x

        # Compute sin(x^3) + 0.1 * x
        te_inner = te_sin_x_cubed + te_0_1_x

        # Compute exp(sin(x^3) + 0.1 * x)
        te_exp_inner = self.translator.exp(te_inner)

        # Compute 0.2 * exp(sin(x^3) + 0.1 * x)
        te_final = 0.2 * te_exp_inner

        # Verify the structure of the result
        assert np.array_equal(te_final.expansion_point, te_x.expansion_point)
        assert np.array_equal(te_final.domain[0], te_x.domain[0])
        assert np.array_equal(te_final.domain[1], te_x.domain[1])

        # Verify that the linear approximation and remainder are computed
        assert te_final.linear_approximation[0] is not None
        assert te_final.linear_approximation[1] is not None
        assert te_final.remainder[0] is not None
        assert te_final.remainder[1] is not None

        # Verify that the first-order approximation with the remainder term contains the composite function on the interval
        x_test = np.linspace(te_x.domain[0], te_x.domain[1], 1000)  # Ensure x_test is a column vector
        composite_x = 0.2 * np.exp(np.sin(np.power(x_test, 3)) + 0.1 * x_test)  # True composite function values
        approx_with_remainder_lower, approx_with_remainder_upper = self.compute_approximation_bounds(
            te_final, x_test, te_x.expansion_point
        )

        assert np.all(composite_x >= approx_with_remainder_lower)
        assert np.all(composite_x <= approx_with_remainder_upper)

        # Manual verification of the composite function
        x_val = te_x.expansion_point
        expected_constant_manual = 0.2 * np.exp(np.sin(x_val**3) + 0.1 * x_val)
        expected_jacobian_manual = (
            0.2 * np.exp(np.sin(x_val**3) + 0.1 * x_val)
            * (np.cos(x_val**3) * 3 * x_val**2 + 0.1)
        )

        assert np.allclose(te_final.linear_approximation[1], expected_constant_manual)
        assert np.allclose(te_final.linear_approximation[0], expected_jacobian_manual)
           
    def test_compute_dynamics(self):
        """Test the compute_dynamics method."""
        def compute_dynamics(x, translator):
            # ẋ = -y - 1.5x² - 0.5x³ - 0.1
            # ẏ = 3x - y
            dx = -x[1] - 1.5 * translator.pow(x[0], 2) - 0.5 * translator.pow(x[0], 3) - 0.1
            dy = 3 * x[0] - x[1]
            return translator.stack([dx, dy])

        # Define the input Taylor expansion for x
        domain = (np.array([-0.5, -0.5]), np.array([0.5, 0.5]))
        expansion_point = (domain[1] + domain[0]) / 2
        x = CertifiedFirstOrderTaylorExpansion(
            expansion_point=expansion_point,
            domain=domain
        )

        # Compute dynamics
        result = compute_dynamics(x, self.translator)

        # Verify the structure of the result
        assert np.array_equal(result.expansion_point, expansion_point)
        assert np.array_equal(result.domain[0], domain[0])
        assert np.array_equal(result.domain[1], domain[1])

        # Verify the linear approximation and remainder
        assert result.linear_approximation[0] is not None
        assert result.linear_approximation[1] is not None
        assert result.remainder[0] is not None
        assert result.remainder[1] is not None

        # Manual verification of the dynamics
        x_val = expansion_point
        dx_expected = -x_val[1] - 1.5 * x_val[0]**2 - 0.5 * x_val[0]**3 - 0.1
        dy_expected = 3 * x_val[0] - x_val[1]
        expected_constant = np.array([dx_expected, dy_expected])

        assert np.allclose(result.linear_approximation[1], expected_constant)

        # Test function approximation over domain using meshgrid for proper 2D sampling
        n_points = 64  # Use fewer points for efficiency
        x1_range = np.linspace(result.domain[0][0], result.domain[1][0], n_points)
        x2_range = np.linspace(result.domain[0][1], result.domain[1][1], n_points)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        x_test = np.column_stack([X1.ravel(), X2.ravel()])
        
        f_x = compute_dynamics(x_test.T, NumpyTranslator()).T
        approx_with_remainder_lower, approx_with_remainder_upper = self.compute_approximation_bounds(
            result, x_test, result.expansion_point
        )

        PLOT_TESTS = False
        if PLOT_TESTS:
            approx_function = (
                result.linear_approximation[1] +
                result.linear_approximation[0].dot((x_test - result.expansion_point).T).T
            )
            self.plot_taylor_approximation(
                x_test=x_test,
                true_values=f_x,
                approx_function=approx_function,
                approx_with_remainder_lower=approx_with_remainder_lower,
                approx_with_remainder_upper=approx_with_remainder_upper,
                expansion_point=expansion_point,
                title=f"Dynamics Function and First-Order Approximation",
                ylabel="f(x)"
            )

        assert np.all(f_x >= approx_with_remainder_lower)
        assert np.all(f_x <= approx_with_remainder_upper)
            
    def plot_taylor_approximation(self, x_test, true_values, approx_function, approx_with_remainder_lower, approx_with_remainder_upper, expansion_point, title, ylabel):
        """
        Helper function to plot Taylor approximation and bounds for 1D and 2D cases.

        Args:
            x_test (np.ndarray): Test points for x.
            true_values (np.ndarray): True function values.
            approx_function (np.ndarray): Approximation function values.
            approx_with_remainder_lower (np.ndarray): Lower bound of approximation with remainder.
            approx_with_remainder_upper (np.ndarray): Upper bound of approximation with remainder.
            expansion_point (np.ndarray): Expansion point for the Taylor approximation.
            title (str): Title of the plot.
            ylabel (str): Label for the y-axis.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        if x_test.shape[1] == 1 and true_values.shape[1]==1:  # 1D case
            plt.figure(figsize=(8, 6))
            plt.plot(x_test, true_values, label="True Function", color="blue")
            plt.plot(x_test, approx_with_remainder_lower, label="Lower Bound", linestyle="--", color="green")
            plt.plot(x_test, approx_with_remainder_upper, label="Upper Bound", linestyle="--", color="red")
            plt.plot(x_test, approx_function, label="Approximation", linestyle=":", color="orange")
            plt.axvline(expansion_point.item(), color="black", linestyle=":", label="Expansion Point")
            plt.title(title)
            plt.xlabel("x")
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid(True)
            plt.show()

        elif x_test.shape[1] == 2 and true_values.shape[1]==2:  # 2D case - create 3D scatter plots
            fig = plt.figure(figsize=(15, 6))
            
            for i in range(2):  # For each output dimension
                ax = fig.add_subplot(1, 2, i+1, projection='3d')
                
                # Use scatter plots for irregular grids
                ax.scatter(x_test[:, 0], x_test[:, 1], true_values[:, i], alpha=0.7, color='blue', label='True Function', s=1)
                ax.scatter(x_test[:, 0], x_test[:, 1], approx_function[:, i], alpha=0.5, color='orange', label='Approximation', s=1)
                ax.scatter(x_test[:, 0], x_test[:, 1], approx_with_remainder_lower[:, i], alpha=0.3, color='green', label='Lower Bound', s=1)
                ax.scatter(x_test[:, 0], x_test[:, 1], approx_with_remainder_upper[:, i], alpha=0.3, color='red', label='Upper Bound', s=1)
                
                # Mark expansion point
                ax.scatter(expansion_point[0], expansion_point[1], 
                          true_values[np.argmin(np.sum((x_test - expansion_point)**2, axis=1)), i],
                          color='black', s=100, label='Expansion Point')
                
                ax.set_title(f"{title} - Output {i+1}")
                ax.set_xlabel("x₁")
                ax.set_ylabel("x₂") 
                ax.set_zlabel(f"{ylabel}[{i}]")
                ax.legend()
                
            plt.tight_layout()
            plt.show()

        else:
            print("Skipping unsupported dimensionality for plotting.")

    def compute_approximation_bounds(self, result, x_test, expansion_point):
        """
        Compute the lower and upper bounds of the first-order approximation with remainder.

        Args:
            result: Taylor expansion result containing linear approximation and remainder.
            x_test: Test points for x.
            expansion_point: Expansion point for the Taylor approximation.

        Returns:
            Tuple of (approx_with_remainder_lower, approx_with_remainder_upper).
        """
        approx_with_remainder_lower = (
            result.linear_approximation[1] +
            result.linear_approximation[0].dot((x_test - expansion_point).T).T +
            result.remainder[0]
        )
        approx_with_remainder_upper = (
            result.linear_approximation[1] +
            result.linear_approximation[0].dot((x_test - expansion_point).T).T +
            result.remainder[1]
        )
        return approx_with_remainder_lower, approx_with_remainder_upper

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
