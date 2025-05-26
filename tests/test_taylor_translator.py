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

    def test_division_by_scalar(self):
        """Test division by scalar."""
        scalar = 2.0
        result = self.te / scalar
        
        assert np.allclose(result.linear_approximation[0], self.linear_approx[0] / scalar)
        assert np.allclose(result.linear_approximation[1], self.linear_approx[1] / scalar)
        assert np.allclose(result.remainder[0], self.remainder[0] / scalar)
        assert np.allclose(result.remainder[1], self.remainder[1] / scalar)

    def test_division_by_zero(self):
        """Test division by zero raises error."""
        with pytest.raises(ZeroDivisionError):
            self.te / 0

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
        assert np.array_equal(result.linear_approximation[0], np.ones(point.shape[0]))
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
