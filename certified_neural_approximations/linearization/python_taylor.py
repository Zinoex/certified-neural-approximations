import numpy as np
from ..certification_results import AugmentedSample
from ..translators.taylor_translator import TaylorTranslator, CertifiedFirstOrderTaylorExpansion


def first_order_certified_taylor_expansion_python(dynamics, expansion_point, delta):
    """
    A 1st-order Taylor expansion including residual (certified) of a function around a point.
    
    This is computed using the Python TaylorTranslator implementation.
    
    :param dynamics: An object representing the dynamics to be expanded.
    :param expansion_point: The point around which to expand the dynamics.
    :param delta: The (hyperrectangular) radius of the expansion.
    :return: CertifiedFirstOrderTaylorExpansion object representing f(x) = f(c) + ∇f(c)(x-c) + R(x)
    """
    assert isinstance(expansion_point, np.ndarray), "Expansion point must be a numpy array"
    assert isinstance(delta, np.ndarray), "Delta must be a numpy array"
    
    translator = TaylorTranslator()
    
    # Define the domain bounds
    lower_bounds = expansion_point - delta
    upper_bounds = expansion_point + delta
    
    # Create the initial Taylor expansion for the identity function f(x) = x
    x = translator.to_format(expansion_point, lower_bounds, upper_bounds)
    
    # Compute the dynamics using the Taylor translator
    y = dynamics.compute_dynamics(x, translator)
    
    return y


class PythonTaylorLinearization:
    """
    Taylor linearization using the Python taylor translator implementation.
    """
    
    def __init__(self, dynamics):
        self.dynamics = dynamics
    
    def linearize(self, samples):
        """
        Linearizes a batch of samples using Taylor expansion.
        
        :param samples: List of samples to linearize
        :return: List of AugmentedSample objects with linearization information
        """
        return [self.linearize_sample(sample) for sample in samples]
    
    def linearize_sample(self, sample):
        """
        Linearize a single sample using certified Taylor expansion.
        
        :param sample: Sample object with center, radius, and output_dim
        :return: AugmentedSample with linearization bounds
        """
        # Compute the Taylor expansion
        taylor_expansion = first_order_certified_taylor_expansion_python(
            self.dynamics, sample.center, sample.radius
        )
        
        # Extract components for the specific output dimension
        output_idx = sample.output_dim
        
        # Get the Jacobian (gradient) and function value at center
        jacobian, f_c = taylor_expansion.linear_approximation
        remainder_lower, remainder_upper = taylor_expansion.remainder
        
        # Extract values for the specific output dimension
        if jacobian.ndim > 1:
            # Multi-dimensional output
            df_c = jacobian[output_idx]  # Gradient for this output dimension
            f_c_val = f_c[output_idx]    # Function value for this output dimension
            r_lower = remainder_lower[output_idx]  # Lower remainder bound
            r_upper = remainder_upper[output_idx]  # Upper remainder bound
        else:
            # Single-dimensional output
            df_c = jacobian.flatten()
            f_c_val = f_c.item() if hasattr(f_c, 'item') else f_c
            r_lower = remainder_lower.item() if hasattr(remainder_lower, 'item') else remainder_lower
            r_upper = remainder_upper.item() if hasattr(remainder_upper, 'item') else remainder_upper
        
        # Construct affine bounds: f(x) ≈ f(c) + ∇f(c)·(x - c) + R
        # In affine form: A·x + b where A = ∇f(c), b = f(c) - ∇f(c)·c + R
        
        # Upper bound: A_upper·x + b_upper
        A_upper = df_c
        b_upper = f_c_val - np.dot(df_c, sample.center) + r_upper
        
        # Lower bound: A_lower·x + b_lower  
        A_lower = df_c
        b_lower = f_c_val - np.dot(df_c, sample.center) + r_lower
        
        # Maximum gap between upper and lower bounds
        max_gap = r_upper - r_lower
        
        return AugmentedSample.from_certification_region(
            sample,
            ((A_lower, b_lower), (A_upper, b_upper), max_gap)
        )
    
    def get_taylor_expansion(self, expansion_point, delta):
        """
        Get the full Taylor expansion object for analysis.
        
        :param expansion_point: Point around which to expand
        :param delta: Radius of expansion domain
        :return: CertifiedFirstOrderTaylorExpansion object
        """
        return first_order_certified_taylor_expansion_python(
            self.dynamics, expansion_point, delta
        )
    
    def evaluate_at_point(self, point):
        """
        Evaluate the dynamics at a specific point.
        
        :param point: Point at which to evaluate
        :return: Function value at the point
        """
        # Create a trivial expansion at the point
        delta = np.zeros_like(point)
        expansion = first_order_certified_taylor_expansion_python(
            self.dynamics, point, delta
        )
        
        # Return the function value (constant term)
        return expansion.linear_approximation[1]
    
    def get_jacobian_at_point(self, point):
        """
        Get the Jacobian matrix at a specific point.
        
        :param point: Point at which to compute Jacobian
        :return: Jacobian matrix
        """
        # Create a small expansion around the point
        delta = np.full_like(point, 1e-8)
        expansion = first_order_certified_taylor_expansion_python(
            self.dynamics, point, delta
        )
        
        # Return the Jacobian (linear term)
        return expansion.linear_approximation[0]
