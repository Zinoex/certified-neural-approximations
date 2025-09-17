import numpy as np
from ..certification_results import AugmentedSample
from .taylor import TaylorLinearization, first_order_certified_taylor_expansion
from .python_taylor import PythonTaylorLinearization, first_order_certified_taylor_expansion_python


class ComparisonTaylorLinearization:
    """
    Comparison class that runs both Julia and Python Taylor linearization
    implementations and compares their results for debugging purposes.
    """
    
    def __init__(self, dynamics, tolerance=1e-6):
        self.dynamics = dynamics
        self.julia_linearizer = TaylorLinearization(dynamics)
        self.python_linearizer = PythonTaylorLinearization(dynamics)
        self.tolerance = tolerance
        self.comparison_results = []
    
    def linearize(self, samples):
        """
        Linearize samples using both implementations and compare results.
        
        :param samples: List of samples to linearize
        :return: List of AugmentedSample objects from Python implementation
        """
        results = []
        
        for i, sample in enumerate(samples):
            try:
                # Get results from both implementations
                julia_result = self.julia_linearizer.linearize_sample(sample)
                python_result = self.python_linearizer.linearize_sample(sample)
                
                # Compare the results
                comparison = self._compare_results(julia_result, python_result, sample, i)
                self.comparison_results.append(comparison)
                
                # Return Python result (could be either, but Python is more debuggable)
                results.append(python_result)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                # Fallback to Python implementation
                results.append(self.python_linearizer.linearize_sample(sample))
                self.comparison_results.append({
                    'sample_index': i,
                    'error': str(e),
                    'status': 'error'
                })
        
        return results
    
    def linearize_sample(self, sample):
        """
        Linearize a single sample and compare implementations.
        
        :param sample: Sample object to linearize
        :return: AugmentedSample with comparison metadata
        """
        try:
            julia_result = self.julia_linearizer.linearize_sample(sample)
            python_result = self.python_linearizer.linearize_sample(sample)
            
            comparison = self._compare_results(julia_result, python_result, sample, 0)
            if not comparison["within_tolerance"]:
                if not comparison["julia_in_py"]:
                    print(f"Python tighter than Julia: {comparison}")
            # Add comparison metadata to the result
            python_result.comparison_metadata = comparison
            
            return python_result
            
        except Exception as e:
            print(f"Error in comparison: {e}")
            result = self.python_linearizer.linearize_sample(sample)
            result.comparison_metadata = {'error': str(e), 'status': 'error'}
            return result
    
    def _compare_results(self, julia_result, python_result, sample, sample_index):
        """
        Compare results from Julia and Python implementations.
        
        :param julia_result: AugmentedSample from Julia implementation
        :param python_result: AugmentedSample from Python implementation
        :param sample: Original sample
        :param sample_index: Index of sample for tracking
        :return: Dictionary with comparison results
        """
        comparison = {
            'sample_index': sample_index,
            'center': sample.center.tolist(),
            'radius': sample.radius.tolist(),
            'output_dim': sample.output_dim,
            'status': 'compared'
        }
        
        # Extract linearization data from first_order_model attribute
        julia_bounds = julia_result.first_order_model
        python_bounds = python_result.first_order_model
        
        # Compare lower bounds
        julia_A_lower, julia_b_lower = julia_bounds[0]
        python_A_lower, python_b_lower = python_bounds[0]
        
        # Compare upper bounds  
        julia_A_upper, julia_b_upper = julia_bounds[1]
        python_A_upper, python_b_upper = python_bounds[1]
        
        # Compare gaps
        julia_gap = julia_bounds[2]
        python_gap = python_bounds[2]
        
        # Compute differences
        A_lower_diff = np.abs(julia_A_lower - python_A_lower)
        b_lower_diff = np.abs(julia_b_lower - python_b_lower)
        A_upper_diff = np.abs(julia_A_upper - python_A_upper)
        b_upper_diff = np.abs(julia_b_upper - python_b_upper)
        gap_diff = np.abs(julia_gap - python_gap)
        
        comparison.update({
            'A_lower_max_diff': np.max(A_lower_diff),
            'b_lower_diff': b_lower_diff,
            'A_upper_max_diff': np.max(A_upper_diff),
            'b_upper_diff': b_upper_diff,
            'gap_diff': gap_diff,
            'within_tolerance': (
                np.max(A_lower_diff) < self.tolerance and
                np.max(b_lower_diff) < self.tolerance and
                np.max(A_upper_diff) < self.tolerance and
                np.max(b_upper_diff) < self.tolerance and
                np.max(gap_diff) < self.tolerance
            ),
            'julia_in_py': (
                np.any(python_A_lower <= julia_A_lower + self.tolerance) and
                np.any(python_b_lower <= julia_b_lower + self.tolerance) and
                np.any(python_A_upper >= julia_A_upper - self.tolerance) and
                np.any(python_b_upper >= julia_b_upper - self.tolerance) and
                np.any(python_gap >= julia_gap)
            )
        })
        
        # Store raw values for detailed analysis
        comparison['julia_values'] = {
            'A_lower': julia_A_lower.tolist() if hasattr(julia_A_lower, 'tolist') else julia_A_lower,
            'b_lower': float(julia_b_lower),
            'A_upper': julia_A_upper.tolist() if hasattr(julia_A_upper, 'tolist') else julia_A_upper,
            'b_upper': float(julia_b_upper),
            'gap': float(julia_gap)
        }
        
        comparison['python_values'] = {
            'A_lower': python_A_lower.tolist() if hasattr(python_A_lower, 'tolist') else python_A_lower,
            'b_lower': float(python_b_lower),
            'A_upper': python_A_upper.tolist() if hasattr(python_A_upper, 'tolist') else python_A_upper,
            'b_upper': float(python_b_upper),
            'gap': float(python_gap)
        }
        
        return comparison
    
    def compare_taylor_expansions(self, expansion_point, delta):
        """
        Compare Taylor expansions directly from both implementations.
        
        :param expansion_point: Point around which to expand
        :param delta: Radius of expansion domain
        :return: Dictionary with comparison of Taylor expansions
        """
        try:
            # Get Julia expansion (returns tuple format)
            julia_lower, julia_upper = first_order_certified_taylor_expansion(
                self.dynamics, expansion_point, delta
            )
            
            # Get Python expansion (returns CertifiedFirstOrderTaylorExpansion)
            python_expansion = first_order_certified_taylor_expansion_python(
                self.dynamics, expansion_point, delta
            )
            
            # Extract Python components
            python_jacobian, python_f_c = python_expansion.linear_approximation
            python_r_lower, python_r_upper = python_expansion.remainder
            
            # Extract Julia components
            julia_f_c_lower, julia_jacobian_lower, julia_r_lower_j = julia_lower
            julia_f_c_upper, julia_jacobian_upper, julia_r_upper_j = julia_upper
            
            # Compare function values at center
            f_c_diff_lower = np.abs(julia_f_c_lower.flatten() - python_f_c.flatten())
            f_c_diff_upper = np.abs(julia_f_c_upper.flatten() - python_f_c.flatten())
            
            # Compare Jacobians
            jacobian_diff_lower = np.abs(julia_jacobian_lower - python_jacobian)
            jacobian_diff_upper = np.abs(julia_jacobian_upper - python_jacobian)
            
            # Compare remainders
            remainder_lower_diff = np.abs(julia_r_lower_j.flatten() - python_r_lower.flatten())
            remainder_upper_diff = np.abs(julia_r_upper_j.flatten() - python_r_upper.flatten())
            
            return {
                'expansion_point': expansion_point.tolist(),
                'delta': delta.tolist(),
                'f_c_max_diff_lower': np.max(f_c_diff_lower),
                'f_c_max_diff_upper': np.max(f_c_diff_upper),
                'jacobian_max_diff_lower': np.max(jacobian_diff_lower),
                'jacobian_max_diff_upper': np.max(jacobian_diff_upper),
                'remainder_max_diff_lower': np.max(remainder_lower_diff),
                'remainder_max_diff_upper': np.max(remainder_upper_diff),
                'within_tolerance': (
                    np.max(f_c_diff_lower) < self.tolerance and
                    np.max(f_c_diff_upper) < self.tolerance and
                    np.max(jacobian_diff_lower) < self.tolerance and
                    np.max(jacobian_diff_upper) < self.tolerance and
                    np.max(remainder_lower_diff) < self.tolerance and
                    np.max(remainder_upper_diff) < self.tolerance
                )
            }
            
        except Exception as e:
            return {
                'expansion_point': expansion_point.tolist(),
                'delta': delta.tolist(),
                'error': str(e),
                'status': 'error'
            }
    
    def get_comparison_summary(self):
        """
        Get a summary of all comparisons performed.
        
        :return: Dictionary with summary statistics
        """
        if not self.comparison_results:
            return {'message': 'No comparisons performed yet'}
        
        successful_comparisons = [r for r in self.comparison_results if r.get('status') == 'compared']
        error_comparisons = [r for r in self.comparison_results if r.get('status') == 'error']
        
        if not successful_comparisons:
            return {
                'total_samples': len(self.comparison_results),
                'successful_comparisons': 0,
                'error_comparisons': len(error_comparisons),
                'message': 'No successful comparisons'
            }
        
        within_tolerance = sum(1 for r in successful_comparisons if r.get('within_tolerance', False))
        
        # Compute statistics for differences
        A_lower_diffs = [r['A_lower_max_diff'] for r in successful_comparisons]
        b_lower_diffs = [r['b_lower_diff'] for r in successful_comparisons]
        A_upper_diffs = [r['A_upper_max_diff'] for r in successful_comparisons]
        b_upper_diffs = [r['b_upper_diff'] for r in successful_comparisons]
        gap_diffs = [r['gap_diff'] for r in successful_comparisons]
        
        return {
            'total_samples': len(self.comparison_results),
            'successful_comparisons': len(successful_comparisons),
            'error_comparisons': len(error_comparisons),
            'within_tolerance_count': within_tolerance,
            'tolerance_rate': within_tolerance / len(successful_comparisons),
            'max_differences': {
                'A_lower': np.max(A_lower_diffs) if A_lower_diffs else 0,
                'b_lower': np.max(b_lower_diffs) if b_lower_diffs else 0,
                'A_upper': np.max(A_upper_diffs) if A_upper_diffs else 0,
                'b_upper': np.max(b_upper_diffs) if b_upper_diffs else 0,
                'gap': np.max(gap_diffs) if gap_diffs else 0
            },
            'mean_differences': {
                'A_lower': np.mean(A_lower_diffs) if A_lower_diffs else 0,
                'b_lower': np.mean(b_lower_diffs) if b_lower_diffs else 0,
                'A_upper': np.mean(A_upper_diffs) if A_upper_diffs else 0,
                'b_upper': np.mean(b_upper_diffs) if b_upper_diffs else 0,
                'gap': np.mean(gap_diffs) if gap_diffs else 0
            }
        }
    
    def print_comparison_summary(self):
        """Print a formatted summary of the comparison results."""
        summary = self.get_comparison_summary()
        
        print("=== Taylor Linearization Comparison Summary ===")
        print(f"Total samples: {summary.get('total_samples', 0)}")
        print(f"Successful comparisons: {summary.get('successful_comparisons', 0)}")
        print(f"Error comparisons: {summary.get('error_comparisons', 0)}")
        
        if 'tolerance_rate' in summary:
            print(f"Within tolerance: {summary['within_tolerance_count']}/{summary['successful_comparisons']} ({summary['tolerance_rate']:.2%})")
            print(f"Tolerance used: {self.tolerance}")
            
            print("\nMaximum differences:")
            for key, value in summary['max_differences'].items():
                print(f"  {key}: {value:.2e}")
            
            print("\nMean differences:")
            for key, value in summary['mean_differences'].items():
                print(f"  {key}: {value:.2e}")
        
        print("=" * 50)
