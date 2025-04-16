import sys
import unittest
import inspect
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from grid_neural_abstractions.dynamics import DynamicalSystem, NonlinearOscillator, WaterTank
from grid_neural_abstractions.train_nn import train_nn, save_onnx_model, generate_data
from grid_neural_abstractions.verify_nn import verify_nn


def get_1D_systems():
    """
    Get all 1D dynamics systems defined in the dynamics module.
    """
    from grid_neural_abstractions import dynamics
    
    # Find all classes that inherit from DynamicalSystem and have input_dim=1
    dynamics_systems = []
    for name, obj in inspect.getmembers(dynamics):
        if (inspect.isclass(obj) and 
            issubclass(obj, DynamicalSystem) and 
            obj != DynamicalSystem):
            # Create an instance to check dimensions
            instance = obj()
            if instance.input_dim == 1:
                dynamics_systems.append((name, obj))
    
    return dynamics_systems


def plot_1D_dynamics(dynamics_model, output_path=None):
    """
    Visualize the 1D dynamics of a system by plotting dx/dt vs x.
    """
    # Get the domain
    x_min, x_max = dynamics_model.input_domain[0]
    
    # Create x values
    x_values = np.linspace(x_min, x_max, 100)
    
    # Reshape for dynamics computation
    x_reshaped = np.reshape(x_values, (1, -1))
    
    # Compute derivatives
    dx_dt = dynamics_model(x_reshaped)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, dx_dt[0], 'b-', linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.grid(True)
    plt.title(f"1D Dynamics: {dynamics_model.__class__.__name__}")
    plt.xlabel("x")
    plt.ylabel("dx/dt")
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    
    plt.close()


def model_exists(dynamics_name):
    """
    Check if a trained model exists for the given dynamics system.
    """
    model_dir = Path(__file__).parent.parent / "data"
    model_path = model_dir / f"{dynamics_name.lower()}_model.onnx"
    return model_path.exists()


def verify_dynamics_model(model_path, dynamics_model):
    """
    Verify the trained model against the dynamics system.
    """
    print(f"Verifying model: {model_path}")
    # Use a larger delta and epsilon for faster verification in tests
    delta = 0.2
    epsilon = 0.2
    agg = verify_nn(model_path, delta=delta, epsilon=epsilon, num_workers=1, dynamics_model=dynamics_model)
    return agg


class Test1DModels(unittest.TestCase):
    
    def setUp(self):
        # Create necessary directories
        self.model_dir = Path(__file__).parent.parent / "data"
        self.model_dir.mkdir(exist_ok=True)
        
        self.plot_dir = Path(__file__).parent.parent / "plots"
        self.plot_dir.mkdir(exist_ok=True)
    
    def test_1D_systems(self):
        dynamics_systems = get_1D_systems()
        self.assertTrue(len(dynamics_systems) > 0, "No 1D dynamics systems found")
        
        model_dir = self.model_dir
        
        for name, dynamics_class in dynamics_systems:
            print(f"\nTesting 1D dynamics system: {name}")
            
            # Model file path
            model_path = model_dir / f"{name.lower()}_model.onnx"
            
            # Initialize the dynamics model
            dynamics_instance = dynamics_class()
            
            # Plot the dynamics
            plot_path = self.plot_dir / f"{name.lower()}_dynamics.png"
            plot_1D_dynamics(dynamics_instance, str(plot_path))
            print(f"Plot saved to {plot_path}")
            
            # Check if model exists or train one
            if not model_exists(name):
                print(f"Training model for {name}...")
                
                # Using the train_nn function but with the specific dynamics model
                model = train_nn(dynamics_model=dynamics_instance)
                
                # Save the model with the specific name
                save_onnx_model(model, str(model_path))
            else:
                print(f"Using existing model: {model_path}")
            
            # Verify the model
            verification_result = verify_dynamics_model(str(model_path), dynamics_instance)
            
            # Just report the result
            print(f"Verification result for {name}: {'Passed' if verification_result is None else 'Failed'}")


if __name__ == "__main__":
    unittest.main()
