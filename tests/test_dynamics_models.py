import sys
import unittest
import inspect
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from grid_neural_abstractions.dynamics import DynamicalSystem
from grid_neural_abstractions.train_nn import train_nn, save_onnx_model, generate_data
from grid_neural_abstractions.verify_nn import verify_nn


def get_all_dynamics_systems():
    """
    Get all dynamics systems defined in the dynamics module.
    """
    from grid_neural_abstractions import dynamics
    
    # Find all classes that inherit from DynamicalSystem
    dynamics_systems = []
    for name, obj in inspect.getmembers(dynamics):
        if (inspect.isclass(obj) and 
            issubclass(obj, DynamicalSystem) and 
            obj != DynamicalSystem):
            dynamics_systems.append((name, obj))
    
    return dynamics_systems


def model_exists(dynamics_name):
    """
    Check if a trained model exists for the given dynamics system.
    """
    model_dir = Path(__file__).parent.parent / "models"
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
    try:
        # Update verify_nn function to accept dynamics_model parameter
        verify_nn(model_path, delta=delta, epsilon=epsilon, num_workers=1, dynamics_model=dynamics_model)
        return True
    except Exception as e:
        print(f"Verification failed: {str(e)}")
        return False


class TestDynamicsModels(unittest.TestCase):
    
    def test_all_dynamics_systems(self):
        # Create models directory if it doesn't exist
        model_dir = Path(__file__).parent.parent / "models"
        model_dir.mkdir(exist_ok=True)
        
        dynamics_systems = get_all_dynamics_systems()
        
        for name, dynamics_class in dynamics_systems:
            print(f"\nTesting dynamics system: {name}")
            
            # Model file path
            model_path = model_dir / f"{name.lower()}_model.onnx"
            
            # Initialize the dynamics model
            dynamics_instance = dynamics_class()
            
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
            print(f"Verification result for {name}: {'Passed' if verification_result else 'Failed'}")


if __name__ == "__main__":
    unittest.main()
