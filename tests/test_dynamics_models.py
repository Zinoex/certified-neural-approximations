import sys
import unittest
import inspect
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from grid_neural_abstractions.dynamics import DynamicalSystem
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
    model_dir = Path(__file__).parent.parent / "data"
    model_path = model_dir / f"{dynamics_name.lower()}_model.onnx"
    return model_path.exists()


def verify_dynamics_model(model_path, dynamics_model, epsilon=0.05):
    """
    Verify the trained model against the dynamics system.
    """
    print(f"Verifying model: {model_path}")
    # Use a larger delta and epsilon for faster verification in tests
    delta = [(high - low) / 2 for low, high in dynamics_model.input_domain]
    verify_nn(model_path, delta=delta, epsilon=epsilon, dynamics_model=dynamics_model)


class TestDynamicsModels(unittest.TestCase):
    
    def test_all_dynamics_systems(self):
        # Create models directory if it doesn't exist
        model_dir = Path(__file__).parent.parent / "data"
        model_dir.mkdir(exist_ok=True)
        
        dynamics_systems = get_all_dynamics_systems()
        
        # Define default parameters for training and verification
        hidden_sizes = [20, 20]
        epsilon = 0.05
        
        for name, dynamics_class in dynamics_systems:
            print(f"\nTesting dynamics system: {name}")
            
            # Model file path
            model_path = model_dir / f"{name.lower()}_model.onnx"
            
            # Initialize the dynamics model
            dynamics_instance = dynamics_class()
            
            # Check if model exists or train one
            if not model_exists(name):
                from grid_neural_abstractions.train_nn import train_nn, save_onnx_model
                print(f"Training model for {name}...")
                
                # Using the train_nn function but with the specific dynamics model
                model = train_nn(
                    dynamics_model=dynamics_instance,
                    hidden_sizes=hidden_sizes,
                    epsilon=epsilon/2  # Use a smaller epsilon for training
                )
                
                # Save the model with the specific name
                save_onnx_model(model, str(model_path))
            else:
                print(f"Using existing model: {model_path}")
            
            # Verify the model
            verify_dynamics_model(
                str(model_path), 
                dynamics_instance, 
                epsilon=epsilon
            )
    
    def test_sine2d_system(self):
        """Test specifically for the Sine2D system."""
        from grid_neural_abstractions.dynamics import Sine2D
        
        # Create models directory if it doesn't exist
        model_dir = Path(__file__).parent.parent / "data"
        model_dir.mkdir(exist_ok=True)
        
        # Test with default frequencies
        self._test_sine2d_with_freq(model_dir, 1.0, 1.0, "sine2d")
        
        # Test with custom frequencies
        self._test_sine2d_with_freq(model_dir, 2.0, 0.5, "sine2d_custom_freq")
    
    def _test_sine2d_with_freq(self, model_dir, freq_x, freq_y, model_name):
        """Helper method to test Sine2D with specific frequencies."""
        from grid_neural_abstractions.dynamics import Sine2D
        
        # Define default parameters for training and verification
        hidden_sizes = [10, 10]
        epsilon = 0.05
        
        # Model file path
        model_path = model_dir / f"{model_name}_model.onnx"
        
        # Initialize the dynamics model with specified frequencies
        dynamics_instance = Sine2D(freq_x=freq_x, freq_y=freq_y)
        
        print(f"\nTesting Sine2D system with freq_x={freq_x}, freq_y={freq_y}")
        
        # Check if model exists or train one
        if not model_path.exists():
            from grid_neural_abstractions.train_nn import train_nn, save_onnx_model
            print(f"Training model for Sine2D...")
            
            # Generate data and train the neural network
            model = train_nn(
                dynamics_model=dynamics_instance,
                hidden_sizes=hidden_sizes,
                epsilon=epsilon/2  # Use a smaller epsilon for training
            )
            
            # Save the model
            save_onnx_model(model, str(model_path))
        else:
            print(f"Using existing model: {model_path}")
        
        # Verify the model
        verify_dynamics_model(
            str(model_path), 
            dynamics_instance, 
            epsilon=epsilon
        )
    
    def test_jet_engine(self):
        """Test specifically for the JetEngine system."""        
        from grid_neural_abstractions.dynamics import JetEngine
        
        # Create models directory if it doesn't exist
        model_dir = Path(__file__).parent.parent / "data"
        model_dir.mkdir(exist_ok=True)
        
        # Model file path
        model_path = model_dir / f"jetengine_model.onnx"
        
        # Initialize the dynamics model
        dynamics_instance = JetEngine()
        epsilon = 0.05

        print(f"\nTesting Jet Engine system")
        
        # Check if model exists or train one
        if not model_path.exists():
            from grid_neural_abstractions.train_nn import train_nn, save_onnx_model
            print(f"Training model for Jet Engine...")
            
            # Generate data and train the neural network
            # Use a specific network architecture for the Jet Engine
            model = train_nn(
                dynamics_model=dynamics_instance,
                hidden_sizes=[15,15],
                epsilon=epsilon/2,  # Use a smaller epsilon for training
            )
            
            # Save the model
            save_onnx_model(model, str(model_path))
        else:
            print(f"Using existing model: {model_path}")
        
        # Verify the model
        verify_dynamics_model(str(model_path), dynamics_instance, epsilon=epsilon)
    
    def test_quadcopter(self):
        """Test specifically for the Quadcopter system."""        
        from grid_neural_abstractions.dynamics import Quadcopter
        
        import torch
        # Check if CUDA is available
        print("Cuda device available:", torch.cuda.is_available())

        # Create models directory if it doesn't exist
        model_dir = Path(__file__).parent.parent / "data"
        model_dir.mkdir(exist_ok=True)
        
        # Model file path
        model_path = model_dir / f"quadcopter_model.onnx"
        
        # Initialize the dynamics model
        dynamics_instance = Quadcopter()
        epsilon = 0.25  # Use a slightly larger epsilon due to system complexity

        print(f"\nTesting Quadcopter system")
        
        # Check if model exists or train one
        if not model_path.exists():
            from grid_neural_abstractions.train_nn import train_nn, save_onnx_model
            print(f"Training model for Quadcopter...")
            
            # Use a larger network for the complex 12D dynamics
            model = train_nn(
                dynamics_model=dynamics_instance,
                hidden_sizes=[2048, 1024, 1024],
                epsilon=epsilon/2,  # Use a smaller epsilon for training
                batch_size=1024,    # Larger batch size for stability
                num_epochs=5000000     # More epochs for convergence
            )
            
            # Save the model
            save_onnx_model(model, str(model_path))
        else:
            print(f"Using existing model: {model_path}")
        
        # Verify the model
        verify_dynamics_model(str(model_path), dynamics_instance, epsilon=epsilon, nr_initial_samples=2)
    
    def test_aircraft(self):
        """Test specifically for the Aircraft system."""        
        from grid_neural_abstractions.dynamics import Aircraft
        
        # Create models directory if it doesn't exist
        model_dir = Path(__file__).parent.parent / "data"
        model_dir.mkdir(exist_ok=True)
        
        # Model file path
        model_path = model_dir / f"aircraft_model.onnx"
        
        # Initialize the dynamics model
        dynamics_instance = Aircraft()
        
        # Use a larger epsilon for this complex system
        epsilon = 0.08

        print(f"\nTesting Aircraft system")
        
        # Check if model exists or train one
        if not model_path.exists():
            from grid_neural_abstractions.train_nn import train_nn, save_onnx_model
            print(f"Training model for Aircraft...")
            
            # Use a deep network for this complex 12D to 9D system
            model = train_nn(
                dynamics_model=dynamics_instance,
                hidden_sizes=[128, 256, 256],
                epsilon=epsilon/2,  # Use a smaller epsilon for training
                batch_size=2048,    # Larger batch size for stability
                num_epochs=25000    # More epochs for convergence
            )
            
            # Save the model
            save_onnx_model(model, str(model_path))
        else:
            print(f"Using existing model: {model_path}")
        
        # Verify the model
        verify_dynamics_model(str(model_path), dynamics_instance, epsilon=epsilon)


if __name__ == "__main__":
    unittest.main()
