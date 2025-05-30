import sys
import unittest
import inspect
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from certified_neural_approximations.dynamics import DynamicalSystem
from certified_neural_approximations.verify_nn import verify_nn


def get_all_dynamics_systems():
    """
    Get all dynamics systems defined in the dynamics module.
    """
    from certified_neural_approximations import dynamics
    
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


def verify_dynamics_model(model_path, dynamics_model):
    """
    Verify the trained model against the dynamics system.
    """
    print(f"Verifying model: {model_path}")
    # Use a larger delta and epsilon for faster verification in tests
    verify_nn(model_path, dynamics_model=dynamics_model)

class TestDynamicsModels(unittest.TestCase):
    
    def test_all_dynamics_systems(self):
        # Create models directory if it doesn't exist
        model_dir = Path(__file__).parent.parent / "data"
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
                from certified_neural_approximations.train_nn import train_nn, save_onnx_model
                print(f"Training model for {name}...")
                
                # Using the train_nn function but with the specific dynamics model
                model = train_nn(
                    dynamics_model=dynamics_instance,
                    learning_rate=0.001,
                    num_epochs=50000,
                    batch_size=4096
                )
                
                # Save the model with the specific name
                save_onnx_model(model, str(model_path))
            else:
                print(f"Using existing model: {model_path}")
            
            # Verify the model
            verify_dynamics_model(
                str(model_path), 
                dynamics_instance
            )
    
    def test_sine2d_system(self):
        """Test specifically for the Sine2D system."""
        from certified_neural_approximations.dynamics import Sine2D
        
        # Create models directory if it doesn't exist
        model_dir = Path(__file__).parent.parent / "data"
        model_dir.mkdir(exist_ok=True)
        
        # Test with default frequencies
        self._test_sine2d_with_freq(model_dir, 1.0, 1.0, "sine2d")
        
        # Test with custom frequencies
        self._test_sine2d_with_freq(model_dir, 2.0, 0.5, "sine2d_custom_freq")
    
    def _test_sine2d_with_freq(self, model_dir, freq_x, freq_y, model_name):
        """Helper method to test Sine2D with specific frequencies."""
        from certified_neural_approximations.dynamics import Sine2D
        
        # Model file path
        model_path = model_dir / f"{model_name}_model.onnx"
        
        # Initialize the dynamics model with specified frequencies
        dynamics_instance = Sine2D(freq_x=freq_x, freq_y=freq_y)
        
        print(f"\nTesting Sine2D system with freq_x={freq_x}, freq_y={freq_y}")
        
        # Check if model exists or train one
        if not model_path.exists():
            from certified_neural_approximations.train_nn import train_nn, save_onnx_model
            print(f"Training model for Sine2D...")
            
            # Generate data and train the neural network
            model = train_nn(
                dynamics_model=dynamics_instance,
                learning_rate=0.001,
                num_epochs=50000,
                batch_size=4096
            )
            
            # Save the model
            save_onnx_model(model, str(model_path))
        else:
            print(f"Using existing model: {model_path}")
        
        # Verify the model
        verify_dynamics_model(
            str(model_path), 
            dynamics_instance
        )
    
    def test_jet_engine(self):
        """Test specifically for the JetEngine system."""        
        from certified_neural_approximations.dynamics import JetEngine
        
        # Create models directory if it doesn't exist
        model_dir = Path(__file__).parent.parent / "data"
        model_dir.mkdir(exist_ok=True)
        
        # Model file path
        model_path = model_dir / f"jetengine_model.onnx"
        
        # Initialize the dynamics model
        dynamics_instance = JetEngine()

        print(f"\nTesting Jet Engine system")
        
        # Check if model exists or train one
        if not model_path.exists():
            from certified_neural_approximations.train_nn import train_nn, save_onnx_model
            print(f"Training model for Jet Engine...")
            
            # Generate data and train the neural network
            # Use a specific network architecture for the Jet Engine
            model = train_nn(
                dynamics_model=dynamics_instance,
                learning_rate=0.001,
                num_epochs=50000,
                batch_size=4096
            )
            
            # Save the model
            save_onnx_model(model, str(model_path))
        else:
            print(f"Using existing model: {model_path}")
        
        # Verify the model
        verify_dynamics_model(str(model_path), dynamics_instance)
    
    def test_quadcopter(self):
        """Test specifically for the Quadcopter system."""        
        from certified_neural_approximations.dynamics import Quadcopter

        # Create models directory if it doesn't exist
        model_dir = Path(__file__).parent.parent / "data"
        model_dir.mkdir(exist_ok=True)
        
        # Model file path
        model_path = model_dir / f"quadcopter_model.onnx"
        
        # Initialize the dynamics model
        dynamics_instance = Quadcopter()

        print(f"\nTesting Quadcopter system")
        
        # Check if model exists or train one
        if not model_path.exists():
            from certified_neural_approximations.train_nn import train_nn, save_onnx_model
            print(f"Training model for Quadcopter...")
            
            # Use a larger network for the complex 12D dynamics
            model = train_nn(
                dynamics_model=dynamics_instance,
                learning_rate=0.001,
                batch_size=1024,    # Larger batch size for stability
                num_epochs=5000000  # More epochs for convergence
            )
            
            # Save the model
            save_onnx_model(model, str(model_path))
        else:
            print(f"Using existing model: {model_path}")
        
        # Verify the model
        verify_dynamics_model(str(model_path), dynamics_instance)
    
    def test_vortex_shedding(self):
        """Test specifically for the VortexShedding3D system."""
        from certified_neural_approximations.dynamics import VortexShedding3D
        
        # Create models directory if it doesn't exist
        model_dir = Path(__file__).parent.parent / "data"
        model_dir.mkdir(exist_ok=True)
        
        # Model file path
        model_path = model_dir / f"vortexshedding3d_model.onnx"
        
        # Initialize the dynamics model
        dynamics_instance = VortexShedding3D()
        
        print(f"\nTesting VortexShedding3D system")
        
        # Check if model exists or train one
        if not model_path.exists():
            from certified_neural_approximations.train_nn import train_nn, save_onnx_model
            print(f"Training model for VortexShedding3D...")
            
            # Generate data and train the neural network
            model = train_nn(
                dynamics_model=dynamics_instance,
                learning_rate=0.001,
                num_epochs=50000,
                batch_size=512     # Reasonable batch size for 3D system
            )
            
            # Save the model
            save_onnx_model(model, str(model_path))
        else:
            print(f"Using existing model: {model_path}")
        
        # Verify the model
        verify_dynamics_model(str(model_path), dynamics_instance)
        
    def test_vortex_shedding_4d(self):
        """Test specifically for the VortexShedding4D system."""
        from certified_neural_approximations.dynamics import VortexShedding4D
        
        # Create models directory if it doesn't exist
        model_dir = Path(__file__).parent.parent / "data"
        model_dir.mkdir(exist_ok=True)
        
        # Model file path
        model_path = model_dir / f"vortexshedding4d_model.onnx"
        
        # Initialize the dynamics model
        dynamics_instance = VortexShedding4D()
        
        print(f"\nTesting VortexShedding4D system")
        
        # Check if model exists or train one
        if not model_path.exists():
            from certified_neural_approximations.train_nn import train_nn, save_onnx_model
            print(f"Training model for VortexShedding4D...")
            
            # Generate data and train the neural network
            # Use a larger network for the more complex 4D system
            model = train_nn(
                dynamics_model=dynamics_instance,
                learning_rate=0.001,
                num_epochs=300000,      # More epochs for convergence
                batch_size=512        # Good batch size for training
            )
            
            # Save the model
            save_onnx_model(model, str(model_path))
        else:
            print(f"Using existing model: {model_path}")
        
        # Verify the model
        verify_dynamics_model(str(model_path), dynamics_instance)


if __name__ == "__main__":
    unittest.main()
