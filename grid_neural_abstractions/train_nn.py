import numpy as np  # Add numpy for grid generation
import torch
import torch.nn as nn
import torch.optim as optim
from grid_neural_abstractions.dynamics import VanDerPolOscillator, Quadcopter


# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size, device=self.device))  # Create layer on device
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size, device=self.device))  # Create layer on device
        self.network = nn.Sequential(*layers)
        self._initialize_weights()  # Add weight initialization

    def forward(self, x):
        return self.network(x)

    def _initialize_weights(self):  # Initialize weights for better training
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)


# Generate some synthetic data for training
def generate_data(input_size, input_domain, delta=0.01, grid=False, batch_size=256, dynamics_model=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Generate data points for training or verification.
    If grid=True, generate a fixed grid of points with spacing at most delta.
    If grid=False, randomly sample data points.
    
    Args:
        input_size: Dimension of input space
        input_domain: List of tuples [(min_1, max_1), ..., (min_n, max_n)] defining the domain for each dimension
                      If None, defaults to [(-1.0, 1.0)] * input_size
        delta: Grid spacing when grid=True
        grid: Whether to use grid sampling (True) or random sampling (False)
        batch_size: Number of samples when grid=False
        dynamics_model: The dynamics model to use for generating outputs
        device: The device to place tensors on
        
    Returns:
        X_train: Input data with shape [input_dim, batch_size]
        y_train: Output data with shape [input_dim, batch_size]
    """
    
    # Ensure domain size matches input_size
    assert len(input_domain) == input_size, f"Input domain size {len(input_domain)} must match input size {input_size}"

    if grid:
        # Generate grid points for each dimension based on its domain
        grid_points_per_dim = []
        for i in range(input_size):
            min_val, max_val = input_domain[i]
            # Remove edge of domain, as this is covered by the hypercubes
            min_val = min_val + delta[i] 
            max_val = max_val - delta[i]
            num_points = int(np.ceil((max_val - min_val) / (2 * delta[i]))) + 1
            grid_points_per_dim.append(np.linspace(min_val, max_val, num_points))
        
        # Create meshgrid from the points
        mesh = np.meshgrid(*grid_points_per_dim)
        X_train = np.vstack(list(map(np.ravel, mesh))).T
        X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    else:
        # Randomly sample points within each dimension's domain
        X_train = torch.zeros(input_size, batch_size, device=device)
        for i in range(input_size):
            min_val, max_val = input_domain[i]
            X_train[i] = (max_val - min_val) * torch.rand(batch_size, device=device) + min_val

    if dynamics_model is None:
        y_train = None
    else:
        # Get outputs in [output_dim, batch_size] format
        y_train = dynamics_model(X_train)
    
    return X_train, y_train


# Train the neural network
def train_nn(dynamics_model=None, epsilon = 0.05, hidden_sizes=[128,128,128], learning_rate = 0.001, num_epochs = 50000, batch_size = 4096):
    if dynamics_model is None:
        dynamics_model = Quadcopter()
    
    input_size = dynamics_model.input_dim
    output_size = dynamics_model.output_dim  # Update output size to match target size
    input_domain = dynamics_model.input_domain  # Get input domain from dynamics model
    
    # Add parameters for gradient clipping and early stopping
    max_grad_norm = 1.0
    patience = 1000
    best_loss = float('inf')
    patience_counter = 0

    model = SimpleNN(input_size, hidden_sizes, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Use AdamW optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=200, min_lr=1e-7
    )

    # Load data
    for epoch in range(num_epochs):
        X_train, y_train = generate_data(input_size, input_domain, batch_size=batch_size, dynamics_model=dynamics_model, device=model.device)

        model.train()
        outputs = model(X_train.T)
        max_loss = torch.max(torch.abs(outputs - y_train.T))
        loss = criterion(outputs, y_train.T)
            
        optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        scheduler.step(loss)  # Update learning rate based on loss

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}, Max: {max_loss.item():.6f},LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
            
        # Early stopping logic
        # Use max_loss for early stopping to ensure the maximum error across all predictions is minimized,
        # which is critical for applications requiring strict error bounds.
        if max_loss < best_loss:
            best_loss = max_loss
            if best_loss > epsilon:
                patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience and best_loss < epsilon:
            print(f"Early stopping triggered at epoch {epoch+1}. Max loss: {best_loss:.6f}")
            break

    return model


# Save the trained model as an ONNX file
def save_onnx_model(model, file_name="data/simple_nn.onnx"):
    # Infer input size from the model if not provided
    input_size = model.network[0].in_features

    dummy_input = torch.randn(1, input_size, dtype=torch.float32, device=model.device)
    torch.onnx.export(
        model,
        dummy_input,
        file_name,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,  # Set to False as dynamo=True may not be supported in all cases
    )
    print(f"Model saved as {file_name}")


def save_torch_model(model, file_name="data/simple_nn.pth"):
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")


def save_model(model, file_name="data/simple_nn.onnx"):
    save_onnx_model(model, file_name)
    save_torch_model(model, file_name.replace(".onnx", ".pth"))


# Load the ONNX model for Marabou
def load_onnx_model(file_name="data/simple_nn.onnx"):
    from maraboupy import Marabou

    network = Marabou.read_onnx(file_name)
    print(f"ONNX model {file_name} loaded for Marabou")
    return network


def load_torch_model(file_name="data/simple_nn.pth", input_size=3, hidden_sizes=[128, 128, 128], output_size=3):
    model = SimpleNN(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
    model.load_state_dict(torch.load(file_name, map_location=torch.device('cpu')))
    print(f"PyTorch model {file_name} loaded")
    return model


if __name__ == "__main__":
    model = train_nn()  # Use the correct input_size from the dynamics model

    # Save the trained model
    save_model(model)
