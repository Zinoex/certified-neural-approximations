import numpy as np  # Add numpy for grid generation
import torch
import torch.nn as nn
import torch.optim as optim
from .dynamics import VanDerPolOscillator, Quadcopter


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
def generate_data(input_size, delta=0.01, grid=False, batch_size=256, dynamics_model=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Generate data points for training or verification.
    If grid=True, generate a fixed grid of points with spacing at most delta.
    If grid=False, randomly sample data points.
    
    Args:
        input_size: Dimension of input space
        delta: Grid spacing when grid=True
        grid: Whether to use grid sampling (True) or random sampling (False)
        batch_size: Number of samples when grid=False
        dynamics_model: The dynamics model to use for generating outputs
        
    Returns:
        X_train: Input data
        y_train: Output data (dynamics evaluated at input points)
    """

    if grid:
        # Generate a grid of points within a fixed range
        range_min, range_max = -1.0, 1.0  # Define the range for each dimension
        num_points_per_dim = int((range_max - range_min) / delta) + 1
        grid_points = np.linspace(range_min, range_max, num_points_per_dim)
        mesh = np.meshgrid(*[grid_points] * input_size)
        X_train = np.vstack(list(map(np.ravel, mesh))).T  # Convert map to list
        X_train = torch.tensor(X_train, dtype=torch.float32, device=device)  # Move to device
    else:
        # Randomly sample points
        X_train = torch.randn(batch_size, input_size, device=device)  # Move to device

    if dynamics_model is None:
        y_train = None
    else:
        y_train = dynamics_model(X_train.T).T.to(device)  # Ensure y_train is on the same device
    return X_train, y_train


# Train the neural network
def train_nn():
    dynamics_model = Quadcopter()
    input_size = dynamics_model.input_dim
    hidden_sizes = [128, 128, 128]  # Adjust hidden layer sizes
    output_size = dynamics_model.output_dim  # Update output size to match target size
    num_epochs = 50000
    learning_rate = 0.001  # Reduced learning rate
    batch_size = 2048

    model = SimpleNN(input_size, hidden_sizes, output_size) # Move model to device
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=10000, gamma=0.5
    )  # Add learning rate scheduler

    # Load data
    for epoch in range(num_epochs):
        X_train, y_train = generate_data(input_size, batch_size=batch_size, dynamics_model=dynamics_model, device=model.device)
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate

        if (epoch + 1) % 1000 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    # Save the trained model
    save_onnx_model(model)
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


# Load the ONNX model for Marabou
def load_onnx_model(file_name="data/simple_nn.onnx"):
    from maraboupy import Marabou

    network = Marabou.read_onnx(file_name)
    print(f"ONNX model {file_name} loaded for Marabou")
    return network


if __name__ == "__main__":
    model = train_nn()  # Use the correct input_size from the dynamics model
