import numpy as np
import torch


def generate_grid(input_size, input_domain, delta=0.01, batch_size=256, dynamics_model=None):
    """
    Generate data points for training or verification.
    If grid=True, generate a fixed grid of points with spacing at most delta.
    
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

    if dynamics_model is None:
        y_train = None
    else:
        # Get outputs in [output_dim, batch_size] format
        y_train = dynamics_model(X_train)
    
    return X_train, y_train


# Generate some synthetic data for training
def generate_data(input_size, input_domain, delta=0.01, batch_size=256, dynamics_model=None, device=None):
    """
    Generate data points for training or verification.
    If grid=True, generate a fixed grid of points with spacing at most delta.
    
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
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure domain size matches input_size
    assert len(input_domain) == input_size, f"Input domain size {len(input_domain)} must match input size {input_size}"

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