import numpy as np
from certified_neural_approximations.dynamics import LorenzAttractor, NNDynamics
from certified_neural_approximations.generate_data import generate_data
from certified_neural_approximations.train_nn import SimpleNN, load_torch_model, train_nn, save_model

import torch
from torch import nn, optim
from scipy.integrate import solve_ivp

from certified_neural_approximations.translators.numpy_translator import NumpyTranslator


def generate_data_from_trajectories(dynamics_model, batch_size=128, dt=0.02, trajectory_length=64, device=None):
    input_size = dynamics_model.input_dim
    input_domain = dynamics_model.input_domain  # Get input domain from dynamics model

    x0, _ = generate_data(input_size, input_domain, batch_size=batch_size, device=torch.device('cpu'))

    translator = NumpyTranslator()

    def dynamics(t, x):
        return dynamics_model.compute_dynamics(x, translator=translator)

    res = [
        solve_ivp(
            dynamics,
            (0.0, trajectory_length * dt),
            x0[:, i].numpy(),
            t_eval=np.linspace(0, trajectory_length * dt, trajectory_length + 1),
            method="RK45",
            rtol=1e-6,
            atol=1e-6,
        )
        for i in range(batch_size)
    ]

    trajectories = np.stack([r.y.T for r in res])
    X_train = torch.tensor(trajectories[:, :-1, :],
                           dtype=torch.float32).reshape(trajectory_length * batch_size, x0.size(0)).to(device)
    y_train = torch.tensor(trajectories[:, 1:, :], dtype=torch.float32).reshape(
        trajectory_length * batch_size, x0.size(0)).to(device)

    return X_train.T, y_train.T


# Train the neural network
def train_nn_from_trajectories(dynamics_model, learning_rate=1e-6, num_epochs=500000, batch_size=128,
                               trajectory_length=32, leaky_relu=False):
    if hasattr(dynamics_model, 'leaky_relu'):
        leaky_relu = dynamics_model.leaky_relu

    input_size = dynamics_model.input_dim
    hidden_sizes = dynamics_model.hidden_sizes  # Get hidden sizes from dynamics model
    output_size = dynamics_model.output_dim  # Update output size to match target size
    epsilon = dynamics_model.epsilon  # Get epsilon from dynamics model

    # Add parameters for gradient clipping and early stopping
    max_grad_norm = 1.0
    patience = 40000
    best_loss = float('inf')
    patience_counter = 0

    # Add variable to store the best model state
    best_model_state = None

    model = SimpleNN(input_size, hidden_sizes, output_size, leaky_relu=leaky_relu)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Use AdamW optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.90, patience=2000, min_lr=1e-8
    )

    # Load data
    for epoch in range(num_epochs):
        if epoch % 50 == 0:
            X_train, y_train = generate_data_from_trajectories(
                dynamics_model, batch_size=batch_size, trajectory_length=trajectory_length, device=model.device)

        model.train()
        outputs = model(X_train.T)
        max_loss = torch.max(torch.abs(outputs - y_train.T))
        avg_loss = criterion(outputs, y_train.T)
        loss = avg_loss + 0.001 * max_loss  # Add max loss to the MSE loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Apply gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step(loss)  # Update learning rate based on loss

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss.item():.6f}, Max: {max_loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.8f}"
            )

        # Early stopping logic and best model tracking
        if max_loss < best_loss and epoch > 2500:
            best_loss = max_loss
            # Save the best model state
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience and best_loss < epsilon:
            print(f"Early stopping triggered at epoch {epoch+1}. Best max loss: {best_loss:.6f}")
            break

    # Restore the best model if we found one during training
    if best_model_state is not None:
        print(f"Restoring best model with max loss: {best_loss:.6f}")
        model.load_state_dict(best_model_state)

    return model


def train_compression(dynamics_model=None, load_ground_truth=False):
    if load_ground_truth:
        model = load_torch_model("data/compression_ground_truth.pth", input_size=dynamics_model.input_dim,
                                 hidden_sizes=[1024, 1024, 1024, 1024, 1024], output_size=dynamics_model.output_dim)
    else:
        dynamics_model.hidden_sizes = [1024, 1024, 1024, 1024, 1024]  # Set hidden sizes for the dynamics model
        model = train_nn_from_trajectories(dynamics_model, learning_rate=1e-6)
        save_model(model, "data/compression_ground_truth.onnx")

    nn_dynamics = NNDynamics(model, dynamics_model.input_domain)
    nn_dynamics.hidden_sizes = [128, 128, 128, 128, 128]  # Set hidden sizes for the compression model
    nn_dynamics.epsilon = dynamics_model.epsilon
    compressed_model = train_nn(nn_dynamics, learning_rate=1e-6, num_epochs=1000000)
    save_model(compressed_model, "data/compression_compressed.onnx")
