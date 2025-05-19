import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from certified_neural_approximations.dynamics import NNDynamics, LorenzAttractor
from certified_neural_approximations.train_nn import load_torch_model
from certified_neural_approximations.translators import NumpyTranslator, TorchTranslator


@torch.no_grad()
def plot_compressed_dynamics(true_dynamics, large_dynamics, compressed_dynamics, x0, dt=0.02, trajectory_length=700):
    # True dynamics
    np_translator = NumpyTranslator()

    def dynamics(t, x):
        return true_dynamics.compute_dynamics(x, translator=np_translator)

    res = solve_ivp(
            dynamics,
            (0.0, trajectory_length * dt),
            x0,
            t_eval=np.linspace(0, trajectory_length * dt, trajectory_length + 1),
            method="RK45",
            rtol=1e-6,
            atol=1e-6,
        )
    
    x_true = res.y.T

    # Large dynamics
    torch_translator = TorchTranslator()

    x_large = torch.zeros((trajectory_length, len(x0)))
    x_large[0] = torch.as_tensor(x0, dtype=torch.float32)

    for i in range(1, trajectory_length):
        x_large[i] = large_dynamics.compute_dynamics(x_large[i - 1], torch_translator)

    x_large = x_large.numpy()

    # Compressed dynamics
    x_compressed = torch.zeros((trajectory_length, len(x0)))
    x_compressed[0] = torch.as_tensor(x0, dtype=torch.float32)

    for i in range(1, trajectory_length):
        x_compressed[i] = compressed_dynamics.compute_dynamics(x_compressed[i - 1], torch_translator)

    x_compressed = x_compressed.numpy()

    # Plotting 3D trajectories
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_true[:, 0], x_true[:, 1], x_true[:, 2], label='True Dynamics', color='blue')
    ax.plot(x_large[:, 0], x_large[:, 1], x_large[:, 2], label='Large Dynamics', color='red')
    ax.plot(x_compressed[:, 0], x_compressed[:, 1], x_compressed[:, 2], label='Compressed Dynamics', color='green')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.legend()

    plt.show()


if __name__ == "__main__":
    true_dynamics = LorenzAttractor()

    large_network = load_torch_model("data/compression_ground_truth.pth",
                                     input_size=true_dynamics.input_dim,
                                     hidden_sizes=[1024, 1024, 1024, 1024, 1024],
                                     output_size=true_dynamics.output_dim)
    large_network_dynamics = NNDynamics(large_network, true_dynamics.input_domain)

    compressed_network = load_torch_model("data/compression_compressed.pth",
                                          input_size=true_dynamics.input_dim,
                                          hidden_sizes=[128, 128, 128, 128, 128],
                                          output_size=true_dynamics.output_dim)
    compressed_network_dynamics = NNDynamics(compressed_network, true_dynamics.input_domain)

    x0 = np.array([2.0, 1.0, 1.0])

    plot_compressed_dynamics(true_dynamics, large_network_dynamics, compressed_network_dynamics, x0)