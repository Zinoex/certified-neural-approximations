from grid_neural_abstractions.dynamics import Quadcopter, NNDynamics
from grid_neural_abstractions.train_nn import train_nn, save_model


def train_compression(dynamics_model=None):
    model = train_nn(dynamics_model, hidden_sizes=[1024, 1024, 1024, 1024], learning_rate=1e-4)
    save_model(model, "data/compression_ground_truth.onnx")

    nn_dynamics = NNDynamics(model, dynamics_model.input_domain)
    compressed_model = train_nn(nn_dynamics, hidden_sizes=[128, 128, 128])
    save_model(compressed_model, "data/compression_compressed.onnx")


if __name__ == "__main__":
    # Train compression model
    dynamics_model = Quadcopter()
    train_compression(dynamics_model)
