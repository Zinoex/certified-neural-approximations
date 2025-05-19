import os
import pickle

import torch

from certified_neural_approximations.dynamics import QuadraticSystem
from certified_neural_approximations.koopman.koopman_verification import train_koopman
from certified_neural_approximations.verify_nn import verify_nn


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(REPO_DIR, "data")


def train_koopman_model():
    torch.manual_seed(0)
    data_path = os.path.join(REPO_DIR, "certified_neural_approximations/koopman/quadratic.pkl")

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    num_epochs = 200
    encoded_size = 64
    encoder_hidden_layers = []
    decoder_hidden_layers = []
    batch_norm = False

    train_koopman(data, data_path, num_epochs, encoded_size, encoder_hidden_layers, decoder_hidden_layers, batch_norm)


def verify_koopman_model():
    dynamics = QuadraticSystem()

    if hasattr(dynamics, 'small_epsilon'):
        dynamics.epsilon = dynamics.small_epsilon

    path = os.path.join(DATA_DIR, "trajpred_quadratic_ver.onnx")

    print(f"\nVerifying model Koopman autoencoder for {dynamics.system_name}")
    verify_nn(
        onnx_path=path,
        dynamics_model=dynamics,
        visualize=False
    )
    print(f"Verification completed for {dynamics.system_name} system")


def main(train=True, verify=True):
    if train:
        # Train Koopman model
        train_koopman_model()

    if verify:
        # Verify the Koopman model
        verify_koopman_model()


if __name__ == "__main__":
    main()
