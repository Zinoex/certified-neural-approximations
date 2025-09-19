
import os

import torch
from certified_neural_approximations.dynamics import WaterTank, JetEngine, SteamGovernor, Exponential, \
    NonLipschitzVectorField1, NonLipschitzVectorField2
from certified_neural_approximations.dynamics import VanDerPolOscillator, Sine2D, NonlinearOscillator, LowThrustSpacecraft

from certified_neural_approximations.neural_abstractions.cli import get_config
from certified_neural_approximations.neural_abstractions.main import verify_neural_abstractions
from certified_neural_approximations.train_nn import train_nn, save_model
from certified_neural_approximations.verify_nn import verify_nn


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(REPO_DIR, "data")

NA_SYSTEMS = [
    (WaterTank, "watertank/config.yaml"),
    (JetEngine, "jet/config.yaml"),
    (SteamGovernor, "steam/config.yaml"),
    (Exponential, "exp/config.yaml"),
    (NonLipschitzVectorField1, "nl1/config.yaml"),
    (NonLipschitzVectorField2, "nl2/config.yaml"),
]

NEW_SYSTEMS = [
    (VanDerPolOscillator, "vdp/config.yaml"),
    (Sine2D, "sine2d/config.yaml"),
    (NonlinearOscillator, "nonlin-osc/config.yaml"),
    (LowThrustSpacecraft, "low-thrust/config.yaml")
]

SYSTEMS = NA_SYSTEMS + NEW_SYSTEMS


def train_na_models():
    for (dynamics_cls, _) in NA_SYSTEMS:
        dynamics = dynamics_cls()
        torch.manual_seed(0)

        print(f"\nTraining {dynamics.system_name} system")
        model = train_nn(dynamics_model=dynamics)
        path = os.path.join(DATA_DIR, f"{dynamics.system_name}_simple_nn.onnx")

        save_model(model, path)

        print(f"Done training for {dynamics.system_name}")


def train_64_models():
    for (dynamics_cls, _) in SYSTEMS:
        dynamics = dynamics_cls()
        dynamics.hidden_sizes = [64, 64, 64]
        torch.manual_seed(0)

        print(f"\nTraining {dynamics.system_name} system with 3x[64] neurons")
        model = train_nn(dynamics_model=dynamics)
        path = os.path.join(DATA_DIR, f"{dynamics.system_name}_64_simple_nn.onnx")

        save_model(model, path)

        print(f"Done training for {dynamics.system_name}")


def verify_na_models():
    for (dynamics_cls, _) in NA_SYSTEMS:
        dynamics = dynamics_cls()
        path = os.path.join(DATA_DIR, f"{dynamics.system_name}_simple_nn.onnx")

        print(f"\nVerifying model {dynamics.system_name}")
        verify_nn(
            onnx_path=path,
            dynamics_model=dynamics,
            visualize=False
        )
        print(f"Verification completed for {dynamics.system_name} system")


def verify_64_models():
    for (dynamics_cls, _) in SYSTEMS:
        dynamics = dynamics_cls()
        dynamics.hidden_sizes = [64, 64, 64]

        if hasattr(dynamics, 'small_epsilon'):
            dynamics.epsilon = dynamics.small_epsilon

        path = os.path.join(DATA_DIR, f"{dynamics.system_name}_64_simple_nn.onnx")

        print(f"\nVerifying model {dynamics.system_name} with 3x[64] neurons")
        verify_nn(
            onnx_path=path,
            dynamics_model=dynamics,
            visualize=False
        )
        print(f"Verification completed for {dynamics.system_name} system")


def verify_models_dreal_simple():
    for (dynamics_cls, config_path) in SYSTEMS:
        dynamics = dynamics_cls()

        path = os.path.join(DATA_DIR, "neural_abstractions", config_path)

        print(f"\nVerifying model {dynamics.system_name} with DReal")
        c = get_config(["-c", path])
        verify_neural_abstractions(c)
        print(f"Verification completed for {dynamics.system_name} system")


def verify_models_dreal_larger():
    for (dynamics_cls, config_path) in NA_SYSTEMS:
        dynamics = dynamics_cls()

        config_path = config_path.replace("/", "64/")
        path = os.path.join(DATA_DIR, "neural_abstractions", config_path)

        print(f"\nVerifying model {dynamics.system_name} (3x[64]) with DReal")
        c = get_config(["-c", path])
        verify_neural_abstractions(c)
        print(f"Verification completed for {dynamics.system_name} system")


def main(train=False, verify=True):
    os.makedirs(DATA_DIR, exist_ok=True)

    if train:
        train_na_models()
        train_64_models()

    if verify:
        verify_na_models()
        verify_64_models()
        # verify_models_dreal_simple()
        # verify_models_dreal_larger()


if __name__ == "__main__":
    main()
