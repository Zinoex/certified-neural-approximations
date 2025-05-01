
import os
import time
from grid_neural_abstractions.dynamics import WaterTank, JetEngine, SteamGovernor, Exponential, \
    NonLipschitzVectorField1, NonLipschitzVectorField2
from grid_neural_abstractions.dynamics import VanDerPolOscillator, Quadcopter, Sine2D, NonlinearOscillator

from grid_neural_abstractions.train_nn import train_nn, save_model
from grid_neural_abstractions.verify_nn import verify_nn


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(REPO_DIR, "data")

NA_SYSTEMS = [
    WaterTank,
    JetEngine,
    SteamGovernor,
    Exponential,
    NonLipschitzVectorField1,
    NonLipschitzVectorField2,
]

NEW_SYSTEMS = [
    VanDerPolOscillator,
    # Quadcopter,
    Sine2D,
    NonlinearOscillator,
]

SYSTEMS = NA_SYSTEMS + NEW_SYSTEMS

def train_models():
    for dynamics_cls in SYSTEMS:
        dynamics = dynamics_cls()

        print(f"\nTraining {dynamics.system_name} system")
        model = train_nn(dynamics_model=dynamics)
        path = os.path.join(DATA_DIR, f"{dynamics.system_name}_simple_nn.onnx")

        save_model(model, path)

        print(f"Done training for {dynamics.system_name}")


def train_na_128_models():
    for dynamics_cls in NA_SYSTEMS:
        dynamics = dynamics_cls()
        dynamics.hidden_sizes = [128, 128, 128]

        print(f"\nTraining {dynamics.system_name} system with 3x[128] neurons")
        model = train_nn(dynamics_model=dynamics)
        path = os.path.join(DATA_DIR, f"{dynamics.system_name}_128_simple_nn.onnx")

        save_model(model, path)

        print(f"Done training for {dynamics.system_name}")


def verify_models():
    for dynamics_cls in SYSTEMS:
        dynamics = dynamics_cls()
        path = os.path.join(DATA_DIR, f"{dynamics.system_name}_simple_nn.onnx")

        print(f"\nVerifying model {dynamics.system_name}")
        t1 = time.time()
        verify_nn(
            onnx_path=path,
            dynamics_model=dynamics,
            visualize=False
        )
        t2 = time.time()
        print(f"Verification completed for {dynamics.system_name} system, took {t2 - t1:.2f} seconds")


def verify_na_128_models():
    for dynamics_cls in NA_SYSTEMS:
        dynamics = dynamics_cls()
        dynamics.hidden_sizes = [128, 128, 128]
        path = os.path.join(DATA_DIR, f"{dynamics.system_name}_128_simple_nn.onnx")

        print(f"\nVerifying model {dynamics.system_name} with 3x[128] neurons")
        t1 = time.time()
        verify_nn(
            onnx_path=path,
            dynamics_model=dynamics,
            visualize=False
        )
        t2 = time.time()
        print(f"Verification completed for {dynamics.system_name} system, took {t2 - t1:.2f} seconds")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # train_models()
    # train_na_128_models()
    # verify_models()
    verify_na_128_models()


if __name__ == "__main__":
    main()
