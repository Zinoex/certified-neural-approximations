
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


def train_and_verify_dynamics_model(dynamics_cls):
    dynamics = dynamics_cls()

    print(f"\nTraining and verifying {dynamics.system_name} system")
    model = train_nn(dynamics_model=dynamics)
    path = os.path.join(DATA_DIR, f"{dynamics.system_name}_simple_nn.onnx")

    save_model(model, path)

    print("Verifying model...")
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

    for dynamics_cls in SYSTEMS:
        try:
            train_and_verify_dynamics_model(dynamics_cls)
        except Exception as e:
            print(f"Error processing {dynamics_cls.__name__}: {e}")


if __name__ == "__main__":
    main()
