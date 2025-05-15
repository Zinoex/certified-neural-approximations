import os
import time
from certified_neural_approximations.dynamics import WaterTank, JetEngine, SteamGovernor, Exponential, \
    NonLipschitzVectorField1, NonLipschitzVectorField2
from certified_neural_approximations.dynamics import VanDerPolOscillator, LowThrustSpacecraft, Sine2D, NonlinearOscillator, QuadraticSystem
from certified_neural_approximations.verify_nn import verify_nn
from certified_neural_approximations.train_nn import train_nn, save_model


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
    Sine2D,
    NonlinearOscillator,
    LowThrustSpacecraft,
]

SYSTEMS = NA_SYSTEMS + NEW_SYSTEMS


def train_na_models(leaky_relu=False):
    leaky_relu_path = 'leaky_relu' if leaky_relu else ''

    for dynamics_cls in NA_SYSTEMS:
        dynamics = dynamics_cls()

        print(f"\nTraining {dynamics.system_name} system")
        model = train_nn(dynamics_model=dynamics, leaky_relu=leaky_relu)
        path = os.path.join(DATA_DIR, f"{dynamics.system_name}_simple_nn{leaky_relu_path}.onnx")

        save_model(model, path)

        print(f"Done training for {dynamics.system_name}")


def train_64_models(residual=False, leaky_relu=False):
    residual_path = '_residual' if residual else ''
    leaky_relu_path = 'leaky_relu' if leaky_relu else ''

    for dynamics_cls in SYSTEMS:
        dynamics = dynamics_cls()
        dynamics.hidden_sizes = [64, 64, 64]

        print(f"\nTraining {dynamics.system_name} system with 3x[64] neurons")
        model = train_nn(dynamics_model=dynamics, residual=residual, leaky_relu=leaky_relu)
        path = os.path.join(DATA_DIR, f"{dynamics.system_name}_64_simple_nn{residual_path}{leaky_relu_path}.onnx")

        save_model(model, path)

        print(f"Done training for {dynamics.system_name}")


def verify_na_models(leaky_relu=False):
    leaky_relu_path = 'leaky_relu' if leaky_relu else ''

    for dynamics_cls in NA_SYSTEMS:
        dynamics = dynamics_cls()
        path = os.path.join(DATA_DIR, f"{dynamics.system_name}_simple_nn{leaky_relu_path}.onnx")

        print(f"\nVerifying model {dynamics.system_name}")
        t1 = time.time()
        verify_nn(
            onnx_path=path,
            dynamics_model=dynamics,
            visualize=False
        )
        t2 = time.time()
        print(f"Verification completed for {dynamics.system_name} system, took {t2 - t1:.2f} seconds")


def verify_64_models(residual=False, leaky_relu=False):
    residual_path = '_residual' if residual else ''
    leaky_relu_path = 'leaky_relu' if leaky_relu else ''

    for dynamics_cls in SYSTEMS:
        dynamics = dynamics_cls()
        dynamics.hidden_sizes = [64, 64, 64]

        if hasattr(dynamics, 'small_epsilon'):
            dynamics.epsilon = dynamics.small_epsilon

        path = os.path.join(DATA_DIR, f"{dynamics.system_name}_64_simple_nn{residual_path}{leaky_relu_path}.onnx")

        print(f"\nVerifying model {dynamics.system_name} with 3x[64] neurons")
        t1 = time.time()
        verify_nn(
            onnx_path=path,
            dynamics_model=dynamics,
            visualize=False
        )
        t2 = time.time()
        print(f"Verification completed for {dynamics.system_name} system, took {t2 - t1:.2f} seconds including precompilation and process spawn time")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # train_na_models()
    # train_64_models()
    verify_na_models()
    verify_64_models()


if __name__ == "__main__":
    main()
