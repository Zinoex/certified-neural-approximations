

from certified_neural_approximations.verify_compression import verify_compression
from certified_neural_approximations.dynamics import LorenzAttractor
from certified_neural_approximations.train_compress_nn import train_compression


def main():
    dynamics_model = LorenzAttractor()

    # Train large model and compressed model
    train_compression(dynamics_model)

    # Verify the compressed model relative to the large model
    verify_compression(dynamics_model)


if __name__ == "__main__":
    main()
