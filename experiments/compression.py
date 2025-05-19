

from certified_neural_approximations.verify_compression import verify_compression
from certified_neural_approximations.dynamics import LorenzAttractor
from certified_neural_approximations.train_compress_nn import train_compression


def main(train=False, verify=True):
    dynamics_model = LorenzAttractor()

    if train:
        # Train large model and compressed model
        train_compression(dynamics_model)

    if verify:
        # Verify the compressed model relative to the large model
        verify_compression(dynamics_model)


if __name__ == "__main__":
    main()
