# Certified Neural Approximations

This repository contains the code and data underlying the publication "Certified Neural Approximations of Nonlinear Dynamics" by anonymous authors. The idea is to provide a framework for verifying that a neural network approximates a nonlinear dynamical system with a certain degree of accuracy. 

The code is based on Marabou 2.0 and certified first-order Taylor expansions. 

## Prerequisites
- Docker

## Reproducibility
We provide this respository as a reproducibility package for the publication. For reproducibility, we set an explicit seed with `torch.manual_seed` for each benchmark, i.e. to ensure same initialization of weights for every run. In addition, we provide .pth and .onnx files for each benchmark; as an added benefit, it also enables one to run the verification that is the focus of the paper without having to train the networks.

We provide a Dockerfile to run all three sets of benchmarks. To access the Docker image, first build a Docker image with the following command:
```bash
docker build -t ubuntu:cna .
```

Then start a container with:
```bash
docker run --name cna --rm -v $(pwd)/:/certified-neural-approximations/ -it ubuntu:cna
```

Inside the container, the initial directory is `/certified-neural-approximations`, which contains the contents of this repository.

### Dynamics approximation

To run the dynamics approximation benchmarks where we compare with state-of-the-art verification based on dReal, execute the following sequence of commands:
1. `cd experiments` (to navigate to the `experiments`).
2. `python3 dynamics_approximation.py`

Executing the above will perform the verification with both our proposed approach and with dReal. If you want to train the networks from scratch before verifying them, modify `main` in `dynamics_approximation.py` to read `train=True` (can be done in command line via `nano`).

First, the script will run our approach with the excepted output for each system:
```
Number of counterexamples found: 0
Certified percentage: 100.0000%, uncertified percentage: 0.0000%, computation time: <computation time> seconds
Verification completed for <benchmark name> system
```

Then, it will run dReal with the expected output for each system:
```
Verifying model <benchmark name> with DReal
Benchmark: Water-tank
finished... let's get the result...
Result: True, []
Verifier Timers: <computation time>
Verification completed for <benchmark name> system
```

### Compression

To run the compression benchmark where a larger network is trained from trajectories to well-approximate a dynamical system and then a smaller network is trained to approximate the larger to a high degree of accuracy, execute the following sequence of commands:
1. `cd experiments` if not already in the folder.
2. `python3 compression.py`

Executing the above will perform the verification with both our proposed approach and with dReal. If you want to train the networks from scratch before verifying them, modify `main` in `compression.py` to read `train=True` (can be done in command line via `nano`).


First, the script will run our approach with the excepted output:
```
Number of counterexamples found: 0
Certified percentage: 100.0000%, uncertified percentage: 0.0000%, computation time: <computation time> seconds
```


### Koopman auto-encoder verification

To run the Koopman auto-encoder verification benchmark where an auto-encoder is trained as a Koopman embedding encoder of a dynamical system, execute the following sequence of commands:
1. `cd experiments` if not already in the folder.
2. `python3 koopman_verification.py`

Executing the above will perform the verification with both our proposed approach and with dReal. If you want to train the networks from scratch before verifying them, modify `main` in `koopman_verification.py` to read `train=True` (can be done in command line via `nano`).


First, the script will run our approach with the excepted output:
```
Number of counterexamples found: 0
Certified percentage: 100.0000%, uncertified percentage: 0.0000%, computation time: <computation time> seconds
Verification completed for QuadraticSystem system
```