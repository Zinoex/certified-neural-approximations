# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from functools import partial
import multiprocessing
import os
import time

import torch
from torch import nn
import numpy as np

from benchmarks import read_benchmark
from certified_neural_approximations.train_nn import SimpleNN
from verifier import DRealVerifier
from translator import Translator
from config import Config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(REPO_DIR, "data")


class VerificationWrapperNetwork(nn.Module):
    def __init__(self, net):
        super(VerificationWrapperNetwork, self).__init__()
        self.model = net.network

    def forward(self, x):
        return self.model(x)


def verify(res_queue, verify, network, translator):
    candidate = translator.translate(network)
    found, cex = verify(candidate)
    res_queue.put((found, cex))


def verify_neural_abstractions(config: Config):
    benchmark = read_benchmark(config.benchmark)

    verifier_type = DRealVerifier
    x = verifier_type.new_vars(benchmark.dimension)
    verifier = verifier_type(
        x,
        benchmark.dimension,
        benchmark.get_domain,
        verbose=config.verbose,
    )
    translator = Translator(x, verifier.relu)
    truef = np.array(benchmark.f(x)).reshape(-1, 1)

    network = SimpleNN(benchmark.dimension, config.widths, benchmark.dimension)
    path = os.path.join(DATA_DIR, config.model_path)
    network.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    network = VerificationWrapperNetwork(network)

    res_queue = multiprocessing.Queue()
    verifier_verify = partial(verifier.verify, truef, epsilon=[config.target_error for _ in range(benchmark.dimension)])

    timeout = 3600
    print("Benchmark: {}".format(benchmark.name))

    process = multiprocessing.Process(target=verify, args=(res_queue, verifier_verify, network, translator))

    t0 = time.perf_counter()
    process.start()

    # Wait for {timeout} seconds or until process finishes
    process.join(timeout)

    # If thread is still active
    if process.is_alive():
        print("running... let's kill it...")

        process.kill()
        process.join()
    else:
        print("finished... let's get the result...")
        t1 = time.perf_counter()
        delta_t = t1 - t0
        found, cex = res_queue.get()
        print("Result: {}, {}".format(found, cex))
        print("Verifier Timers: {} \n".format(delta_t))
