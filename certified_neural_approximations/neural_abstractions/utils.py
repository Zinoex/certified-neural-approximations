# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import signal
from decimal import Decimal
import traceback
from typing import List
from math import log, exp, isnan
import timeit
import functools

import torch
import pandas as pd

BASEDIR = os.path.dirname(os.path.abspath(__file__))


def vprint(m, v: bool):
    """Prints first arg if second arg is True"""
    if v:
        print(m)


class Timeout:
    """Class to handle running functions with a timeout

    from https://stackoverflow.com/a/22348885.

    Requires UNIX
    """

    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def save_net_dict(filename: str, net):
    """Saves a network's state_dict to a file"""
    dir = os.path.join(BASEDIR, "results/nets/")
    filename = os.path.join(dir, filename + "_net.pth")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(net.state_dict(), filename)


def check_timeout_loc() -> bool:
    """return True if timeout occurs while in verification, else False"""
    trace = traceback.format_exc()
    return "self.verifier.verify" in trace


def log_interpolate(x, x1, x2, y1, y2):
    """Perform interpolation on a logarithmic scale.

    Interpolate the y for a corresponding x given two data points
    (x1, y1) and (x2, y2)"""
    return exp(log(y1) + log(x / x1) * log(y2 / y1) / log(x2 / x1))


def lin_interpolate(x, x1, x2, y1, y2):
    """Perform interpolation on a linear scale

    Interpolate the y for a corresponding x given two data points
    (x1, y1) and (x2, y2)"""
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


def interpolate_error(
    benchmark, partitions, results="results/hybridisation-results.csv"
):
    """Interpolate (log) the error for a given benchmark and number of partitions"""
    if benchmark == "log" or benchmark == "logsin":
        hybridisation_method = "RA"
    else:
        hybridisation_method = "ASM"
    base_results = pd.read_csv(results)
    base_results = base_results[
        (base_results["Benchmark"] == benchmark)
        & (base_results["Method"] == hybridisation_method)
    ]
    x1, x2 = (
        base_results[base_results["Partitions"] > partitions].Partitions.min(),
        base_results[base_results["Partitions"] < partitions].Partitions.max(),
    )
    y1, y2 = (
        base_results[base_results["Partitions"] > partitions].Error_1norm.max(),
        base_results[base_results["Partitions"] < partitions].Error_1norm.min(),
    )
    e = round(log_interpolate(partitions, x1, x2, y1, y2), 9)
    if isnan(e):
        e = base_results.Error_1norm.max()
    return e


def get_partitions(benchmark, width: List[int], scalar: bool):
    """Return max number of partitions for a given network width"""
    return 2 ** sum(width)


def timer(t):
    """Times the execution of a function"""
    assert isinstance(t, Timer)

    def dec(f):
        @functools.wraps(f)
        def wrapper(*a, **kw):
            t.start()
            x = f(*a, **kw)
            t.stop()
            return x

        return wrapper

    return dec


class Timer:
    """Class to assist with timing the execution of a function

    Stores min, max mean and total run times of repeat calls of a function"""

    def __init__(self):
        self.min = self.max = self.n_updates = self._sum = self._start = 0
        self.reset()

    def reset(self):
        """Resets the timer"""
        """min diff, in seconds"""
        self.min = 2**63  # arbitrary
        """max diff, in seconds"""
        self.max = 0
        """number of times the timer has been stopped"""
        self.n_updates = 0

        self._sum = 0
        self._start = 0
        self._lastest = 0

    def start(self):
        """Starts the timer"""
        self._start = timeit.default_timer()

    def stop(self):
        """Stops the timer"""
        now = timeit.default_timer()
        diff = now - self._start
        assert now >= self._start > 0
        self._start = 0
        self.n_updates += 1
        self._lastest = diff
        self._sum += diff
        self.min = min(self.min, diff)
        self.max = max(self.max, diff)

    @property
    def avg(self):

        if self.n_updates == 0:
            return 0
        assert self._sum > 0
        return self._sum / self.n_updates

    @property
    def sum(self):
        return self._sum

    def __repr__(self):
        return "total={}s,min={}s,max={}s,avg={}s,latest={}s".format(
            self._sum, self.min, self.max, self.avg, self._lastest
        )


def decimal_from_fraction(frac):
    """Converts fraction to decimal"""
    return frac.numerator / Decimal(frac.denominator)


if __name__ == "__main__":
    import benchmarks

    b = benchmarks.read_benchmark("lv")
    partitions = 2**14
    # e = interpolate_error(b, partitions)
    # print(e)  # 0.0019
