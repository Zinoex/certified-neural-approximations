from concurrent.futures import ProcessPoolExecutor
from queue import LifoQueue
import threading
import time
import os
import json
import datetime
import pickle
import glob
import psutil
import numpy as np
from certified_neural_approximations.certification_results import CertificationRegion

from tqdm import tqdm

from certified_neural_approximations.executors.stats import Statistics  # Added tqdm for progress tracking

# Possible future states (for internal use by the futures package).
PENDING = 'PENDING'
RUNNING = 'RUNNING'
# The future was cancelled by the user...
CANCELLED = 'CANCELLED'
# ...and _Waiter.add_cancelled() was called by a worker.
CANCELLED_AND_NOTIFIED = 'CANCELLED_AND_NOTIFIED'
FINISHED = 'FINISHED'


class _Waiter(object):
    """Provides the event that wait() and as_completed() block on."""

    def __init__(self):
        self.event = threading.Event()
        self.finished_futures = []

    def add_result(self, future):
        self.finished_futures.append(future)

    def add_exception(self, future):
        self.finished_futures.append(future)

    def add_cancelled(self, future):
        self.finished_futures.append(future)


class _AsCompletedWaiter(_Waiter):
    """Used by as_completed()."""

    def __init__(self):
        super(_AsCompletedWaiter, self).__init__()
        self.lock = threading.Lock()

    def add_result(self, future):
        with self.lock:
            super(_AsCompletedWaiter, self).add_result(future)
            self.event.set()

    def add_exception(self, future):
        with self.lock:
            super(_AsCompletedWaiter, self).add_exception(future)
            self.event.set()

    def add_cancelled(self, future):
        with self.lock:
            super(_AsCompletedWaiter, self).add_cancelled(future)
            self.event.set()


class _AcquireFutures(object):
    """A context manager that does an ordered acquire of Future conditions."""

    def __init__(self, futures):
        self.futures = sorted(futures, key=id)

    def __enter__(self):
        for future in self.futures:
            future._condition.acquire()

    def __exit__(self, *args):
        for future in self.futures:
            future._condition.release()


class ExpandableAsCompleted:
    """
    A wrapper for as_completed that allows for adding new futures dynamically.
    """

    def __init__(self, fs, timeout=None):
        self.fs = set(fs)
        self.timeout = timeout

        self.waiter = None

    def __len__(self):
        return len(self.fs)

    def add(self, f):
        if self.waiter is None:
            raise RuntimeError("Cannot add futures before calling __iter__()")

        f._waiters.append(self.waiter)
        self.fs.add(f)

    def create_and_install_as_completed_waiter(self):
        waiter = _AsCompletedWaiter()

        for f in self.fs:
            f._waiters.append(waiter)

        self.waiter = waiter

    def yield_finished_futures(self, finished_futures):
        while finished_futures:
            f = finished_futures[-1]

            self.fs.remove(f)

            with f._condition:
                f._waiters.remove(self.waiter)

            del f
            # Careful not to keep a reference to the popped value
            yield finished_futures.pop()

    def as_completed(self):
        if self.timeout is not None:
            end_time = self.timeout + time.monotonic()

        with _AcquireFutures(self.fs):
            finished = set(
                f for f in self.fs
                if f._state in [CANCELLED_AND_NOTIFIED, FINISHED])
            self.create_and_install_as_completed_waiter()
        finished = list(finished)

        try:
            yield from self.yield_finished_futures(finished)

            while self.fs:
                if self.timeout is None:
                    wait_timeout = None
                else:
                    wait_timeout = end_time - time.monotonic()
                    if wait_timeout < 0:
                        raise TimeoutError(
                            '%d futures unfinished' % len(self.fs))

                self.waiter.event.wait(wait_timeout)

                with self.waiter.lock:
                    finished = self.waiter.finished_futures
                    self.waiter.finished_futures = []
                    self.waiter.event.clear()

                # reverse to keep finishing order
                finished.reverse()
                yield from self.yield_finished_futures(finished)

        finally:
            # Remove waiter from unfinished futures
            for f in self.fs:
                with f._condition:
                    f._waiters.remove(self.waiter)


class MultiprocessExecutor:

    def __init__(self, linearization_strategy, verification_strategy, linear_batch_size=1, num_workers=None):
        # If num_workers is not provided, use the default of ProcessPoolExecutor os.process_cpu_count()
        self.num_workers = num_workers
        self.linearization_strategy = linearization_strategy
        self.verification_strategy = verification_strategy
        self.linear_batch_size = linear_batch_size

    def execute(
        self,
        aggregate, samples, plotter=None
    ):
        agg = None
        statistics = Statistics(samples)

        computation_time = 0.0

        with ProcessPoolExecutor(max_workers=self.num_workers, initializer=self.verification_strategy.initialize_worker) as executor:
            executor._work_ids = LifoQueue()

            with tqdm(desc="Overall Progress", smoothing=0.1) as pbar:
                futures = []
                lin_batch = []

                linearization_result = self.linearization_strategy.linearize(samples)
                for new_sample in linearization_result:
                    future = executor.submit(self.verification_strategy.verify_sample, new_sample)
                    future.add_done_callback(lambda p: pbar.update())
                    futures.append(future)

                waiter = ExpandableAsCompleted(futures)
                start_time = None

                for future in waiter.as_completed():
                    result = future.result()

                    # Take earliest start time from all futures.
                    # This is to subtract the process spawn time
                    # from the computation time.
                    if start_time is None:
                        start_time = result.start_time
                    else:
                        start_time = min(start_time, result.start_time)

                    statistics.add_sample(result)

                    # If plotting is enabled, update the plot
                    if result.isleaf() and plotter is not None:
                        plotter.update_figure(result)

                    # Store results however caller wants
                    agg = aggregate(agg, result)

                    # Add new results to the queue
                    if result.hasnewsamples():
                        for new_sample in result.newsamples():
                            lin_batch.append(new_sample)

                    # If the batch is full or the queue is empty, linearize the batch
                    # and put the results back into the queue
                    if len(lin_batch) >= self.linear_batch_size or len(waiter) == 0:
                        linearization_result = self.linearization_strategy.linearize(lin_batch)
                        lin_batch = []

                        # Submit new samples to the executor
                        for new_sample in linearization_result:
                            new_future = executor.submit(self.verification_strategy.verify_sample, new_sample)
                            new_future.add_done_callback(lambda p: pbar.update())
                            waiter.add(new_future)

                    # Update the progress bar
                    pbar.set_description_str(
                        (
                            f"Overall Progress (remaining samples: {len(waiter)}, "
                            f"certified: {statistics.get_certified_percentage():.4f}%, "
                            f"uncertified: {statistics.get_uncertified_percentage():.4f}%)"
                        )
                    )

        end_time = time.time()
        computation_time = end_time - start_time

        return agg, statistics.get_certified_percentage(), statistics.get_uncertified_percentage(), computation_time
