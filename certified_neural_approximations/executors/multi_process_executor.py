from concurrent.futures import ProcessPoolExecutor
from queue import LifoQueue
import threading
import time

from tqdm import tqdm  # Added tqdm for progress tracking

from .stats import Statistics


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
    def __init__(self, num_workers=None):
        # If num_workers is not provided, use the default of ProcessPoolExecutor os.process_cpu_count()
        self.num_workers = num_workers

    def execute(self, initializer, process_sample, aggregate, samples, plotter=None):
        agg = None
        statistics = Statistics(samples)

        with ProcessPoolExecutor(max_workers=self.num_workers, initializer=initializer) as executor:
            # Use a LifoQueue to achieve DFS (Depth-First Search)-like behavior.
            # For a single-threaded executor, this is true DFS, but for a multi-threaded
            # executor, it depends on the order results are available.
            executor._work_ids = LifoQueue()

            with tqdm(desc="Overall Progress", smoothing=0.1) as pbar:
                futures = []
                for sample in samples:
                    future = executor.submit(process_sample, sample)
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

                    # Update statistics
                    statistics.add_sample(result)

                    # Update visualization if plotter is provided
                    if result.isleaf() and plotter is not None:
                        plotter.update_figure(result)

                    # Store results however caller wants
                    agg = aggregate(agg, result)

                    # Add new results to the queue
                    if result.hasnewsamples():
                        # Put the new samples into the queue
                        new_samples = result.newsamples()

                        # Submit new samples to the executor
                        for new_sample in new_samples:
                            new_future = executor.submit(process_sample, new_sample)
                            new_future.add_done_callback(lambda p: pbar.update())
                            waiter.add(new_future)

                    # Update the progress bar
                    pbar.set_description_str(
                        f"Overall Progress (remaining samples: {len(waiter)}, "
                        f"certified: {statistics.get_certified_percentage():.4f}%, "
                        f"uncertified: {statistics.get_uncertified_percentage():.4f}%)"
                    )

        end_time = time.time()
        computation_time = end_time - start_time

        return agg, statistics.get_certified_percentage(), statistics.get_uncertified_percentage(), computation_time
