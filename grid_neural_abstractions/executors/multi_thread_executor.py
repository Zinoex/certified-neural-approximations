from concurrent.futures import ThreadPoolExecutor
import threading
import time

from tqdm import tqdm  # Added tqdm for progress tracking


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


class MultithreadExecutor:
    """
    Note: Normally, the global interpreter lock (GIL) in Python can limit the performance of multi-threadeding.
    However, in this case, since Marabou is a C++ library and the heavy lifting is done in C++, the lock is released
    during the execution of the C++ code, thus not prohibiting the performance benefits of threading.
    """

    def __init__(self, num_workers=None):
        # If num_workers is not provided, use the default of ThreadPoolExecutor min(32, os.cpu_count() + 4)
        self.num_workers = num_workers

    def execute(
        self,
        initializer, process_sample, aggregate, samples
    ):
        local = threading.local()
        agg = None

        # Calculate the total domain size
        total_domain_size = sum(sample.calculate_size() for sample in samples)
        certified_domain_size = 0

        with ThreadPoolExecutor(max_workers=self.num_workers, initializer=initializer, initargs=(local,)) as executor:
            with tqdm(desc="Overall Progress", smoothing=0.1) as pbar:
                futures = []
                for sample in samples:
                    future = executor.submit(process_sample, local, sample)
                    future.add_done_callback(lambda p: pbar.update())
                    futures.append(future)

                waiter = ExpandableAsCompleted(futures)

                for future in waiter.as_completed():
                    returned_samples, result = future.result()
                    
                    if len(returned_samples) == 1:
                        # Sample was succesfully verified, no new samples to process
                        # Update certified domain size in a thread-safe manner
                        certified_domain_size += returned_samples[0].calculate_size()
                    else:   
                        agg = aggregate(agg, result)
                        # Put the new samples into the queue
                        for new_sample in returned_samples:
                            new_future = executor.submit(process_sample, local, new_sample)
                            new_future.add_done_callback(lambda p: pbar.update())
                            waiter.add(new_future)
                    
                    certified_percentage = (certified_domain_size / total_domain_size) * 100
                    pbar.set_description_str(
                        f"Overall Progress (remaining samples: {len(waiter)}, certified: {certified_percentage:.2f}%)"
                    )
                    
        return agg
