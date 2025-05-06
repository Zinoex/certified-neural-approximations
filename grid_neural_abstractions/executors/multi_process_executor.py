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
from grid_neural_abstractions.certification_results import CertificationRegion

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


class MultiprocessExecutor:

    def __init__(self, num_workers=None, log_interval=60, checkpoint_interval=3600, log_dir=None, 
                 resume_from_checkpoint=None):
        # If num_workers is not provided, use the default of ProcessPoolExecutor os.process_cpu_count()
        self.num_workers = num_workers
        self.log_interval = log_interval  # Log progress (default every minute)
        self.checkpoint_interval = checkpoint_interval  # Save checkpoint (default every hour)
        self.resume_from_checkpoint = resume_from_checkpoint
        self.verified_domains = []  # Store previously verified domains when resuming
        
        # Set up log directory
        if log_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = os.path.join(os.getcwd(), f"logs/verification_logs_{timestamp}")
        else:
            self.log_dir = log_dir
            
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        # Track verified samples since last checkpoint
        self.new_verified_samples = []
        
        # Load checkpoint if resuming
        if resume_from_checkpoint:
            self._load_checkpoints(resume_from_checkpoint)
            
    def _load_checkpoints(self, checkpoint_path):
        """Load verified samples from checkpoint files"""
        loaded_results = []
        
        # If checkpoint_path is a directory, load all checkpoint files
        if os.path.isdir(checkpoint_path):
            checkpoint_files = glob.glob(os.path.join(checkpoint_path, "verification_checkpoint_*.pkl"))
            print(f"Found {len(checkpoint_files)} checkpoint files in directory {checkpoint_path}")
            
            for checkpoint_file in checkpoint_files:
                try:
                    with open(checkpoint_file, 'rb') as f:
                        checkpoint_data = pickle.load(f)
                        loaded_results.extend(checkpoint_data.get('verified_samples', []))
                    print(f"Loaded {len(checkpoint_data.get('verified_samples', []))} results from {checkpoint_file}")
                except (pickle.PickleError, EOFError, AttributeError) as e:
                    print(f"Warning: Error loading checkpoint file {checkpoint_file}: {e}")
        
        # If checkpoint_path is a file, load just that file
        elif os.path.isfile(checkpoint_path):
            try:
                with open(checkpoint_path, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    loaded_results.extend(checkpoint_data.get('verified_samples', []))
                print(f"Loaded {len(checkpoint_data.get('verified_samples', []))} results from {checkpoint_path}")
            except (pickle.PickleError, EOFError, AttributeError) as e:
                print(f"Warning: Error loading checkpoint file {checkpoint_path}: {e}")
        
        # Process and merge the loaded results
        if loaded_results:
            print(f"Total of {len(loaded_results)} results loaded from checkpoints")
            self.verified_domains = loaded_results  # Store the results directly, no merging for now
            print(f"Loaded {len(self.verified_domains)} verified domains")
            
    def _is_already_verified(self, sample):
        """Check if a sample is contained within an already verified domain"""
        # For SampleResult objects, we need to check the sample inside the result
        for verified_result in self.verified_domains:
            # Each verified_result is a SampleResult, and we need to access its sample
            if self._is_contained_in(sample, verified_result.sample):
                return True
        return False
    
    def _is_contained_in(self, test_sample, verified_sample):
        """Check if test_sample is fully contained within verified_sample"""
        # Access center and radius from CertificationRegion objects
        test_center, test_radius, _ = test_sample
        verified_center, verified_radius, _ = verified_sample
        
        # For each dimension, check if test_sample's bounds are within verified_sample's bounds
        for i in range(len(test_center)):
            # Calculate bounds for both samples
            test_lower = test_center[i] - test_radius[i]
            test_upper = test_center[i] + test_radius[i]
            verified_lower = verified_center[i] - verified_radius[i]
            verified_upper = verified_center[i] + verified_radius[i]
            
            # Check if test sample is outside the verified sample in this dimension
            if test_lower < verified_lower or test_upper > verified_upper:
                return False
                
        return True

    def _try_merge_samples(self):
        """
        Try to merge as many verified samples as possible to reduce storage and improve efficiency.
        This method attempts to merge samples that are adjacent or overlapping.
        """
        if not self.new_verified_samples:
            return []
            
        # Start with the first sample and try to merge others into it
        merged_samples = []
        remaining = list(self.new_verified_samples)
        
        while remaining:
            base_sample = remaining.pop(0)
            merged = False
            
            # Try to merge with existing merged samples first
            for i, existing in enumerate(merged_samples):
                if self._can_merge_samples(base_sample, existing):
                    merged_samples[i] = self._merge_samples(base_sample, existing)
                    merged = True
                    break
            
            if not merged:
                # Try to merge with remaining samples
                i = 0
                while i < len(remaining):
                    if self._can_merge_samples(base_sample, remaining[i]):
                        base_sample = self._merge_samples(base_sample, remaining.pop(i))
                        # Stay at same index as we removed an element
                    else:
                        i += 1
                
                # Add the potentially merged base_sample
                merged_samples.append(base_sample)
        
        #print(f"Merged {len(self.new_verified_samples)} samples into {len(merged_samples)} samples")
        return merged_samples
    
    def _can_merge_samples(self, sample1, sample2, precision=1e-14):
        """
        Check if two samples can be merged based on adjacency or overlap.
        Returns True if samples can be merged, False otherwise.
        """
        # Extract centers and radii
        center1, radius1, j1 = sample1.sample if hasattr(sample1, 'sample') else sample1
        center2, radius2, j2 = sample2.sample if hasattr(sample2, 'sample') else sample2
        
        if j2 != j1:
            return False  # Different output dimensions cannot be merged

        # Check if samples are adjacent or overlapping in all dimensions
        matching_dim = []
        for i in range(len(center1)):
            if center1[i] == center2[i]:
                matching_dim.append(i)
            else:
                # Calculate bounds
                min1 = center1[i] - radius1[i]
                max1 = center1[i] + radius1[i]
                min2 = center2[i] - radius2[i]
                max2 = center2[i] + radius2[i]
            
                # Check if there's no overlap or adjacency in this dimension
                if min1 > max2 + precision or min2 > max1 + precision:  # Small epsilon to handle floating point errors
                    return False
        can_merge = len(matching_dim) == len(center1) -1
        return can_merge
    
    def _merge_samples(self, sample1, sample2):
        """
        Merge two samples into a new sample that encompasses both.
        Returns a new sample with the merged properties.
        """
        # Extract centers and radii
        center1, radius1, j1 = sample1.sample if hasattr(sample1, 'sample') else sample1
        center2, radius2, j2 = sample2.sample if hasattr(sample2, 'sample') else sample2
        
        assert j2 == j1, "Cannot merge samples with different output dimensions"
        
        # Calculate new bounds
        new_min = []
        new_max = []
        for i in range(len(center1)):
            min1 = center1[i] - radius1[i]
            max1 = center1[i] + radius1[i]
            min2 = center2[i] - radius2[i]
            max2 = center2[i] + radius2[i]
            
            new_min.append(min(min1, min2))
            new_max.append(max(max1, max2))
        
        # Calculate new center and radius
        new_center = []
        new_radius = []
        for i in range(len(center1)):
            new_center.append((new_min[i] + new_max[i]) / 2)
            new_radius.append((new_max[i] - new_min[i]) / 2)
        
        # Use properties from the first sample (or we could merge them if needed)
        j = j1
        
        merged_sample = CertificationRegion(np.array(new_center), np.array(new_radius), j)
        computation_time = sample1.computation_time + sample2.computation_time
        result = type(sample1)(merged_sample, computation_time)  # Create same type of result
        return result
            
    def _write_progress_log(self, start_time, certified_domain_size, uncertified_domain_size, 
                           total_domain_size, remaining_samples):
        """Write execution progress to log file"""
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        certified_percentage = (certified_domain_size / total_domain_size) * 100
        uncertified_percentage = (uncertified_domain_size / total_domain_size) * 100
        
        log_file = os.path.join(self.log_dir, "verification_progress.log")
        with open(log_file, 'a') as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = (f"[{timestamp}] Elapsed: {elapsed_time:.2f}s | "
                        f"Remaining samples: {remaining_samples} | "
                        f"Certified: {certified_percentage:.4f}% | "
                        f"Uncertified: {uncertified_percentage:.4f}% | "
                        f"CPU Usage: {psutil.cpu_percent():.4f}% | "
                        f"RAM Usage: {psutil.virtual_memory().percent:.4f}%\n")
            f.write(log_entry)
            
    def _save_checkpoint(self, computation_time):
        """Save verified samples to checkpoint file for potential resumption"""
        if not self.new_verified_samples:
            # Skip if no new samples to save
            return
        
        # Try to merge samples before saving
        merged_samples = self._try_merge_samples()
            
        # Create a checkpoint filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = os.path.join(self.log_dir, f"verification_checkpoint_{timestamp}.pkl")
        
        checkpoint_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'verified_samples': merged_samples,  # Use the merged samples
            'computation_time': computation_time
        }
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        # Also save a human-readable summary
        summary_file = os.path.join(self.log_dir, "checkpoint_summary.json")
        summary_data = {
            'timestamp': checkpoint_data['timestamp'],
            'new_samples_in_checkpoint': len(merged_samples),  # Update count to reflect merged samples
            'original_sample_count': len(self.new_verified_samples),
            'current_checkpoint_file': checkpoint_file,
            'computation_time': computation_time
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
            
        # Clear the new verified samples after saving
        self.new_verified_samples = []

    def execute(
        self,
        initializer, process_sample, aggregate, samples, plotter=None
    ):
        agg = None
        
        # Calculate the total domain size
        total_domain_size = sum(sample.lebesguemeasure() for sample in samples)
        certified_domain_size = 0.0
        uncertified_domain_size = 0.0

        computation_time = 0.0
        start_time = time.time()
        last_log_time = start_time
        last_checkpoint_time = start_time

        # Reset verified samples at the beginning of execution
        self.new_verified_samples = []

        # First, check if any initial samples are already verified (if resuming)
        if self.resume_from_checkpoint and self.verified_domains:
            print("Checking initial samples against verified domains...")
            already_verified = []
            to_process = []
            
            for sample in samples:
                if self._is_already_verified(sample):
                    certified_domain_size += sample.lebesguemeasure()
                    already_verified.append(sample)
                else:
                    to_process.append(sample)
            
            print(f"Found {len(already_verified)} already verified samples out of {len(samples)}")
            samples = to_process
        
        with ProcessPoolExecutor(max_workers=self.num_workers, initializer=initializer) as executor:
            executor._work_ids = LifoQueue()

            with tqdm(desc="Overall Progress", smoothing=0.1) as pbar:
                futures = []
                for sample in samples:
                    future = executor.submit(process_sample, sample)
                    future.add_done_callback(lambda p: pbar.update())
                    futures.append(future)

                waiter = ExpandableAsCompleted(futures)

                for future in waiter.as_completed():
                    result = future.result()

                    computation_time += result.computation_time
                
                    if result.issat():
                        # Sample was succesfully verified, no new samples to process
                        # Update certified domain size in a thread-safe manner
                        certified_domain_size += result.lebesguemeasure()
                        # Add to new verified samples list for checkpointing
                        self.new_verified_samples.append(result)
                        # Update visualization if plotter is provided
                        if plotter is not None:
                            plotter.update_figure(result)
                    
                    if result.isunsat():
                        # Sample was not verified, add to the uncertified domain size
                        uncertified_domain_size += result.lebesguemeasure()
                        # Update visualization if plotter is provided
                        if plotter is not None:
                            plotter.update_figure(result)

                    agg = aggregate(agg, result)

                    if result.hasnewsamples():
                        # Put the new samples into the queue
                        new_samples = result.newsamples()
                        
                        # Check if any new samples are already verified
                        if self.verified_domains:
                            filtered_new_samples = []
                            for new_sample in new_samples:
                                if self._is_already_verified(new_sample):
                                    # Skip processing, count as certified
                                    certified_domain_size += new_sample.lebesguemeasure()
                                else:
                                    filtered_new_samples.append(new_sample)
                            new_samples = filtered_new_samples
                        
                        # Submit new samples to the executor
                        for new_sample in new_samples:
                            new_future = executor.submit(process_sample, new_sample)
                            new_future.add_done_callback(lambda p: pbar.update())
                            waiter.add(new_future)
                
                    certified_percentage = (certified_domain_size / total_domain_size) * 100
                    uncertified_percentage = (uncertified_domain_size / total_domain_size) * 100
                    
                    pbar.set_description_str(
                        f"Overall Progress (remaining samples: {len(waiter)}, certified: {certified_percentage:.4f}%, uncertified: {uncertified_percentage:.4f}%)"
                    )
                    
                    # Periodically write progress to log file
                    current_time = time.time()
                    if current_time - last_log_time >= self.log_interval:
                        self._write_progress_log(
                            start_time, certified_domain_size, uncertified_domain_size, 
                            total_domain_size, len(waiter)
                        )
                        last_log_time = current_time
                    
                    # Periodically save checkpoint with verified samples
                    if current_time - last_checkpoint_time >= self.checkpoint_interval:
                        self._save_checkpoint(computation_time)
                        last_checkpoint_time = current_time
                
                # Final checkpoint and log at the end of execution
                self._write_progress_log(
                    start_time, certified_domain_size, uncertified_domain_size, 
                    total_domain_size, len(waiter)
                )
                self._save_checkpoint(computation_time)
                
        return agg, certified_percentage, uncertified_percentage, computation_time
