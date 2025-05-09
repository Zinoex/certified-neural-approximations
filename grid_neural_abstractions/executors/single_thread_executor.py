import types
from tqdm import tqdm  # Added tqdm for progress tracking
from queue import LifoQueue
import os
import time
import datetime
import json
import pickle
import psutil

class SinglethreadExecutor:
    def __init__(self, log_interval=60, checkpoint_interval=3600, log_dir=None):
        self.log_interval = log_interval  # Log progress (default every minute)
        self.checkpoint_interval = checkpoint_interval  # Save checkpoint (default every hour)
        
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

    def _write_progress_log(self, start_time, certified_domain_size, uncertified_domain_size, 
                            total_domain_size, remaining_samples):
        """Write execution progress to log file"""
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        certified_percentage = (certified_domain_size / total_domain_size) * 100
        uncertified_percentage = (uncertified_domain_size / total_domain_size) * 100
        
        log_file = os.path.join(self.log_dir, "verification_progress.log")
        with open(log_file, 'a') as f:
            timestamp = datetime.datetime.now().strftime("%d %H:%M")
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
        
        # Create a checkpoint filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = os.path.join(self.log_dir, f"verification_checkpoint_{timestamp}.pkl")
        
        checkpoint_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'verified_samples': self.new_verified_samples,
            'computation_time': computation_time
        }
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        # Also save a human-readable summary
        summary_file = os.path.join(self.log_dir, "checkpoint_summary.json")
        summary_data = {
            'timestamp': checkpoint_data['timestamp'],
            'new_samples_in_checkpoint': len(self.new_verified_samples),
            'current_checkpoint_file': checkpoint_file,
            'computation_time': computation_time
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
            
        # Clear the new verified samples after saving
        self.new_verified_samples = []

    def execute(self, initializer, process_sample, aggregate, samples, plotter=None):
        agg = None
        initializer()

        # Calculate the total domain size
        total_domain_size = sum(sample.lebesguemeasure() for sample in samples)
        certified_domain_size = 0.0
        uncertified_domain_size = 0.0

        computation_time = 0.0
        start_time = time.time()
        last_log_time = start_time
        last_checkpoint_time = start_time

        queue = LifoQueue()
        for sample in samples:
            queue.put(sample)

        with tqdm(desc="Overall Progress", smoothing=0.1) as pbar:
            while not queue.empty():
                sample = queue.get()

                # Execute the batches
                result = process_sample(sample)

                computation_time += result.computation_time
                
                if result.issat():
                    # Sample was successfully verified, no new samples to process
                    certified_domain_size += result.lebesguemeasure()
                    self.new_verified_samples.append(result)  # Track for checkpointing
                    if plotter is not None:
                        plotter.update_figure(result)
                
                if result.isunsat():
                    uncertified_domain_size += result.lebesguemeasure()
                    if plotter is not None:
                        plotter.update_figure(result)

                agg = aggregate(agg, result)

                if result.hasnewsamples():
                    for new_sample in result.newsamples():
                        queue.put(new_sample)
                
                pbar.update(1)
                certified_percentage = (certified_domain_size / total_domain_size) * 100
                uncertified_percentage = (uncertified_domain_size / total_domain_size) * 100
                
                pbar.set_description_str(
                    f"Overall Progress (remaining samples: {queue.qsize()}, certified: {certified_percentage:.4f}%, uncertified: {uncertified_percentage:.4f}%)"
                )

                # Periodically write progress to log file
                current_time = time.time()
                if current_time - last_log_time >= self.log_interval:
                    self._write_progress_log(
                        start_time, certified_domain_size, uncertified_domain_size, 
                        total_domain_size, queue.qsize()
                    )
                    last_log_time = current_time
                
                # Periodically save checkpoint with verified samples
                if current_time - last_checkpoint_time >= self.checkpoint_interval:
                    self._save_checkpoint(computation_time)
                    last_checkpoint_time = current_time

            # Final checkpoint and log at the end of execution
            self._write_progress_log(
                start_time, certified_domain_size, uncertified_domain_size, 
                total_domain_size, queue.qsize()
            )
            self._save_checkpoint(computation_time)

        return agg, certified_percentage, uncertified_percentage, computation_time
