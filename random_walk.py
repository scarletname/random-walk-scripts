#!/usr/bin/env python3
"""
Script for 1D random walk simulation
for execution on supercomputer and local machine.
"""

import numpy as np
import time
import json
import sys
from pathlib import Path
from datetime import datetime


class TeeOutput:
    """Class for simultaneous output to console and file."""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w', encoding='utf-8')
        self.start_time = time.time()
    
    def write(self, message):
        """Write message to console and file."""
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        """Flush buffers."""
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        """Close file."""
        if self.log_file:
            self.log_file.close()
    
    def get_elapsed_time(self):
        """Return elapsed time from start."""
        return time.time() - self.start_time


def simulate_random_walk(N, M, platform='local', seed=None, batch_size=None, progress_log=None):
    """
    Simulates M trajectories of 1D random walk with N steps.
    Uses batch processing to save memory on local machines.
    
    Parameters:
    ----------
    N : int
        Number of steps in each trajectory
    M : int
        Number of trajectories to simulate
    platform : str
        Platform ('local' or 'supercomputer')
    seed : int, optional
        Seed for random number generator (for reproducibility)
    batch_size : int, optional
        Batch size for processing (None = automatic selection)
    progress_log : list, optional
        List to save progress timestamps
    
    Returns:
    -----------
    dict : Dictionary with results (mean position, MSD, execution time)
    """
    print(f"Starting simulation: N={N}, M={M}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if seed is not None:
        np.random.seed(seed)
        print(f"Using seed: {seed}")
    
    if batch_size is None:
        if platform == 'local':
            batch_size = min(1000, M)
        else:
            # On supercomputer, use smaller batches for better progress feedback
            # Calculate maximum batch size based on available memory (assume 50GB available)
            # Each element is 8 bytes, so: batch_size * N * 8 < 50GB
            max_memory_gb = 50.0  # Conservative estimate for available memory
            max_batch_size = int(max_memory_gb * 1024**3 / (N * 8))
            # Use smaller batches (1M) to show progress more frequently
            batch_size = min(max_batch_size, M, 1000000)  # Max 1M trajectories per batch
    
    print(f"Batch size: {batch_size:,} trajectories")
    print(f"Expected memory usage: ~{batch_size * N * 8 / 1024**3:.3f} GB per batch")
    
    start_time = time.time()
    
    sum_positions = 0.0
    sum_squared_positions = 0.0
    sum_positions_for_std = []
    min_position = float('inf')
    max_position = float('-inf')
    
    num_batches = (M + batch_size - 1) // batch_size
    
    print(f"Processing {num_batches:,} batches...")
    print("This may take significant time on local machine.")
    
    last_progress_time = start_time
    progress_time_interval = 10.0  # Print progress every 10 seconds
    
    for batch_idx in range(num_batches):
        batch_start_time = time.time()
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, M)
        current_batch_size = end_idx - start_idx
        
        # Print batch start message
        print(f"Starting batch {batch_idx + 1}/{num_batches} (trajectories {start_idx:,} to {end_idx:,})...")
        sys.stdout.flush()
        
        progress_interval = max(1, min(num_batches // 100, 100))
        current_time_check = time.time()
        should_print_progress = (
            (batch_idx + 1) % progress_interval == 0 or 
            batch_idx == num_batches - 1 or
            (current_time_check - last_progress_time) >= progress_time_interval or
            batch_idx == 0
        )
        
        if should_print_progress:
            progress = (batch_idx + 1) / num_batches * 100
            elapsed = time.time() - start_time
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            if progress_log is not None:
                progress_log.append({
                    'batch': batch_idx + 1,
                    'total_batches': num_batches,
                    'progress_percent': progress,
                    'elapsed_seconds': elapsed,
                    'elapsed_minutes': elapsed / 60,
                    'timestamp': current_time,
                    'trajectories_processed': end_idx
                })
            
            if batch_idx > 0:
                avg_time_per_batch = elapsed / (batch_idx + 1)
                remaining_batches = num_batches - (batch_idx + 1)
                eta_seconds = avg_time_per_batch * remaining_batches
                eta_minutes = eta_seconds / 60
                trajectories_per_sec = end_idx / elapsed if elapsed > 0 else 0
                print(f"  Progress: {progress:.1f}% ({batch_idx + 1:,}/{num_batches:,} batches) | "
                      f"Time: {elapsed/60:.1f} min | ETA: {eta_minutes:.1f} min | "
                      f"Speed: {trajectories_per_sec:.0f} trajectories/sec | "
                      f"Time: {current_time}")
            else:
                print(f"  Progress: {progress:.1f}% ({batch_idx + 1:,}/{num_batches:,} batches) | "
                      f"Time: {current_time}")
            sys.stdout.flush()
            last_progress_time = current_time_check
        
        # Generate random values and compute steps
        random_values = np.random.random((current_batch_size, N))
        steps = np.where(random_values > 0.5, 1, -1)
        
        # Compute positions for this batch
        batch_positions = np.sum(steps, axis=1)
        
        sum_positions += np.sum(batch_positions)
        sum_squared_positions += np.sum(batch_positions ** 2)
        
        if len(sum_positions_for_std) < 100000:
            sample_size = min(100, len(batch_positions))
            if sample_size > 0:
                indices = np.random.choice(len(batch_positions), sample_size, replace=False)
                sum_positions_for_std.extend(batch_positions[indices])
        
        batch_min = np.min(batch_positions)
        batch_max = np.max(batch_positions)
        if batch_min < min_position:
            min_position = batch_min
        if batch_max > max_position:
            max_position = batch_max
        
        del steps, random_values, batch_positions
        
        # Print batch completion message
        batch_elapsed = time.time() - batch_start_time
        print(f"Batch {batch_idx + 1}/{num_batches} completed in {batch_elapsed:.2f} seconds")
        sys.stdout.flush()
    
    mean_position = sum_positions / M
    mean_squared_displacement = sum_squared_positions / M
    
    if sum_positions_for_std:
        std_position = np.std(sum_positions_for_std)
    else:
        std_position = np.sqrt(N)
    
    elapsed_time = time.time() - start_time
    end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    trajectories_per_second = M / elapsed_time if elapsed_time > 0 else 0
    
    final_positions_sample = np.array(sum_positions_for_std[:10000]) if sum_positions_for_std else None
    
    results = {
        'N': N,
        'M': M,
        'platform': platform,
        'seed': seed,
        'batch_size': batch_size,
        'start_time': start_time,
        'end_time': time.time(),
        'start_time_str': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
        'end_time_str': end_time_str,
        'mean_position': float(mean_position),
        'mean_squared_displacement': float(mean_squared_displacement),
        'std_position': float(std_position),
        'min_position': int(min_position),
        'max_position': int(max_position),
        'elapsed_time_seconds': elapsed_time,
        'elapsed_time_minutes': elapsed_time / 60,
        'elapsed_time_hours': elapsed_time / 3600,
        'trajectories_per_second': trajectories_per_second,
        'theoretical_mean_position': 0.0,
        'theoretical_mean_squared_displacement': float(N),
        'error_mean_position': abs(mean_position),
        'relative_error_msd': abs(mean_squared_displacement - N) / N * 100
    }
    
    print(f"\nSimulation results:")
    print(f"  End time: {end_time_str}")
    print(f"  Mean position: {mean_position:.6f} (theoretical: 0.0)")
    print(f"  Mean squared displacement: {mean_squared_displacement:.2f} (theoretical: {N})")
    print(f"  Standard deviation: {std_position:.2f}")
    print(f"  Min position: {min_position}")
    print(f"  Max position: {max_position}")
    print(f"  Execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes, {elapsed_time/3600:.2f} hours)")
    print(f"  Performance: {trajectories_per_second:.0f} trajectories/sec")
    print(f"  Relative MSD error: {results['relative_error_msd']:.4f}%")
    
    return results, final_positions_sample


def save_results(results, output_dir='results', progress_log=None):
    """
    Save results to JSON file.
    
    Parameters:
    ----------
    results : dict
        Results dictionary
    output_dir : str
        Output directory
    progress_log : list, optional
        Progress log to save
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    output_file = Path(output_dir) / f"results_N{results['N']}_M{results['M']}.json"
    
    if progress_log is not None:
        results['progress_log'] = progress_log
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    if progress_log:
        timing_file = Path(output_dir) / f"timing_N{results['N']}_M{results['M']}.json"
        timing_data = {
            'N': results['N'],
            'M': results['M'],
            'platform': results['platform'],
            'total_time_seconds': results['elapsed_time_seconds'],
            'total_time_minutes': results['elapsed_time_minutes'],
            'total_time_hours': results['elapsed_time_hours'],
            'start_time': results.get('start_time_str', ''),
            'end_time': results.get('end_time_str', ''),
            'trajectories_per_second': results.get('trajectories_per_second', 0),
            'progress_log': progress_log
        }
        with open(timing_file, 'w', encoding='utf-8') as f:
            json.dump(timing_data, f, indent=4, ensure_ascii=False)
        print(f"Timing details saved to: {timing_file}")
    
    return output_file


def save_sample_positions(final_positions, N, M, output_dir='results', sample_size=10000):
    """
    Save sample of final positions for visualization.
    
    Parameters:
    ----------
    final_positions : ndarray
        Array of final positions
    N : int
        Number of steps
    M : int
        Number of trajectories
    output_dir : str
        Output directory
    sample_size : int
        Sample size to save
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    if len(final_positions) > sample_size:
        indices = np.random.choice(len(final_positions), sample_size, replace=False)
        sample = final_positions[indices]
    else:
        sample = final_positions
    
    output_file = Path(output_dir) / f"positions_sample_N{N}_M{M}.npy"
    np.save(output_file, sample)
    
    print(f"Position sample saved to: {output_file} (sample size: {len(sample)})")
    return output_file


def main():
    """Main function."""
    N = 10**4
    M = 10**8
    platform = 'local'
    seed = None
    batch_size = None
    
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    if len(sys.argv) > 2:
        M = int(sys.argv[2])
    if len(sys.argv) > 3:
        platform = sys.argv[3]
    if len(sys.argv) > 4:
        seed = int(sys.argv[4])
    if len(sys.argv) > 5:
        batch_size = int(sys.argv[5])
    
    print("=" * 60)
    print("RANDOM WALK SIMULATION")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  N (steps): {N:,}")
    print(f"  M (trajectories): {M:,}")
    print(f"  Platform: {platform}")
    if seed is not None:
        print(f"  Seed: {seed}")
    print("=" * 60)
    
    Path('results').mkdir(exist_ok=True)
    log_filename = f"results/log_N{N}_M{M}_{platform}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    tee = TeeOutput(log_filename)
    sys.stdout = tee
    
    try:
        print(f"Logging output to file: {log_filename}")
        print("=" * 60)
        
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / 1024**3
            if batch_size is None:
                if platform == 'local':
                    actual_batch_size = min(1000, M)
                else:
                    max_memory_gb = 50.0
                    max_batch_size = int(max_memory_gb * 1024**3 / (N * 8))
                    actual_batch_size = min(max_batch_size, M, 1000000)
            else:
                actual_batch_size = batch_size
            required_memory = actual_batch_size * N * 8 / 1024**3
            print(f"Available memory: {available_memory:.2f} GB")
            print(f"Expected usage per batch: ~{required_memory:.3f} GB")
            if required_memory > available_memory * 0.8:
                print("WARNING: Possibly insufficient memory!")
                print("Script will automatically use smaller batch size.")
                if batch_size is None:
                    if platform == 'local':
                        max_batch_size = int(available_memory * 0.6 * 1024**3 / (N * 8))
                        batch_size = max(100, min(1000, max_batch_size))
                    else:
                        max_batch_size = int(available_memory * 0.6 * 1024**3 / (N * 8))
                        batch_size = min(max_batch_size, M, 1000000)
                    print(f"Automatically set batch size: {batch_size:,} trajectories")
        except ImportError:
            pass
        
        progress_log = []
        
        results, final_positions_sample = simulate_random_walk(N, M, platform, seed, batch_size, progress_log)
        
        save_results(results, progress_log=progress_log)
        
        if final_positions_sample is not None:
            save_sample_positions(final_positions_sample, N, M)
        else:
            print("Position sample unavailable (too few data)")
        
        print(f"\nPlatform: {platform}")
        print("\nSimulation completed successfully!")
        print(f"Full log saved to: {log_filename}")
        
    finally:
        sys.stdout = tee.terminal
        tee.close()
        print(f"\nLogging completed. Log saved to: {log_filename}")
    
    return results


if __name__ == "__main__":
    results = main()
