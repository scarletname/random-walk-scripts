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
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
            # On supercomputer, use optimized batch size for speed and stability
            # Each element is 8 bytes, so: batch_size * N * 8 = memory
            # For N=10000, 500K trajectories = 500000 * 10000 * 8 = 40GB (optimized for speed)
            batch_size = min(500000, M)  # 500K trajectories per batch = ~40GB for N=10000
    
    print(f"Batch size: {batch_size:,} trajectories")
    print(f"Expected memory usage: ~{batch_size * N * 8 / 1024**3:.3f} GB per batch")
    
    start_time = time.time()
    
    sum_positions = 0.0
    sum_squared_positions = 0.0
    # Use numpy array with fixed size instead of list to prevent memory growth
    max_std_sample = 10000  # Fixed sample size for std calculation
    sum_positions_for_std = np.zeros(max_std_sample, dtype=np.int32)
    std_sample_count = 0
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
        
        # On supercomputer, process in chunks to avoid memory issues
        if platform == 'supercomputer' and current_batch_size > 50000:
            chunk_size = 50000
            num_chunks = (current_batch_size + chunk_size - 1) // chunk_size
            
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, current_batch_size)
                chunk_size_actual = chunk_end - chunk_start
                
                # Generate random numbers for this chunk
                chunk_random = np.random.random((chunk_size_actual, N))
                
                # Compute positions for this chunk
                chunk_positive = np.sum(chunk_random > 0.5, axis=1, dtype=np.int32)
                chunk_positions = (chunk_positive * 2 - N).astype(np.int32)
                
                # Aggregate immediately
                sum_positions += np.sum(chunk_positions, dtype=np.float64)
                chunk_squared = np.sum(chunk_positions.astype(np.int64) ** 2, dtype=np.float64)
                sum_squared_positions += chunk_squared
                
                # Sample for std and visualization (collect more samples)
                if std_sample_count < max_std_sample:
                    remaining_slots = max_std_sample - std_sample_count
                    sample_size = min(50, len(chunk_positions), remaining_slots)
                    if sample_size > 0:
                        indices = np.random.choice(len(chunk_positions), sample_size, replace=False)
                        end_idx = std_sample_count + sample_size
                        sum_positions_for_std[std_sample_count:end_idx] = chunk_positions[indices].astype(np.int32)
                        std_sample_count = end_idx
                
                # Update min/max
                chunk_min = int(np.min(chunk_positions))
                chunk_max = int(np.max(chunk_positions))
                if chunk_min < min_position:
                    min_position = chunk_min
                if chunk_max > max_position:
                    max_position = chunk_max
                
                # Cleanup chunk immediately (no need to store - already aggregated)
                del chunk_random, chunk_positive, chunk_positions, chunk_squared
                gc.collect()
            
        else:
            # Standard processing for local or small batches
            random_values = np.random.random((current_batch_size, N))
            positive_steps = np.sum(random_values > 0.5, axis=1, dtype=np.int32)
            batch_positions = (positive_steps * 2 - N).astype(np.int32)
            del random_values, positive_steps
            
            sum_positions += np.sum(batch_positions, dtype=np.float64)
            batch_squared = np.sum(batch_positions.astype(np.int64) ** 2, dtype=np.float64)
            sum_squared_positions += batch_squared
            del batch_squared
            
            # Sample positions efficiently with fixed-size array
            if std_sample_count < max_std_sample:
                remaining_slots = max_std_sample - std_sample_count
                sample_size = min(50, len(batch_positions), remaining_slots)
                if sample_size > 0:
                    indices = np.random.choice(len(batch_positions), sample_size, replace=False)
                    end_idx = std_sample_count + sample_size
                    sum_positions_for_std[std_sample_count:end_idx] = batch_positions[indices].astype(np.int32)
                    std_sample_count = end_idx
                    del indices
            
            # Find min/max
            batch_min = int(np.min(batch_positions))
            batch_max = int(np.max(batch_positions))
            if batch_min < min_position:
                min_position = batch_min
            if batch_max > max_position:
                max_position = batch_max
            
            del batch_positions
        
        # Additional cleanup on supercomputer after aggregation
        if platform == 'supercomputer':
            import gc
            gc.collect()
        
        # Force garbage collection every batch on supercomputer, every 10 on local
        if platform == 'supercomputer':
            if (batch_idx + 1) % 10 == 0:  # Cleanup every 10 batches, not every batch
                import gc
                gc.collect()
        elif (batch_idx + 1) % 10 == 0:
            import gc
            gc.collect()
        
        # Print progress - only every 10 batches or at important milestones
        batch_elapsed = time.time() - batch_start_time
        elapsed_total = time.time() - start_time
        progress = (batch_idx + 1) / num_batches * 100
        
        # Print progress every 10 batches, or at 1%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 99%, 100%
        should_print = (
            (batch_idx + 1) % 10 == 0 or
            batch_idx == 0 or
            batch_idx == num_batches - 1 or
            progress >= 99.0 or
            (progress >= 1.0 and progress < 1.1) or
            (progress >= 5.0 and progress < 5.1) or
            (progress >= 10.0 and progress < 10.1) or
            (progress >= 25.0 and progress < 25.1) or
            (progress >= 50.0 and progress < 50.1) or
            (progress >= 75.0 and progress < 75.1) or
            (progress >= 90.0 and progress < 90.1) or
            (progress >= 95.0 and progress < 95.1)
        )
        
        if should_print:
            if batch_idx > 0:
                avg_time_per_batch = elapsed_total / (batch_idx + 1)
                remaining_batches = num_batches - (batch_idx + 1)
                eta_seconds = avg_time_per_batch * remaining_batches
                eta_minutes = eta_seconds / 60
                eta_hours = eta_minutes / 60
                trajectories_per_sec = end_idx / elapsed_total if elapsed_total > 0 else 0
                current_time = datetime.now().strftime('%H:%M:%S')
                
                if eta_hours >= 1:
                    print(f"Progress: {progress:.1f}% ({batch_idx + 1}/{num_batches} batches) | "
                          f"Time: {elapsed_total/60:.1f} min | ETA: {eta_hours:.1f} h | "
                          f"Speed: {trajectories_per_sec:.0f} traj/sec | {current_time}")
                else:
                    print(f"Progress: {progress:.1f}% ({batch_idx + 1}/{num_batches} batches) | "
                          f"Time: {elapsed_total/60:.1f} min | ETA: {eta_minutes:.1f} min | "
                          f"Speed: {trajectories_per_sec:.0f} traj/sec | {current_time}")
            else:
                print(f"Starting simulation... Progress: {progress:.1f}% ({batch_idx + 1}/{num_batches} batches)")
            sys.stdout.flush()
        
        if progress_log is not None:
            progress_log.append({
                'batch': batch_idx + 1,
                'total_batches': num_batches,
                'progress_percent': progress,
                'elapsed_seconds': elapsed_total,
                'elapsed_minutes': elapsed_total / 60,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'trajectories_processed': end_idx,
                'batch_time_seconds': batch_elapsed
            })
        
        sys.stdout.flush()
        last_progress_time = time.time()
    
    mean_position = sum_positions / M
    mean_squared_displacement = sum_squared_positions / M
    
    # Calculate std from fixed-size sample
    if std_sample_count > 0:
        std_position = float(np.std(sum_positions_for_std[:std_sample_count]))
    else:
        std_position = np.sqrt(N)
    
    elapsed_time = time.time() - start_time
    end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    trajectories_per_second = M / elapsed_time if elapsed_time > 0 else 0
    
    # Create final sample from fixed-size array
    if std_sample_count > 0:
        sample_size = min(10000, std_sample_count)
        final_positions_sample = sum_positions_for_std[:sample_size].copy()
    else:
        final_positions_sample = None
    
    # Clean up memory
    del sum_positions_for_std
    import gc
    gc.collect()
    
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


def load_results(results_dir='results'):
    """
    Load simulation results from JSON files.
    
    Parameters
    ----------
    results_dir : str
        Directory with results files.
    
    Returns
    -------
    dict
        Dictionary with results grouped by platform.
    """
    results_files = sorted(Path(results_dir).glob('results_*.json'))
    
    if not results_files:
        print(f"No result files found in directory {results_dir}")
        return {'local': None, 'supercomputer': None, 'all': []}
    
    results = {'all': []}
    for file in results_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results['all'].append(data)
            platform = data.get('platform', 'unknown')
            if platform in ['local', 'supercomputer']:
                results[platform] = data
    
    return results


def load_sample_positions(results_dir='results'):
    """
    Load position samples from numpy files.
    
    Parameters
    ----------
    results_dir : str
        Directory with results files.
    
    Returns
    -------
    dict
        Mapping from file name to numpy array with positions.
    """
    position_files = sorted(Path(results_dir).glob('positions_sample_*.npy'))
    
    positions = {}
    for file in position_files:
        positions[file.name] = np.load(file)
    
    return positions


def plot_statistics_comparison(results, output_dir='results'):
    """
    Plot comparison between calculated and theoretical statistics.
    
    Parameters
    ----------
    results : list
        List with result dictionaries.
    output_dir : str
        Directory for saving figures.
    """
    if not results:
        print("No data available for statistics plot.")
        return
    
    result = results[0]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Random Walk Statistical Characteristics', fontsize=16, fontweight='bold')
    
    # Mean comparison
    ax1 = axes[0]
    categories = [
        'Mean Position (Theory)',
        'Mean Position (Measured)',
        'MSD (Theory)',
        'MSD (Measured)'
    ]
    values = [
        0,
        result['mean_position'],
        result['theoretical_mean_squared_displacement'],
        result['mean_squared_displacement']
    ]
    colors = ['blue', 'red', 'blue', 'red']
    
    bars = ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Value')
    ax1.set_title('Theoretical vs Measured')
    ax1.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.2f}",
            ha='center',
            va='bottom',
            fontweight='bold'
        )
    
    # Error comparison
    ax2 = axes[1]
    error_types = ['Mean Position Error', 'Relative MSD Error (%)']
    error_values = [result['error_mean_position'], result['relative_error_msd']]
    bars2 = ax2.bar(error_types, error_values, color=['orange', 'green'], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Error Value')
    ax2.set_title('Calculation Error')
    ax2.grid(True, alpha=0.3)
    
    for bar, val in zip(bars2, error_values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.4f}",
            ha='center',
            va='bottom',
            fontweight='bold'
        )
    
    # Detailed info
    ax3 = axes[2]
    ax3.axis('off')
    info_text = (
        "Simulation Parameters:\n"
        "------------------------------\n"
        f"N (steps): {result['N']:,}\n"
        f"M (trajectories): {result['M']:,}\n\n"
        "Results:\n"
        "------------------------------\n"
        f"Mean Position: {result['mean_position']:.6f}\n"
        f"Theoretical Mean: {result['theoretical_mean_position']:.2f}\n\n"
        f"Mean Squared Displacement: {result['mean_squared_displacement']:.2f}\n"
        f"Theoretical MSD: {result['theoretical_mean_squared_displacement']:.2f}\n\n"
        "Statistics:\n"
        "------------------------------\n"
        f"Standard Deviation: {result['std_position']:.2f}\n"
        f"Min Position: {result['min_position']}\n"
        f"Max Position: {result['max_position']}\n\n"
        "Performance:\n"
        "------------------------------\n"
        f"Execution Time: {result['elapsed_time_seconds']:.2f} sec\n"
        f"({result['elapsed_time_minutes']:.2f} minutes)\n"
    )
    ax3.text(
        0.1,
        0.5,
        info_text,
        fontsize=11,
        family='monospace',
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    output_file = Path(output_dir) / 'statistics_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Statistics plot saved: {output_file}")
    plt.close()


def plot_position_distribution(positions_dict, output_dir='results'):
    """
    Plot distribution of final positions.
    
    Parameters
    ----------
    positions_dict : dict
        Mapping from file name to positions array.
    output_dir : str
        Directory for saving figures.
    """
    if not positions_dict:
        print("No position data available for histogram.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Final Position Distribution', fontsize=16, fontweight='bold')
    
    file_name, positions = next(iter(positions_dict.items()))
    
    match = re.search(r'N(\d+)_M', file_name)
    if match:
        N = int(match.group(1))
    else:
        N = 10 ** 4
    
    # Histogram
    ax1 = axes[0]
    n_bins = min(100, max(10, int(np.sqrt(len(positions)))))
    counts, bins, patches = ax1.hist(
        positions,
        bins=n_bins,
        edgecolor='black',
        alpha=0.7,
        color='skyblue'
    )
    ax1.set_xlabel('Final Position')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Histogram')
    ax1.grid(True, alpha=0.3)
    
    mean_pos = np.mean(positions)
    ax1.axvline(mean_pos, color='red', linestyle='--', linewidth=2, label=f"Mean: {mean_pos:.2f}")
    ax1.axvline(0, color='green', linestyle='--', linewidth=2, label='Theoretical Mean: 0')
    ax1.legend()
    
    # Normalized histogram with theoretical curve
    ax2 = axes[1]
    sigma_theoretical = np.sqrt(N)
    
    counts_norm, bins_norm, patches_norm = ax2.hist(
        positions,
        bins=n_bins,
        density=True,
        edgecolor='black',
        alpha=0.7,
        color='lightcoral',
        label='Experimental'
    )
    
    x_theoretical = np.linspace(bins_norm[0], bins_norm[-1], 1000)
    y_theoretical = (1 / (sigma_theoretical * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * (x_theoretical / sigma_theoretical) ** 2
    )
    ax2.plot(x_theoretical, y_theoretical, 'b-', linewidth=2, label='Theory (N(0, N))')
    
    ax2.set_xlabel('Final Position')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Histogram vs Theoretical Normal Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    output_file = Path(output_dir) / 'position_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved: {output_file}")
    plt.close()


def create_comparison_report(results_local=None, results_supercomputer=None, output_dir='results'):
    """
    Create comparison report between local and supercomputer executions.
    
    Parameters
    ----------
    results_local : dict or None
        Results from local execution.
    results_supercomputer : dict or None
        Results from supercomputer execution.
    output_dir : str
        Directory for saving figures.
    """
    if not results_local and not results_supercomputer:
        print("No execution results available for comparison.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Local vs Supercomputer Comparison', fontsize=16, fontweight='bold')
    
    if results_local and results_supercomputer:
        # Execution time
        ax1 = axes[0, 0]
        platforms = ['Local Machine', 'Supercomputer']
        times = [
            results_local['elapsed_time_seconds'],
            results_supercomputer['elapsed_time_seconds']
        ]
        bars = ax1.bar(platforms, times, color=['blue', 'red'], alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Execution Time')
        ax1.grid(True, alpha=0.3)
        for bar, val in zip(bars, times):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.2f} sec",
                ha='center',
                va='bottom',
                fontweight='bold'
            )
        
        speedup = results_local['elapsed_time_seconds'] / results_supercomputer['elapsed_time_seconds']
        ax1.text(
            0.5,
            max(times) * 0.9,
            f"Speedup: {speedup:.2f}x",
            ha='center',
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7)
        )
        
        # Mean positions
        ax2 = axes[0, 1]
        mean_positions = [
            results_local['mean_position'],
            results_supercomputer['mean_position']
        ]
        bars2 = ax2.bar(platforms, mean_positions, color=['blue', 'red'], alpha=0.7, edgecolor='black')
        ax2.axhline(0, color='green', linestyle='--', linewidth=2, label='Theoretical Mean: 0')
        ax2.set_ylabel('Mean Position')
        ax2.set_title('Mean Position')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # MSD
        ax3 = axes[1, 0]
        msd_values = [
            results_local['mean_squared_displacement'],
            results_supercomputer['mean_squared_displacement']
        ]
        theoretical_msd = results_local['theoretical_mean_squared_displacement']
        bars3 = ax3.bar(platforms, msd_values, color=['blue', 'red'], alpha=0.7, edgecolor='black')
        ax3.axhline(
            theoretical_msd,
            color='green',
            linestyle='--',
            linewidth=2,
            label=f"Theoretical MSD: {theoretical_msd:.0f}"
        )
        ax3.set_ylabel('Mean Squared Displacement')
        ax3.set_title('Mean Squared Displacement')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Errors
        ax4 = axes[1, 1]
        error_types = [
            'Mean Error (Local)',
            'Mean Error (Supercomputer)',
            'MSD Error (Local, %)',
            'MSD Error (Supercomputer, %)'
        ]
        error_values = [
            results_local['error_mean_position'],
            results_supercomputer['error_mean_position'],
            results_local['relative_error_msd'],
            results_supercomputer['relative_error_msd']
        ]
        bars4 = ax4.bar(
            error_types,
            error_values,
            color=['lightblue', 'lightcoral', 'lightblue', 'lightcoral'],
            alpha=0.7,
            edgecolor='black'
        )
        ax4.set_ylabel('Error Value')
        ax4.set_title('Accuracy Comparison')
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    else:
        ax1 = axes[0, 0]
        ax1.text(
            0.5,
            0.5,
            'Only local data is available.\nSubmit supercomputer results to compare.',
            ha='center',
            va='center',
            fontsize=14,
            transform=ax1.transAxes
        )
        ax1.axis('off')
        
        ax2 = axes[0, 1]
        info_text = (
            "Local Results:\n"
            "------------------------------\n"
            f"N: {results_local['N']:,}\n"
            f"M: {results_local['M']:,}\n"
            f"Time: {results_local['elapsed_time_seconds']:.2f} sec\n"
            f"Mean Position: {results_local['mean_position']:.6f}\n"
            f"MSD: {results_local['mean_squared_displacement']:.2f}\n"
        )
        ax2.text(
            0.1,
            0.5,
            info_text,
            fontsize=12,
            family='monospace',
            verticalalignment='center',
            transform=ax2.transAxes
        )
        ax2.axis('off')
        
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    output_file = Path(output_dir) / 'comparison_report.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison report saved: {output_file}")
    plt.close()


def generate_visualizations(results_dir='results'):
    """
    Generate all visualizations and comparison reports.
    
    Parameters
    ----------
    results_dir : str
        Directory that stores simulation outputs.
    """
    print("\nGenerating visualizations...")
    results_dict = load_results(results_dir)
    
    if not results_dict['all']:
        print("Visualization skipped: no result files found.")
        return
    
    plot_statistics_comparison(results_dict['all'], results_dir)
    
    positions = load_sample_positions(results_dir)
    if positions:
        plot_position_distribution(positions, results_dir)
    else:
        print("No position samples found; skipping distribution plot.")
    
    create_comparison_report(results_dict.get('local'), results_dict.get('supercomputer'), results_dir)
    print("Visualization completed.")


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
                    actual_batch_size = min(500000, M)
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
                        batch_size = min(500000, M)
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
        
        try:
            generate_visualizations('results')
        except Exception as viz_error:
            print(f"Visualization step skipped due to error: {viz_error}")
        
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
