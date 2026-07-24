#!/usr/bin/env python3
"""
Run all ablation experiments and collect results.
"""

import subprocess
import re
import time
import json
from datetime import datetime

# Define all experiments
experiments = {
    '1_softplus': {
        'file': 'vae_ablation_1_softplus.py',
        'name': 'Exp 1: Softplus',
        'improvements': ['Softplus']
    },
    '2_clamp': {
        'file': 'vae_ablation_2_clamp.py',
        'name': 'Exp 2: Clamp',
        'improvements': ['Clamp']
    },
    '3_relu': {
        'file': 'vae_ablation_3_relu.py',
        'name': 'Exp 3: ReLU',
        'improvements': ['ReLU']
    },
    '4_softplus_clamp': {
        'file': 'vae_ablation_4_softplus_clamp.py',
        'name': 'Exp 4: Softplus + Clamp',
        'improvements': ['Softplus', 'Clamp']
    },
    '5_softplus_relu': {
        'file': 'vae_ablation_5_softplus_relu.py',
        'name': 'Exp 5: Softplus + ReLU',
        'improvements': ['Softplus', 'ReLU']
    },
    '6_clamp_relu': {
        'file': 'vae_ablation_6_clamp_relu.py',
        'name': 'Exp 6: Clamp + ReLU',
        'improvements': ['Clamp', 'ReLU']
    }
}

def extract_results(output_text):
    """Extract key metrics from experiment output."""
    results = {
        'mean_predicted': None,
        'mean_true': None,
        'std_predicted': None,
        'std_true': None,
        'mean_error': None,
        'std_error': None,
        'success': False
    }

    # Extract predicted and true values
    mean_match = re.search(r'Flow mean:\s+([\d.]+)\s+mm/day.*?true:\s+([\d.]+)', output_text)
    if mean_match:
        results['mean_predicted'] = float(mean_match.group(1))
        results['mean_true'] = float(mean_match.group(2))

    std_match = re.search(r'Flow std:\s+([\d.]+)\s+mm/day.*?true:\s+([\d.]+)', output_text)
    if std_match:
        results['std_predicted'] = float(std_match.group(1))
        results['std_true'] = float(std_match.group(2))

    # Extract error percentages
    mean_error_match = re.search(r'Mean error:\s+([\d.]+)%', output_text)
    if mean_error_match:
        results['mean_error'] = float(mean_error_match.group(1))

    std_error_match = re.search(r'Std error:\s+([\d.]+)%', output_text)
    if std_error_match:
        results['std_error'] = float(std_error_match.group(1))

    # Check if successful
    if all([results['mean_error'] is not None, results['std_error'] is not None]):
        results['success'] = True

    return results

def run_experiment(exp_key, exp_info):
    """Run a single experiment."""
    print(f"\nRunning {exp_info['name']}")
    print(f"File: {exp_info['file']}")
    print(f"Improvements: {', '.join(exp_info['improvements'])}")

    start_time = time.time()

    try:
        result = subprocess.run(
            ['python', exp_info['file']],
            capture_output=True,
            text=True,
            timeout=1800  # 30 min timeout
        )

        elapsed_time = time.time() - start_time

        # Save full output
        log_file = f"results_{exp_key}.log"
        with open(log_file, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\n=== STDERR ===\n")
                f.write(result.stderr)

        print(f"  Experiment completed in {elapsed_time:.1f}s")
        print(f"  Log saved to: {log_file}")

        # Extract results
        results = extract_results(result.stdout)
        results['elapsed_time'] = elapsed_time
        results['log_file'] = log_file

        if results['success']:
            print(f"  Mean error: {results['mean_error']:.2f}%")
            print(f"  Std error: {results['std_error']:.2f}%")
        else:
            print(f"  Warning: could not extract results, check log file")

        return results

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"  Experiment timed out ({elapsed_time:.1f}s)")
        return {'success': False, 'error': 'timeout', 'elapsed_time': elapsed_time}

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"  Experiment failed: {str(e)}")
        return {'success': False, 'error': str(e), 'elapsed_time': elapsed_time}

def print_summary(all_results):
    """Print results summary table."""
    print("\n\nAblation Study Results Summary\n")

    # Header
    header = f"{'Experiment':<25} {'Improvements':<25} {'Mean Err(%)':<12} {'Std Err(%)':<15} {'Time(s)':<10}"
    print(header)
    print("-" * 80)

    # Data rows
    for exp_key, results in all_results.items():
        exp_info = experiments[exp_key]
        improvements = ' + '.join(exp_info['improvements'])

        if results['success']:
            mean_err = f"{results['mean_error']:.2f}"
            std_err = f"{results['std_error']:.2f}"
            time_str = f"{results['elapsed_time']:.1f}"
        else:
            mean_err = "FAILED"
            std_err = "FAILED"
            time_str = f"{results.get('elapsed_time', 0):.1f}"

        row = f"{exp_info['name']:<25} {improvements:<25} {mean_err:<12} {std_err:<15} {time_str:<10}"
        print(row)

    print("-" * 80)

    # Find best results
    successful_results = {k: v for k, v in all_results.items() if v['success']}

    if successful_results:
        print("\nBest results:")

        best_mean = min(successful_results.items(), key=lambda x: x[1]['mean_error'])
        print(f"  Lowest mean error: {experiments[best_mean[0]]['name']} ({best_mean[1]['mean_error']:.2f}%)")

        best_std = min(successful_results.items(), key=lambda x: x[1]['std_error'])
        print(f"  Lowest std error: {experiments[best_std[0]]['name']} ({best_std[1]['std_error']:.2f}%)")

        # Compute combined score (mean error + std error)
        for k, v in successful_results.items():
            v['total_error'] = v['mean_error'] + v['std_error']

        best_overall = min(successful_results.items(), key=lambda x: x[1]['total_error'])
        print(f"  Lowest total error: {experiments[best_overall[0]]['name']} ({best_overall[1]['total_error']:.2f}%)")

    print()

def save_results(all_results):
    """Save results to JSON file."""
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiments': {}
    }

    for exp_key, results in all_results.items():
        output['experiments'][exp_key] = {
            'name': experiments[exp_key]['name'],
            'improvements': experiments[exp_key]['improvements'],
            'results': results
        }

    filename = f"ablation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Detailed results saved to: {filename}")

def main():
    """Main function."""
    print("VAE Basin Flow Model - Ablation Study Batch Runner")
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of experiments: {len(experiments)}")
    print(f"Estimated total time: ~60-120 minutes\n")

    # Ask whether to continue
    response = input("Start running all experiments? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled")
        return

    # Run all experiments
    all_results = {}
    total_start_time = time.time()

    for exp_key, exp_info in experiments.items():
        results = run_experiment(exp_key, exp_info)
        all_results[exp_key] = results

    total_elapsed = time.time() - total_start_time

    # Print summary
    print_summary(all_results)

    # Save results
    save_results(all_results)

    print(f"All done! Total time: {total_elapsed/60:.1f} minutes")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

if __name__ == "__main__":
    main()
