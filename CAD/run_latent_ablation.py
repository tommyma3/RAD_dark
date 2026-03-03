"""
Ablation Study: Number of Latent Tokens in CAD (Compressed Algorithm Distillation)

This script runs ablation experiments varying n_compress_tokens while keeping n_transit constant.
Each experiment is saved in a separate folder to avoid conflicts.

Usage:
    python run_latent_ablation.py [--seeds 0 1 2] [--dry_run]
    
For multi-GPU:
    python run_latent_ablation.py --use_accelerate
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# Ablation configurations: (config_name, n_compress_tokens)
ABLATION_CONFIGS = [
    ('cad_dr_latent8', 8),
    ('cad_dr_latent16', 16),   # baseline
    ('cad_dr_latent24', 24),
    ('cad_dr_latent32', 32),
    ('cad_dr_latent48', 48),
]


def run_experiment(config_name, seed, use_accelerate=False, dry_run=False, exp_dir='ablation_latent'):
    """Run a single ablation experiment."""
    
    # Create unique run directory based on config and seed
    run_name = f"{config_name}-seed{seed}"
    log_dir = os.path.join('./runs', exp_dir, run_name)
    
    # Build command
    if use_accelerate:
        cmd = ['accelerate', 'launch', 'train_cad.py']
    else:
        cmd = [sys.executable, 'train_cad.py']
    
    cmd.extend([
        '--config', config_name,
        '--env', 'darkroom',
    ])
    
    print(f"\n{'='*60}")
    print(f"Running: {run_name}")
    print(f"Log dir: {log_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    if dry_run:
        print("[DRY RUN] Would execute above command")
        return True
    
    # Set environment variable for custom log dir
    env = os.environ.copy()
    env['CAD_LOG_DIR'] = log_dir
    env['CAD_SEED'] = str(seed)
    
    try:
        result = subprocess.run(cmd, env=env, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Experiment {run_name} failed with error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run latent token ablation study')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0],
                       help='Random seeds to run (default: [0])')
    parser.add_argument('--configs', type=str, nargs='+', default=None,
                       help='Specific configs to run (default: all)')
    parser.add_argument('--use_accelerate', action='store_true',
                       help='Use accelerate for multi-GPU training')
    parser.add_argument('--dry_run', action='store_true',
                       help='Print commands without executing')
    parser.add_argument('--exp_dir', type=str, default='ablation_latent',
                       help='Experiment directory under ./runs/')
    args = parser.parse_args()
    
    # Filter configs if specified
    if args.configs:
        configs = [(c, n) for c, n in ABLATION_CONFIGS if c in args.configs]
    else:
        configs = ABLATION_CONFIGS
    
    print(f"Ablation Study: Latent Tokens in CAD")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configs: {[c[0] for c in configs]}")
    print(f"Seeds: {args.seeds}")
    print(f"Total experiments: {len(configs) * len(args.seeds)}")
    
    results = []
    
    for config_name, n_latent in configs:
        for seed in args.seeds:
            success = run_experiment(
                config_name=config_name,
                seed=seed,
                use_accelerate=args.use_accelerate,
                dry_run=args.dry_run,
                exp_dir=args.exp_dir,
            )
            results.append((config_name, seed, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*60}")
    for config_name, seed, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {config_name}-seed{seed}: {status}")
    
    successful = sum(1 for _, _, s in results if s)
    print(f"\nTotal: {successful}/{len(results)} experiments completed")


if __name__ == '__main__':
    main()
