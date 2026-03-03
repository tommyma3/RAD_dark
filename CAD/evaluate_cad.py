"""
Evaluation script for Compressed Algorithm Distillation (CAD).

Evaluates a trained CAD model on test environments with in-context learning.

Usage:
    python evaluate_cad.py --ckpt_dir ./runs/CAD-darkroom-seed0
"""

from datetime import datetime
from glob import glob
import argparse
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import torch
import os.path as path
import numpy as np

from env import SAMPLE_ENVIRONMENT, make_env
from model import MODEL
from stable_baselines3.common.vec_env import SubprocVecEnv

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
seed = 0
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='./runs/CAD-darkroom-seed0',
                       help='Directory containing CAD checkpoint')
    parser.add_argument('--eval_episodes', type=int, default=100,
                       help='Number of episodes to evaluate')
    parser.add_argument('--use_best', action='store_true',
                       help='Use best-model.pt instead of latest checkpoint')
    args = parser.parse_args()
    
    ckpt_dir = args.ckpt_dir
    
    # Check for best model first if requested
    best_model_path = path.join(ckpt_dir, 'best-model.pt')
    if args.use_best and path.exists(best_model_path):
        ckpt_path = best_model_path
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        print(f'Best model loaded from {ckpt_path}')
        print(f'Best model was saved at step {ckpt["step"]} with reward {ckpt.get("eval_reward", "N/A")}')
        config = ckpt['config']
    else:
        ckpt_paths = sorted(glob(path.join(ckpt_dir, 'ckpt-*.pt')))
        if len(ckpt_paths) > 0:
            ckpt_path = ckpt_paths[-1]
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            print(f'Checkpoint loaded from {ckpt_path}')
            config = ckpt['config']
        else:
            raise ValueError('No checkpoint found.')
    
    config['device'] = device
    
    model_name = config['model']
    model = MODEL[model_name](config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    env_name = config['env']
    _, test_env_args = SAMPLE_ENVIRONMENT[env_name](config)

    print(f"Model: {model_name}")
    print(f"Evaluation goals: {test_env_args}")
    print(f"Max sequence length: {config['n_transit']}")
    print(f"Compression tokens: {config.get('n_compress_tokens', 'N/A')}")

    if env_name == 'darkroom':
        envs = SubprocVecEnv([make_env(config, goal=arg) for arg in test_env_args])
    elif env_name == 'dark_key_to_door':
        envs = SubprocVecEnv([make_env(config, key=arg[:2], goal=arg[2:]) for arg in test_env_args])
    else:
        raise NotImplementedError(f'Environment not supported: {env_name}')
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    start_time = datetime.now()
    print(f'Starting at {start_time}')

    with torch.no_grad():
        eval_output = model.evaluate_in_context(
            vec_env=envs, 
            eval_timesteps=config['horizon'] * args.eval_episodes
        )
        test_rewards = eval_output['reward_episode']
        
        # CAD-specific metrics
        total_compressions = eval_output.get('total_compressions', 0)
        compression_events = eval_output.get('compression_events', [])
        
        result_path = path.join(ckpt_dir, 'eval_result.npy')
    
    end_time = datetime.now()
    print()
    print(f'Ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')

    envs.close()

    with open(result_path, 'wb') as f:
        np.save(f, test_rewards)

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for i in range(len(test_env_args)):
        print(f'Env {i} (goal={test_env_args[i]}): {test_rewards[i]} , mean={test_rewards[i].mean():.3f}, std={test_rewards[i].std():.3f}')

    print("\n" + "-"*60)
    print(f"Mean reward per environment: {test_rewards.mean(axis=1)}")
    print(f"Overall mean reward: {test_rewards.mean():.3f}")
    print(f"Std deviation: {test_rewards.std():.3f}")
    
    print("\n" + "-"*60)
    print("COMPRESSION STATISTICS")
    print(f"Total compression events: {total_compressions}")
    print(f"Compressions per episode: {total_compressions / args.eval_episodes:.2f}")
    
    if len(compression_events) > 0:
        print(f"First compression at step: {compression_events[0]}")
        print(f"Average steps between compressions: {np.diff(compression_events).mean():.1f}" if len(compression_events) > 1 else "N/A")
