"""
Training script for n_transit ablation study.

This script trains CAD models with different n_transit values (max sequence length)
while keeping n_compress_tokens fixed at 16.

Usage:
    python train_transit_ablation.py [--configs transit30 transit45] [--seed 0]
"""

import argparse
import os
import os.path as path
import sys
from datetime import datetime
from glob import glob
import gc

import yaml
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate import DistributedDataParallelKwargs
from tqdm import tqdm
import multiprocessing
import numpy as np
import torch.nn.functional as F
from functools import partial
from stable_baselines3.common.vec_env import SubprocVecEnv

from dataset import CompressedADDataset, ADDataset
from env import SAMPLE_ENVIRONMENT, make_env
from model import MODEL
from utils import get_config, next_dataloader, get_curriculum_aware_scheduler, ad_collate_fn


# Import curriculum and collate functions from train_cad
from train_cad import (
    get_curriculum_from_config,
    get_curriculum_max_compressions,
    get_curriculum_length_distribution,
    get_curriculum_stage,
    get_cad_data_loader,
    DEFAULT_LENGTH_DISTRIBUTIONS,
    cad_collate_fn,
)


# Ablation configurations: (suffix, n_transit)
ABLATION_CONFIGS = {
    'transit30': 30,
    'transit45': 45,
    'transit60': 60,
    'transit90': 90,
    'transit120': 120,
}


def train_single_config(config_suffix, seed, exp_dir='ablation_transit', env='darkroom'):
    """Train a single ablation configuration."""
    
    config_name = f'cad_dr_{config_suffix}'
    run_name = f'{config_name}-seed{seed}'
    log_dir = path.join('./runs', exp_dir, run_name)
    
    print(f"\n{'='*70}")
    print(f"Training: {run_name}")
    print(f"Config: {config_name}")
    print(f"n_transit: {ABLATION_CONFIGS[config_suffix]}")
    print(f"n_compress_tokens: 16 (fixed)")
    print(f"Log dir: {log_dir}")
    print(f"{'='*70}\n")
    
    # Load configs based on environment (same as train_latent_ablation.py)
    env_config_map = {
        'darkroom': ('darkroom', 'ppo_darkroom'),
        'dark_key_to_door': ('dark_key_to_door', 'ppo_dark_key_to_door'),
    }
    env_cfg, alg_cfg = env_config_map[env]
    
    config = get_config(f'./config/env/{env_cfg}.yaml')
    config.update(get_config(f'./config/algorithm/{alg_cfg}.yaml'))
    config.update(get_config(f'./config/model/{config_name}.yaml'))
    
    # Override seed
    config['seed'] = seed
    config['env_split_seed'] = seed
    
    set_seed(seed)
    
    config['log_dir'] = log_dir
    config['traj_dir'] = './datasets'
    config['mixed_precision'] = 'fp16'
    
    # Curriculum
    curriculum = get_curriculum_from_config(config)
    
    # Initialize accelerator (SAME AS train_latent_ablation.py - with gradient_accumulation_steps)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=config['mixed_precision'],
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
        kwargs_handlers=[ddp_kwargs],
    )
    
    config['device'] = accelerator.device
    is_main = accelerator.is_main_process
    
    if is_main:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir, flush_secs=15)
        
        # Save config (exclude non-serializable objects like torch.device)
        config_to_save = {k: v for k, v in config.items() if not isinstance(v, torch.device)}
        config_to_save['device'] = str(config['device'])  # Save device as string
        with open(path.join(log_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config_to_save, f)
        
        print(f"n_transit: {config['n_transit']}")
        print(f"n_compress_tokens: {config['n_compress_tokens']}")
        print(f"Available space: {config['n_transit'] - config['n_compress_tokens'] - 1}")
    
    # Create model (don't manually move to device - let accelerator handle it)
    model = MODEL[config['model']](config)
    
    # Try to load pretrained compression (SAME AS train_latent_ablation.py)
    pretrain_loaded = False
    pretrain_path = None
    
    # Find pretrain checkpoint
    pretrain_dir = path.join('./runs', f"CAD-pretrain-{config['env']}-seed{config['env_split_seed']}")
    candidate_paths = [
        path.join(pretrain_dir, 'pretrain-final.pt'),
        path.join('./runs/latest', 'pretrain-final.pt'),
    ]
    
    for p in candidate_paths:
        if path.exists(p):
            pretrain_path = p
            break
    
    if pretrain_path:
        # Check if n_compress_tokens matches before loading
        try:
            ckpt = torch.load(pretrain_path, map_location='cpu')
            pretrain_n_compress = ckpt.get('config', {}).get('n_compress_tokens', None)
            
            # Also check from the actual weight shape
            if pretrain_n_compress is None:
                for k, v in ckpt['model'].items():
                    if 'compress_queries' in k:
                        pretrain_n_compress = v.shape[1]
                        break
            
            if pretrain_n_compress == config['n_compress_tokens']:
                if is_main:
                    print(f'Loading pre-trained compression from {pretrain_path}')
                model.load_pretrained_compression(pretrain_path)
                pretrain_loaded = True
            else:
                if is_main:
                    print(f'WARNING: Pretrained compression has n_compress_tokens={pretrain_n_compress}, '
                          f'but current model has n_compress_tokens={config["n_compress_tokens"]}. '
                          f'Training compression from scratch.')
        except Exception as e:
            if is_main:
                print(f'WARNING: Could not load pretrain checkpoint: {e}. Training from scratch.')
    else:
        if is_main:
            print('WARNING: No pre-trained compression found. Training from scratch.')
    
    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {config['model']}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    # Create datasets (use train_n_stream like train_latent_ablation.py)
    train_dataset = CompressedADDataset(
        config, 
        config['traj_dir'], 
        'train', 
        config['train_n_stream'], 
        config['train_source_timesteps']
    )
    
    test_dataset = ADDataset(
        config, 
        config['traj_dir'], 
        'test', 
        1, 
        config['train_source_timesteps']
    )
    
    train_dataloader = get_cad_data_loader(
        train_dataset, 
        batch_size=config['train_batch_size'], 
        config=config, 
        shuffle=True
    )
    train_dataloader = next_dataloader(train_dataloader)
    
    test_collate_fn = partial(ad_collate_fn, grid_size=config['grid_size'])
    from torch.utils.data import DataLoader
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config['test_batch_size'], 
        shuffle=False,
        collate_fn=test_collate_fn,
        num_workers=0,
        persistent_workers=False
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(), 
        lr=config['lr'], 
        betas=(config['beta1'], config['beta2']), 
        weight_decay=config['weight_decay']
    )
    
    lr_sched = get_curriculum_aware_scheduler(
        optimizer=optimizer,
        curriculum=curriculum,
        total_steps=config['train_timesteps'],
        initial_warmup_steps=config.get('num_warmup_steps', 1000),
        stage_warmup_steps=config.get('stage_warmup_steps', 500),
        min_lr_ratio=config.get('min_lr_ratio', 0.1),
    )
    
    step = 0
    
    # Check for existing checkpoint (SAME AS train_latent_ablation.py)
    ckpt_paths = sorted(glob(path.join(log_dir, 'ckpt-*.pt')))
    if len(ckpt_paths) > 0:
        ckpt_path = ckpt_paths[-1]
        ckpt = torch.load(ckpt_path, map_location=config['device'])
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_sched.load_state_dict(ckpt['lr_sched'])
        step = ckpt['step']
        if is_main:
            print(f'Resumed from {ckpt_path} at step {step}')
    
    # Setup evaluation environments
    env_name = config['env']
    train_env_args, test_env_args = SAMPLE_ENVIRONMENT[env_name](config)
    train_env_args = train_env_args[:10]
    test_env_args = test_env_args[:10]
    env_args = train_env_args + test_env_args
    
    if env_name == "darkroom":
        envs = SubprocVecEnv([make_env(config, goal=arg) for arg in env_args])
    elif env_name == "dark_key_to_door":
        envs = SubprocVecEnv([make_env(config, key=arg[:2], goal=arg[2:]) for arg in env_args])
    else:
        raise NotImplementedError(f'Environment not supported: {env_name}')
    
    # Prepare for distributed training
    model, optimizer, train_dataloader, lr_sched = accelerator.prepare(
        model, optimizer, train_dataloader, lr_sched
    )
    
    if is_main:
        start_time = datetime.now()
        print(f'Training started at {start_time}')
    
    current_curriculum_stage = -1
    best_eval_reward = -float('inf')
    best_step = 0
    compression_counts = []
    patience_counter = 0
    patience = config.get('early_stopping_patience', 5)
    save_best_model = config.get('save_best_model', True)
    
    # Training loop
    with tqdm(total=config['train_timesteps'], position=0, leave=True, disable=not is_main) as pbar:
        pbar.update(step)
        
        while step < config['train_timesteps']:
            batch = next(train_dataloader)
            step += 1
            
            # Update curriculum
            max_comp = get_curriculum_max_compressions(step, curriculum)
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.set_curriculum(max_comp)
            
            new_stage = get_curriculum_stage(step, curriculum)
            if new_stage != current_curriculum_stage:
                current_curriculum_stage = new_stage
                new_length_dist = get_curriculum_length_distribution(step, curriculum)
                train_dataset.update_length_distribution(new_length_dist)
                if is_main:
                    print(f"\n[Stage {new_stage}] Updated length distribution: {new_length_dist}")
            
            with accelerator.autocast():
                output = model(batch)
            
            loss = output['loss_total']
            compression_counts.append(output['num_compressions'])
            if len(compression_counts) > 1000:
                compression_counts.pop(0)
            
            # Correct order: zero_grad -> backward -> clip -> step
            optimizer.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
            optimizer.step()
            
            if not accelerator.optimizer_step_was_skipped:
                lr_sched.step()
            
            avg_compressions = np.mean(compression_counts) if compression_counts else 0
            pbar.set_postfix(
                loss=loss.item(), 
                acc=output['acc_action'].item(),
                n_comp=output['num_compressions'],
                avg_comp=f'{avg_compressions:.2f}'
            )
            
            # Logging
            if step % config.get('summary_interval', 100) == 0 and is_main:
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/loss_action', output['loss_action'].item(), step)
                writer.add_scalar('train/lr', lr_sched.get_last_lr()[0], step)
                writer.add_scalar('train/acc_action', output['acc_action'].item(), step)
                writer.add_scalar('train/num_compressions', output['num_compressions'], step)
                writer.add_scalar('train/avg_compressions', avg_compressions, step)
                if output.get('loss_recon') is not None:
                    writer.add_scalar('train/loss_recon', output['loss_recon'].item(), step)
            
            # Test dataloader evaluation
            if is_main and step % config.get('eval_interval', 1000) == 0:
                torch.cuda.empty_cache()
                model.eval()
                eval_start_time = datetime.now()
                
                with torch.no_grad():
                    test_loss_action = 0.0
                    test_acc_action = 0.0
                    test_cnt = 0

                    for j, test_batch in enumerate(test_dataloader):
                        test_output = model(test_batch)
                        cnt = len(test_batch['states'])
                        test_loss_action += test_output['loss_action'].item() * cnt
                        test_acc_action += test_output['acc_action'].item() * cnt
                        test_cnt += cnt

                writer.add_scalar('test/loss_action', test_loss_action / test_cnt, step)
                writer.add_scalar('test/acc_action', test_acc_action / test_cnt, step)

                eval_end_time = datetime.now()
                print(f'\n[Test eval] loss={test_loss_action/test_cnt:.4f}, acc={test_acc_action/test_cnt:.4f}, time={eval_end_time - eval_start_time}')
                
                del test_output, test_batch
                model.train()
                torch.cuda.empty_cache()
                gc.collect()
            
            # In-context evaluation
            if step % config.get('gen_interval', 10000) == 0:
                if is_main:
                    torch.cuda.empty_cache()
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.eval()
                    
                    with torch.no_grad():
                        eval_output = unwrapped.evaluate_in_context(
                            vec_env=envs,
                            eval_timesteps=config['horizon'] * 100
                        )
                    
                    eval_reward = np.mean(eval_output['reward_episode'])
                    writer.add_scalar('eval/reward', eval_reward, step)
                    
                    if eval_reward > best_eval_reward:
                        best_eval_reward = eval_reward
                        best_step = step
                        torch.save({
                            'step': step,
                            'model': unwrapped.state_dict(),
                            'eval_reward': eval_reward,
                            'config': config,
                        }, path.join(log_dir, 'best-model.pt'))
                        print(f'New best model saved! reward={eval_reward:.3f} at step {step}')
                    
                    unwrapped.train()
                    print(f"\n[Step {step}] Eval reward: {eval_reward:.4f} (best: {best_eval_reward:.4f} @ {best_step})")
                    
                    del eval_output
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Save checkpoint
            if step % config.get('ckpt_interval', 10000) == 0:
                if is_main:
                    unwrapped = accelerator.unwrap_model(model)
                    torch.save({
                        'step': step,
                        'model': unwrapped.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_sched': lr_sched.state_dict(),
                        'config': config,
                    }, path.join(log_dir, f'ckpt-{step}.pt'))
            
            pbar.update(1)
    
    # Final save
    if is_main:
        unwrapped = accelerator.unwrap_model(model)
        torch.save({
            'step': step,
            'model': unwrapped.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_sched': lr_sched.state_dict(),
            'config': config,
        }, path.join(log_dir, f'ckpt-{step}.pt'))
        
        writer.close()
    
    # Cleanup
    envs.close()
    del model, optimizer, train_dataloader, train_dataset, test_dataset
    gc.collect()
    torch.cuda.empty_cache()
    
    return best_eval_reward


def main():
    parser = argparse.ArgumentParser(description='Train n_transit ablation study')
    parser.add_argument('--configs', type=str, nargs='+', default=None,
                       help='Specific configs to run: transit30, transit45, transit60, transit90, transit120')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    parser.add_argument('--exp_dir', type=str, default='ablation_transit',
                       help='Experiment directory under ./runs/')
    parser.add_argument('--env', type=str, default='darkroom',
                       help='Environment name')
    args = parser.parse_args()
    
    multiprocessing.set_start_method('spawn', force=True)
    
    # Determine which configs to run
    if args.configs:
        configs = [c for c in args.configs if c in ABLATION_CONFIGS]
    else:
        configs = list(ABLATION_CONFIGS.keys())
    
    print(f"n_transit Ablation Study")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configs: {configs}")
    print(f"Seed: {args.seed}")
    print(f"n_compress_tokens: 16 (fixed)")
    print(f"Environment: {args.env}")
    
    results = {}
    
    for config_suffix in configs:
        try:
            best_reward = train_single_config(
                config_suffix=config_suffix,
                seed=args.seed,
                exp_dir=args.exp_dir,
                env=args.env,
            )
            results[config_suffix] = best_reward
        except Exception as e:
            print(f"Error training {config_suffix}: {e}")
            import traceback
            traceback.print_exc()
            results[config_suffix] = None
    
    # Summary
    print(f"\n{'='*70}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*70}")
    for config, reward in results.items():
        n_transit = ABLATION_CONFIGS[config]
        if reward is not None:
            print(f"  {config} (n_transit={n_transit}): best_reward = {reward:.4f}")
        else:
            print(f"  {config} (n_transit={n_transit}): FAILED")


if __name__ == '__main__':
    main()
