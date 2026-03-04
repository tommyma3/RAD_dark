from datetime import datetime
import os
import os.path as path
from modulefinder import ModuleFinder
from glob import glob
import shutil
import argparse
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"functools\.partial will be a method descriptor in future Python versions; wrap it in enum\.member\(\) if you want to preserve the old behavior",
    category=FutureWarning,
    module=r"torch\.distributed\.algorithms\.ddp_comm_hooks\.__init__",
)

from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs

import yaml
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from dataset import ADDataset, RADDataset
from env import SAMPLE_ENVIRONMENT
from model import MODEL
from utils import get_config, get_data_loader, log_in_context, next_dataloader
from transformers import get_cosine_schedule_with_warmup

import multiprocessing
from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import make_env

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_primary_rank = (local_rank == 0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default='ad_dr', help='Model config name in config/model (without .yaml)')
    parser.add_argument('--run_name', type=str, default='', help='Optional suffix to avoid run directory collisions')
    parser.add_argument(
        '--mixed_precision',
        type=str,
        default='auto',
        choices=['auto', 'fp32', 'fp16', 'bf16'],
        help='Mixed precision mode (auto = follow accelerate config)',
    )
    args = parser.parse_args()
    
    config = get_config('./config/env/darkroom.yaml')
    config.update(get_config('./config/algorithm/ppo_darkroom.yaml'))
    config.update(get_config(f'./config/model/{args.model_config}.yaml'))

    run_id = f"{args.model_config}-{config['env']}-seed{config['env_split_seed']}"
    if args.run_name:
        run_id = f"{run_id}-{args.run_name}"
    log_dir = path.join('./runs', run_id)

    config['log_dir'] = log_dir
    config['model_config'] = args.model_config
    config['run_id'] = run_id
    config_save_path = path.join(config['log_dir'], 'config.yaml')
    try:
        # Try to open config file to bypass NFS cache
        with open(config_save_path, 'r') as f:
            f.read(1)
            config_exists = True
    except FileNotFoundError:
        config_exists = False

    if config_exists:
        if is_primary_rank:
            print(f'WARNING: {log_dir} already exists. Skipping...')
        exit(0)        

    config['traj_dir'] = './datasets'
    set_seed(config.get('seed', 42))

    if torch.cuda.is_available():
        allow_tf32 = bool(config.get('allow_tf32', True))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = bool(config.get('cudnn_benchmark', True))

    if args.mixed_precision == 'auto':
        accelerator_mp = None
    elif args.mixed_precision == 'fp32':
        accelerator_mp = 'no'
    else:
        accelerator_mp = args.mixed_precision

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=bool(config.get('ddp_find_unused_parameters', False))
    )
    accelerator = Accelerator(
        mixed_precision=accelerator_mp,
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
        kwargs_handlers=[ddp_kwargs],
    )
    is_main = accelerator.is_main_process
    if not is_main:
        # Keep worker ranks quiet; rank0 still shows warnings/logs.
        warnings.filterwarnings("ignore")
    config['mixed_precision'] = 'fp32' if accelerator.mixed_precision == 'no' else accelerator.mixed_precision

    if is_main:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir, flush_secs=15)
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f)
    else:
        writer = None

    accelerator.wait_for_everyone()
    config['device'] = accelerator.device
    if is_main:
        print('Using Device: ', config['device'])
        print('Number of processes: ', accelerator.num_processes)

    model_name = config['model']
    model = MODEL[model_name](config)

    load_start_time = datetime.now()
    if is_main:
        print(f'Data loading started at {load_start_time}')

    if config['model'] == 'RAD':
        train_dataset = RADDataset(config, config['traj_dir'], 'train', config['train_n_stream'], config['train_source_timesteps'])
    else:
        train_dataset = ADDataset(config, config['traj_dir'], 'train', config['train_n_stream'], config['train_source_timesteps'])
    test_dataset = ADDataset(config, config['traj_dir'], 'test', 1, config['train_source_timesteps'])

    train_dataloader = get_data_loader(train_dataset, batch_size=config['train_batch_size'], config=config, shuffle=True)
    train_dataloader = next_dataloader(train_dataloader)

    test_dataloader = get_data_loader(test_dataset, batch_size=config['test_batch_size'], config=config, shuffle=False)
    
    load_end_time = datetime.now()
    if is_main:
        print()
        print(f'Data loading ended at {load_end_time}')
        print(f'Elapsed time: {load_end_time - load_start_time}')

    optimizer = AdamW(model.parameters(), lr = config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
    lr_sched = get_cosine_schedule_with_warmup(optimizer, config['num_warmup_steps'], config['train_timesteps'])
    step = 0

    ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
    if len(ckpt_paths) > 0:
        ckpt_path = ckpt_paths[-1]
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_sched.load_state_dict(ckpt['lr_sched'])
        step = ckpt['step']
        if is_main:
            print(f'Checkpoint loaded from {ckpt_path}')

    envs = None
    if is_main:
        env_name = config['env']
        train_env_args, test_env_args = SAMPLE_ENVIRONMENT[env_name](config)
        train_env_args = train_env_args[:10]
        test_env_args = test_env_args[:10]
        env_args = train_env_args + test_env_args

        if env_name == "darkroom":
            envs = SubprocVecEnv([make_env(config, goal=arg) for arg in env_args])
        else:
            raise NotImplementedError('Environment not supported')
    
    model, optimizer, train_dataloader, lr_sched = accelerator.prepare(model, optimizer, train_dataloader, lr_sched)

    if is_main:
        start_time = datetime.now()
        print(f'Trainig started at {start_time}')

    with tqdm(total=config['train_timesteps'], position=0, leave=True, disable=not is_main) as pbar:
        pbar.update(step)

        while True:
            batch = next(train_dataloader)
            
            step += 1
            
            with accelerator.autocast():
                output = model(batch)
            
            loss = output.get('loss_total', output['loss_action'])

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            if not accelerator.optimizer_step_was_skipped:
                lr_sched.step()

            if step % max(1, int(config.get('pbar_update_interval', 20))) == 0:
                pbar.set_postfix(loss=loss.item())

            if is_main and step % config['summary_interval'] == 0:
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/loss_action', output['loss_action'], step)
                writer.add_scalar('train/loss_action_per_state', output['loss_action'], step)
                writer.add_scalar('train/lr', lr_sched.get_last_lr()[0], step)
                writer.add_scalar('train/acc_action', output['acc_action'].item(), step)
                writer.add_scalar('train/acc_action_per_state', output['acc_action'].item(), step)
                if 'loss_recon' in output:
                    writer.add_scalar('train/loss_recon', output['loss_recon'].item(), step)
                if 'num_compressions' in output:
                    writer.add_scalar('train/num_compressions', output['num_compressions'], step)


            # Eval
            if is_main and step % config['eval_interval'] == 0:
                torch.cuda.empty_cache()
                model.eval()
                eval_start_time = datetime.now()
                print(f'Evaluating started at {eval_start_time}')

                with torch.no_grad():
                    test_loss_action = 0.0
                    test_acc_action = 0.0
                    test_loss_reward = 0.0
                    test_acc_reward = 0.0
                    test_loss_next_state = 0.0
                    test_acc_next_state = 0.0
                    test_cnt = 0

                    for j, batch in enumerate(test_dataloader):
                        output = model(batch)
                        cnt = len(batch['states'])
                        test_loss_action += output['loss_action'].item() * cnt
                        test_acc_action += output['acc_action'].item() * cnt
                            
                        if config['dynamics']:
                            test_loss_reward += output['loss_reward'].item() * cnt
                            test_acc_reward += output['acc_reward'].item() * cnt
                            test_loss_next_state += output['loss_next_state'].item() * cnt
                            test_acc_next_state += output['acc_next_state'].item() * cnt
                            
                        test_cnt += cnt

                writer.add_scalar('test/loss_action', test_loss_action / test_cnt, step)
                writer.add_scalar('test/loss_action_per_state', test_loss_action / test_cnt, step)
                writer.add_scalar('test/acc_action', test_acc_action / test_cnt, step)              
                writer.add_scalar('test/acc_action_per_state', test_acc_action / test_cnt, step)

                eval_end_time = datetime.now()
                print()
                print(f'Evaluating ended at {eval_end_time}')
                print(f'Elapsed time: {eval_end_time - eval_start_time}')
                model.train()
                torch.cuda.empty_cache()

            pbar.update(1)

            # LOGGING
            if is_main and step % config['ckpt_interval'] == 0:
                # Remove old checkpoints
                ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
                for ckpt_path in ckpt_paths:
                    os.remove(ckpt_path)

                new_ckpt_path = path.join(config['log_dir'], f'ckpt-{step}.pt')
                unwrapped_model = accelerator.unwrap_model(model)

                torch.save({
                    'step': step,
                    'config': config,
                    'model': unwrapped_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_sched': lr_sched.state_dict(),
                }, new_ckpt_path)
                print(f'\nCheckpoint saved to {new_ckpt_path}')

            if step >= config['train_timesteps']:
                break

    if is_main:
        writer.flush()
        if envs is not None:
            envs.close()

        end_time = datetime.now()
        print()
        print(f'Training ended at {end_time}')
        print(f'Elapsed time: {end_time - start_time}')
