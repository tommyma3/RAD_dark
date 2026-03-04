import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np

from env import map_dark_states
from functools import partial
import matplotlib.pyplot as plt


def get_config(config_path):
    with open(config_path, 'r') as f:
        new_config = yaml.full_load(f)
    config = {}
    if 'include' in new_config:
        include_config = get_config(new_config['include'])
        config.update(include_config)
        del new_config['include']
    config.update(new_config)
    return config


def get_traj_file_name(config):
    if config["env"] == 'metaworld':
        task = config['task']
    else:
        task = config['env']

    path = f'history_{task}_{config["alg"]}_alg-seed{config["alg_seed"]}'

    return path

def ad_collate_fn(batch, grid_size):
    res = {}
    res['query_states'] = torch.tensor(np.array([item['query_states'] for item in batch]), requires_grad=False, dtype=torch.float)
    res['target_actions'] = torch.tensor(np.array([item['target_actions'] for item in batch]), requires_grad=False, dtype=torch.long)
    res['states'] = torch.tensor(np.array([item['states'] for item in batch]), requires_grad=False, dtype=torch.float)
    res['actions'] = torch.tensor(np.array([item['actions'] for item in batch]), requires_grad=False, dtype=torch.long)
    res['rewards'] = torch.tensor(np.array([item['rewards'] for item in batch]), dtype=torch.float, requires_grad=False)
    res['next_states'] = torch.tensor(np.array([item['next_states'] for item in batch]), requires_grad=False, dtype=torch.float)
    
    if 'target_next_states' in batch[0].keys():
        res['target_next_states'] = map_dark_states(torch.tensor(np.array([item['target_next_states'] for item in batch]), dtype=torch.long, requires_grad=False), grid_size=grid_size)
        res['target_rewards'] = torch.tensor(np.array([item['target_rewards'] for item in batch]), dtype=torch.long, requires_grad=False)
        
    return res


def rad_collate_fn(batch, grid_size):
    batch_size = len(batch)
    max_context_len = max(item['states'].shape[0] for item in batch)
    dim_state = batch[0]['states'].shape[1]

    states = np.zeros((batch_size, max_context_len, dim_state), dtype=np.float32)
    actions = np.zeros((batch_size, max_context_len), dtype=np.int64)
    rewards = np.zeros((batch_size, max_context_len), dtype=np.float32)
    next_states = np.zeros((batch_size, max_context_len, dim_state), dtype=np.float32)
    context_lengths = np.zeros((batch_size,), dtype=np.int64)

    query_states = []
    target_actions = []

    for i, item in enumerate(batch):
        ctx_len = item['states'].shape[0]
        states[i, :ctx_len] = item['states']
        actions[i, :ctx_len] = item['actions']
        rewards[i, :ctx_len] = item['rewards']
        next_states[i, :ctx_len] = item['next_states']
        context_lengths[i] = ctx_len
        query_states.append(item['query_states'])
        target_actions.append(item['target_actions'])

    res = {}
    res['query_states'] = torch.tensor(np.array(query_states), requires_grad=False, dtype=torch.float)
    res['target_actions'] = torch.tensor(np.array(target_actions), requires_grad=False, dtype=torch.long)
    res['states'] = torch.tensor(states, requires_grad=False, dtype=torch.float)
    res['actions'] = torch.tensor(actions, requires_grad=False, dtype=torch.long)
    res['rewards'] = torch.tensor(rewards, requires_grad=False, dtype=torch.float)
    res['next_states'] = torch.tensor(next_states, requires_grad=False, dtype=torch.float)
    res['context_lengths'] = torch.tensor(context_lengths, requires_grad=False, dtype=torch.long)
    return res

def get_data_loader(dataset, batch_size, config, shuffle=True):
    if config.get('model') == 'RAD':
        collate_fn = partial(rad_collate_fn, grid_size=config['grid_size'])
    else:
        collate_fn = partial(ad_collate_fn, grid_size=config['grid_size'])

    num_workers = int(config.get('num_workers', 0))
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'collate_fn': collate_fn,
        'num_workers': num_workers,
        'pin_memory': bool(config.get('pin_memory', torch.cuda.is_available())),
    }

    if num_workers > 0:
        loader_kwargs['persistent_workers'] = bool(config.get('persistent_workers', True))
        loader_kwargs['prefetch_factor'] = int(config.get('prefetch_factor', 4))

    return DataLoader(dataset, **loader_kwargs)

def log_in_context(values: np.ndarray, max_reward: int, episode_length: int, tag: str, title: str, xlabel: str, ylabel: str, step: int, success=None, writer=None) -> None:
    steps = np.arange(1, len(values[0])+1) * episode_length
    mean_value = values.mean(axis=0)
    
    plt.plot(steps, mean_value)
    
    if success is not None:
        success_rate = success.astype(np.float32).mean(axis=0)

        for i, (xi, yi) in enumerate(zip(steps, mean_value)):
            if (i+1) % 10 == 0:
                plt.annotate(f'{success_rate[i]:.2f}', (xi, yi))
        
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.ylim(-max_reward * 0.05, max_reward * 1.05)
    writer.add_figure(f'{tag}/mean', plt.gcf(), global_step=step)
    plt.close()

def next_dataloader(dataloader: DataLoader):
    """
    Makes the dataloader never end when the dataset is exhausted.
    This is done to remove the notion of an 'epoch' and to count only the amount
    of training steps.
    """
    while True:
        for batch in dataloader:
            yield batch
