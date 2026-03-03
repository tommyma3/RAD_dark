from torch.utils.data import Dataset
import numpy as np
from utils import get_traj_file_name
import h5py
import random
from einops import rearrange, repeat


class ADDataset(Dataset):
    def __init__(self, config, traj_dir, mode='train', n_stream=None, source_timesteps=None):
        self.config = config
        self.env = config['env']
        self.n_transit = config['n_transit']
        self.dynamics = config['dynamics']
        
        if self.env == 'darkroom':
            n_total_envs = config['grid_size'] ** 2

        else:
            raise ValueError('Invalid env')

        total_env_idx = list(range(n_total_envs))
        random.seed(config['env_split_seed'])
        random.shuffle(total_env_idx)
        
        n_train_envs = round(n_total_envs * config['train_env_ratio'])
        
        if mode == 'train':
            env_idx = total_env_idx[:n_train_envs]
        elif mode == 'test':
            env_idx = total_env_idx[n_train_envs:]
        elif mode == 'all':
            env_idx = total_env_idx
        else:
            raise ValueError('Invalid mode')

        states = []
        actions = []
        rewards = []
        next_states = []

        with h5py.File(f'{traj_dir}/{get_traj_file_name(config)}.hdf5', 'r') as f:
            for i in env_idx:
                states.append(f[f'{i}']['states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
                actions.append(f[f'{i}']['actions'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                rewards.append(f[f'{i}']['rewards'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                next_states.append(f[f'{i}']['next_states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
                    
        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.rewards = np.concatenate(rewards, axis=0)
        self.next_states = np.concatenate(next_states, axis=0)
    
    def __len__(self):
        return (len(self.states[0]) - self.n_transit + 1) * len(self.states)
    
    def __getitem__(self, i):
        history_idx = i // (len(self.states[0]) - self.n_transit + 1)
        transition_idx = i % (len(self.states[0]) - self.n_transit + 1)
            
        traj = {
            'query_states': self.states[history_idx, transition_idx + self.n_transit - 1],
            'target_actions': self.actions[history_idx, transition_idx + self.n_transit - 1],
            'states': self.states[history_idx, transition_idx:transition_idx + self.n_transit - 1],
            'actions': self.actions[history_idx, transition_idx:transition_idx + self.n_transit - 1],
            'rewards': self.rewards[history_idx, transition_idx:transition_idx + self.n_transit - 1],
            'next_states': self.next_states[history_idx, transition_idx:transition_idx + self.n_transit - 1],
        }
        
        if self.dynamics:
            traj.update({
                'target_next_states': self.next_states[history_idx, transition_idx + self.n_transit - 1],
                'target_rewards': self.rewards[history_idx, transition_idx + self.n_transit - 1],
            })
        
        return traj


class RADDataset(Dataset):
    """
    Variable-length dataset for RAD.

    RAD needs long contexts to exercise recursive compression; fixed-length windows
    under-train the compression pathway.
    """

    def __init__(self, config, traj_dir, mode='train', n_stream=None, source_timesteps=None):
        self.config = config
        self.env = config['env']
        self.n_transit = config['n_transit']
        self.dynamics = config['dynamics']

        if self.env == 'darkroom':
            n_total_envs = config['grid_size'] ** 2
        else:
            raise ValueError('Invalid env')

        total_env_idx = list(range(n_total_envs))
        random.seed(config['env_split_seed'])
        random.shuffle(total_env_idx)

        n_train_envs = round(n_total_envs * config['train_env_ratio'])

        if mode == 'train':
            env_idx = total_env_idx[:n_train_envs]
        elif mode == 'test':
            env_idx = total_env_idx[n_train_envs:]
        elif mode == 'all':
            env_idx = total_env_idx
        else:
            raise ValueError('Invalid mode')

        states = []
        actions = []
        rewards = []
        next_states = []

        with h5py.File(f'{traj_dir}/{get_traj_file_name(config)}.hdf5', 'r') as f:
            for i in env_idx:
                states.append(f[f'{i}']['states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
                actions.append(f[f'{i}']['actions'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                rewards.append(f[f'{i}']['rewards'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                next_states.append(f[f'{i}']['next_states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])

        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.rewards = np.concatenate(rewards, axis=0)
        self.next_states = np.concatenate(next_states, axis=0)

        self.seq_length = self.states.shape[1]
        self.n_histories = self.states.shape[0]

        # RAD token budget
        self.tokens_per_transition = 3
        self.rad_max_seq_length = config.get(
            'rad_max_seq_length', self.tokens_per_transition * (self.n_transit - 1) + 1
        )
        self.n_compress_tokens = config.get('n_compress_tokens', 24)
        self.base_capacity = max((self.rad_max_seq_length - 1) // self.tokens_per_transition, 1)
        self.post_compress_capacity = max(
            (self.rad_max_seq_length - self.n_compress_tokens - 1) // self.tokens_per_transition, 1
        )

        default_min = max(20, self.n_transit - 1)
        default_max = min(self.seq_length - 1, config.get('train_source_timesteps', self.seq_length))
        self.min_context_length = config.get('rad_min_context_length', default_min)
        self.max_context_length = min(
            config.get('rad_max_context_length', default_max),
            self.seq_length - 1,
        )

        if self.min_context_length > self.max_context_length:
            raise ValueError(
                f"Invalid RAD context range: min={self.min_context_length}, max={self.max_context_length}"
            )

        self.length_distribution = config.get(
            'rad_length_distribution',
            {'short': 0.25, 'medium': 0.35, 'long': 0.40},
        )
        self._validate_distribution(self.length_distribution)

    def _validate_distribution(self, distribution):
        total = sum(distribution.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f'rad_length_distribution must sum to 1.0, got {total}')

    def __len__(self):
        return self.n_histories * max(1, self.seq_length - self.min_context_length)

    def _sample_context_length(self):
        no_compress_high = max(self.base_capacity, self.min_context_length)
        one_compress_high = max(self.base_capacity + self.post_compress_capacity, no_compress_high)

        ranges = {
            'short': (self.min_context_length, no_compress_high),
            'medium': (no_compress_high + 1, one_compress_high),
            'long': (one_compress_high + 1, self.max_context_length),
        }

        # Clamp ranges
        for key, (low, high) in list(ranges.items()):
            low = min(max(low, self.min_context_length), self.max_context_length)
            high = min(max(high, low), self.max_context_length)
            ranges[key] = (low, high)

        draw = random.random()
        cumulative = 0.0
        for category, prob in self.length_distribution.items():
            cumulative += prob
            if draw <= cumulative:
                low, high = ranges.get(category, ranges['medium'])
                return random.randint(low, high)

        # Fallback
        return random.randint(self.min_context_length, self.max_context_length)

    def __getitem__(self, i):
        history_idx = i % self.n_histories

        context_length = self._sample_context_length()
        max_end = self.seq_length - 1
        min_end = context_length
        if min_end >= max_end:
            end_idx = max_end
        else:
            end_idx = random.randint(min_end, max_end)

        start_idx = end_idx - context_length

        traj = {
            'query_states': self.states[history_idx, end_idx],
            'target_actions': self.actions[history_idx, end_idx],
            'states': self.states[history_idx, start_idx:end_idx],
            'actions': self.actions[history_idx, start_idx:end_idx],
            'rewards': self.rewards[history_idx, start_idx:end_idx],
            'next_states': self.next_states[history_idx, start_idx:end_idx],
            'context_length': context_length,
        }

        if self.dynamics:
            traj.update({
                'target_next_states': self.next_states[history_idx, end_idx],
                'target_rewards': self.rewards[history_idx, end_idx],
            })

        return traj
