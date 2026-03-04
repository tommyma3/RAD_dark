from torch.utils.data import Dataset
import numpy as np
from utils import get_traj_file_name
import h5py
import random
from einops import rearrange, repeat


def _load_env_streams(group, n_stream, source_timesteps, random_timestep_slice, rng):
    states = group['states'][()].transpose(1, 0, 2)
    actions = group['actions'][()].transpose(1, 0)
    rewards = group['rewards'][()].transpose(1, 0)
    next_states = group['next_states'][()].transpose(1, 0, 2)

    if n_stream is not None:
        states = states[:n_stream]
        actions = actions[:n_stream]
        rewards = rewards[:n_stream]
        next_states = next_states[:n_stream]

    if source_timesteps is None:
        return states, actions, rewards, next_states

    source_timesteps = int(source_timesteps)
    if source_timesteps <= 0:
        raise ValueError(f"source_timesteps must be positive, got {source_timesteps}")

    total_timesteps = states.shape[1]
    if total_timesteps <= source_timesteps:
        return (
            states[:, :source_timesteps],
            actions[:, :source_timesteps],
            rewards[:, :source_timesteps],
            next_states[:, :source_timesteps],
        )

    if not random_timestep_slice:
        return (
            states[:, :source_timesteps],
            actions[:, :source_timesteps],
            rewards[:, :source_timesteps],
            next_states[:, :source_timesteps],
        )

    max_start = total_timesteps - source_timesteps
    starts = rng.integers(0, max_start + 1, size=states.shape[0])

    states_slice = np.stack(
        [states[i, st:st + source_timesteps] for i, st in enumerate(starts)],
        axis=0,
    )
    actions_slice = np.stack(
        [actions[i, st:st + source_timesteps] for i, st in enumerate(starts)],
        axis=0,
    )
    rewards_slice = np.stack(
        [rewards[i, st:st + source_timesteps] for i, st in enumerate(starts)],
        axis=0,
    )
    next_states_slice = np.stack(
        [next_states[i, st:st + source_timesteps] for i, st in enumerate(starts)],
        axis=0,
    )
    return states_slice, actions_slice, rewards_slice, next_states_slice


class ADDataset(Dataset):
    def __init__(self, config, traj_dir, mode='train', n_stream=None, source_timesteps=None):
        self.config = config
        self.env = config['env']
        self.n_transit = config['n_transit']
        self.dynamics = config['dynamics']
        self.random_timestep_slice = bool(config.get('random_timestep_slice', False) and mode == 'train')
        seed_base = int(config.get('seed', 42))
        self._slice_rng = np.random.default_rng(seed_base + (0 if mode == 'train' else 10_000))
        
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
                grp = f.get(f'{i}')
                if grp is None:
                    continue
                s, a, r, ns = _load_env_streams(
                    group=grp,
                    n_stream=n_stream,
                    source_timesteps=source_timesteps,
                    random_timestep_slice=self.random_timestep_slice,
                    rng=self._slice_rng,
                )
                states.append(s)
                actions.append(a)
                rewards.append(r)
                next_states.append(ns)

        if len(states) == 0:
            raise ValueError('No trajectories were loaded for ADDataset.')
                    
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
        self.random_timestep_slice = bool(config.get('random_timestep_slice', True) and mode == 'train')
        seed_base = int(config.get('seed', 42))
        self._slice_rng = np.random.default_rng(seed_base + (0 if mode == 'train' else 10_000))

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
                grp = f.get(f'{i}')
                if grp is None:
                    continue
                s, a, r, ns = _load_env_streams(
                    group=grp,
                    n_stream=n_stream,
                    source_timesteps=source_timesteps,
                    random_timestep_slice=self.random_timestep_slice,
                    rng=self._slice_rng,
                )
                states.append(s)
                actions.append(a)
                rewards.append(r)
                next_states.append(ns)

        if len(states) == 0:
            raise ValueError('No trajectories were loaded for RADDataset.')

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
        self.context_length_step = max(1, int(config.get('rad_context_length_step', 8)))

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
                sampled = random.randint(low, high)
                return self._quantize_context_length(sampled)

        # Fallback
        sampled = random.randint(self.min_context_length, self.max_context_length)
        return self._quantize_context_length(sampled)

    def _quantize_context_length(self, context_length):
        step = self.context_length_step
        quantized = (context_length // step) * step
        if quantized < self.min_context_length:
            quantized = self.min_context_length
        if quantized > self.max_context_length:
            quantized = self.max_context_length
        return int(quantized)

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
