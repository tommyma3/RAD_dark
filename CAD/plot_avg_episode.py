"""
Plot average reward per episode in a nearly-square figure with environment names.

Usage:
    python plot_avg_episode.py --dir ./runs/AD-darkroom-seed0 --output ./runs/avg_episode_AD.png

This script expects `eval_result.npy` inside provided directories and will save a nearly-square PNG.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings


def load_eval_results(result_path):
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Evaluation result not found: {result_path}")
    return np.load(result_path)


def combined_smooth(x, window_size=10):
    """Compute a smoothed array using 'same' on the left edge and 'valid' on the right edge.

    Approach:
    - Compute conv_same = convolve(x, window, mode='same')
    - Compute conv_valid = convolve(x, window, mode='valid')
    - Build result so that left half (first window_size-1 points) uses conv_same,
      middle uses conv_valid aligned to the right of conv_same, and the tail uses conv_valid.

    This provides smoothed values that include left-edge smoothing while keeping the right
    side free of zero-padding bias.
    """
    w = np.ones(window_size) / window_size
    conv_same = np.convolve(x, w, mode='same')
    conv_valid = np.convolve(x, w, mode='valid')

    n = len(x)
    if window_size <= 1:
        return x.copy()

    # Build output: use conv_same for the left (0..window-2), use conv_valid for indices window-1..n-1
    out = np.empty(n)
    left_cut = window_size - 1
    out[:left_cut] = conv_same[:left_cut]

    # conv_valid has length n-window+1 and corresponds to indices [window-1 .. n-1]
    out[left_cut:] = conv_valid
    return out


def left_pad_ma(x, window_size=10):
    """Moving average where only the left edge is zero-padded (matches compare_eval.py behavior).

    We prepend window_size-1 zeros to the array then perform a 'valid' convolution. The
    result has the same length as the input and the left-most points are affected by the
    implicit zero-padding while the right edge uses full-window averages.
    """
    if window_size <= 1:
        return x.copy()
    w = np.ones(window_size) / window_size
    pad = window_size - 1
    xp = np.concatenate([np.zeros(pad, dtype=x.dtype), x])
    y = np.convolve(xp, w, mode='valid')
    return y


def gaussian_smooth(x, sigma=3.0):
    """Gaussian smoothing using reflect padding to avoid edge artifacts."""
    if sigma <= 0:
        return x.copy()
    radius = int(3 * sigma)
    kernel_x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (kernel_x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    xp = np.pad(x, (radius, radius), mode='reflect')
    y = np.convolve(xp, kernel, mode='valid')
    return y


def ema_smooth(x, alpha=0.1):
    """Exponential moving average smoothing. alpha in (0,1]."""
    if alpha <= 0 or alpha > 1:
        return x.copy()
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def try_savgol(x, window_length=51, polyorder=3):
    try:
        from scipy.signal import savgol_filter
    except Exception:
        savgol_filter = None
    if savgol_filter is None:
        warnings.warn('scipy not available; falling back to gaussian smoothing for savgol request')
        sigma = max(1.0, (window_length / 6.0))
        return gaussian_smooth(x, sigma=sigma)
    # ensure window_length is odd and <= len(x)
    wl = int(window_length)
    if wl % 2 == 0:
        wl += 1
    wl = min(wl, len(x) if len(x) % 2 == 1 else len(x) - 1)
    if wl < polyorder + 2:
        wl = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
    return savgol_filter(x, wl, polyorder)


def plot_avg_episode(ad_path, cad_path, save_path=None, figsize=(4, 3)):
    ad = load_eval_results(os.path.join(ad_path, 'eval_result.npy'))
    cad = load_eval_results(os.path.join(cad_path, 'eval_result.npy'))

    # Expect shape (n_envs, n_episodes)
    n_envs, n_episodes = ad.shape
    episodes = np.arange(1, n_episodes + 1)

    fig, ax = plt.subplots(figsize=figsize)

    # window_size may be injected as attribute on the function by caller
    window_size = getattr(plot_avg_episode, 'window_size', 20)
    smooth_method = getattr(plot_avg_episode, 'smooth_method', 'ma')
    smooth_params = getattr(plot_avg_episode, 'smooth_params', {})

    # Compute per-episode mean across environments
    ad_mean = ad.mean(axis=0)
    cad_mean = cad.mean(axis=0)

    # Compute standard error across environments (use ddof=1 when possible)
    if n_envs > 1:
        ad_sem = ad.std(axis=0, ddof=1) / np.sqrt(n_envs)
        cad_sem = cad.std(axis=0, ddof=1) / np.sqrt(n_envs)
    else:
        ad_sem = np.zeros_like(ad_mean)
        cad_sem = np.zeros_like(cad_mean)

    # Select smoothing method and apply the same smoothing to SEM
    if smooth_method == 'ma':
        # Use left-side zero-padded moving average to match compare_eval.py
        ad_smooth = left_pad_ma(ad_mean, window_size=window_size)
        cad_smooth = left_pad_ma(cad_mean, window_size=window_size)
        ad_sem_smooth = left_pad_ma(ad_sem, window_size=window_size)
        cad_sem_smooth = left_pad_ma(cad_sem, window_size=window_size)
    elif smooth_method == 'gaussian':
        sigma = float(smooth_params.get('sigma', max(1.0, window_size / 6.0)))
        ad_smooth = gaussian_smooth(ad_mean, sigma=sigma)
        cad_smooth = gaussian_smooth(cad_mean, sigma=sigma)
        ad_sem_smooth = gaussian_smooth(ad_sem, sigma=sigma)
        cad_sem_smooth = gaussian_smooth(cad_sem, sigma=sigma)
    elif smooth_method == 'ema':
        alpha = float(smooth_params.get('alpha', 0.08))
        ad_smooth = ema_smooth(ad_mean, alpha=alpha)
        cad_smooth = ema_smooth(cad_mean, alpha=alpha)
        ad_sem_smooth = ema_smooth(ad_sem, alpha=alpha)
        cad_sem_smooth = ema_smooth(cad_sem, alpha=alpha)
    elif smooth_method == 'savgol':
        wl = int(smooth_params.get('window_length', max(3, window_size // 1)))
        po = int(smooth_params.get('polyorder', 3))
        ad_smooth = try_savgol(ad_mean, window_length=wl, polyorder=po)
        cad_smooth = try_savgol(cad_mean, window_length=wl, polyorder=po)
        # Savgol on SEM is acceptable to produce smooth error bands
        ad_sem_smooth = try_savgol(ad_sem, window_length=wl, polyorder=po)
        cad_sem_smooth = try_savgol(cad_sem, window_length=wl, polyorder=po)
    else:
        warnings.warn(f'Unknown smoothing method {smooth_method}; falling back to moving average')
        ad_smooth = combined_smooth(ad_mean, window_size=window_size)
        cad_smooth = combined_smooth(cad_mean, window_size=window_size)
        ad_sem_smooth = combined_smooth(ad_sem, window_size=window_size)
        cad_sem_smooth = combined_smooth(cad_sem, window_size=window_size)

    ax.plot(episodes, ad_smooth, label='AD', color='blue', linewidth=2)
    ax.fill_between(episodes, ad_smooth - ad_sem_smooth, ad_smooth + ad_sem_smooth,
                    color='blue', alpha=0.2)
    ax.plot(episodes, cad_smooth, label='RAD', color='red', linewidth=2)
    ax.fill_between(episodes, cad_smooth - cad_sem_smooth, cad_smooth + cad_sem_smooth,
                    color='red', alpha=0.2)

    # (Do not plot raw noisy lines.) Only show smoothed curves.

    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.set_title('Average Reward per Episode (AD vs CAD)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, n_episodes])

    # Show the environment name prominently if provided (set by caller)
    # The caller may pass the env_name via the `main()` helper; if not, nothing is shown here.
    # `env_name` will be injected by caller via function attribute if present.
    env_name = getattr(plot_avg_episode, "env_name", None)
    if env_name:
        ax.set_title(f'Average Reward per Episode ({env_name})')
    else:
        ax.set_title('Average Reward per Episode (AD vs CAD)')

    plt.tight_layout()
    if save_path:
        out_path = Path(save_path)
        if out_path.parent and not out_path.parent.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
        print(f"Saved figure to {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ad_dir', type=str, required=True)
    parser.add_argument('--cad_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='./runs/comparison/avg_episode.png')
    parser.add_argument('--window_size', type=int, default=30,
                        help='Moving average window size for smoothing (default: 20)')
    parser.add_argument('--smooth_method', type=str, default='ma',
                        choices=['ma', 'gaussian', 'ema', 'savgol'],
                        help='Smoothing method: ma (moving average), gaussian, ema (exponential), savgol')
    parser.add_argument('--sigma', type=float, default=3.0,
                        help='Sigma for gaussian smoothing (default 3.0)')
    parser.add_argument('--ema_alpha', type=float, default=0.08,
                        help='Alpha for EMA smoothing (default 0.08)')
    parser.add_argument('--savgol_window', type=int, default=51,
                        help='Window length for Savitzky-Golay filter (odd)')
    parser.add_argument('--savgol_poly', type=int, default=3,
                        help='Polynomial order for Savitzky-Golay filter')
    parser.add_argument('--env_name', type=str, default=None,
                        help='Optional environment name (e.g. darkroom, dark_key_to_door)')
    args = parser.parse_args()

    # Infer environment name from directory names if not provided
    env_name = args.env_name
    if env_name is None:
        # try to extract from directory basename by removing common prefixes/suffixes
        def clean(name):
            bn = Path(name).name
            # remove AD- or CAD- prefixes
            if bn.startswith('AD-') or bn.startswith('CAD-'):
                bn = bn.split('-', 1)[1]
            # remove seed suffix like '-seed0' or '-seed42'
            if '-seed' in bn:
                bn = bn.split('-seed', 1)[0]
            return bn

        ad_bn = clean(args.ad_dir)
        cad_bn = clean(args.cad_dir)
        # If both cleaned names match or one contains the other, use the common part
        if ad_bn == cad_bn:
            env_name = ad_bn
        else:
            # pick the longest common substring split on non-alphanum
            # fallback to ad_bn
            env_name = ad_bn

    # attach to function so plot can access it
    setattr(plot_avg_episode, 'env_name', env_name)
    setattr(plot_avg_episode, 'window_size', args.window_size)
    setattr(plot_avg_episode, 'smooth_method', args.smooth_method)
    setattr(plot_avg_episode, 'smooth_params', {
        'sigma': args.sigma,
        'alpha': args.ema_alpha,
        'window_length': args.savgol_window,
        'polyorder': args.savgol_poly,
    })

    plot_avg_episode(args.ad_dir, args.cad_dir, save_path=args.output)


if __name__ == '__main__':
    main()
