"""
data.py
-------
For now this generates SYNTHETIC traffic-speed data so the training pipeline
can be sanity-checked end to end before the real datasets are wired in.

The synthetic generator deliberately bakes in spatial + temporal structure
(a diurnal sine wave shared by clustered "neighbourhoods" of sensors, plus
noise) so that a model that actually uses the graph structure should beat a
model that ignores it -- which is exactly the property you want to verify
before trusting results on the real data.

TO SWITCH TO REAL DATA (METR-LA / PeMS-BAY):
  Both datasets are commonly distributed as:
    - a (T_total, N) speed matrix, e.g. `metr-la.h5` (pandas DataFrame, DatetimeIndex)
    - a (N, N) or (N, 2) sensor distance file used to build the adjacency
  Replace `make_synthetic_dataset()` below with a loader that:
    1. reads the .h5/.csv into a (T_total, N) numpy array `speed_matrix`
    2. builds `static_adj` from the sensor distance file with a thresholded
       Gaussian kernel (standard STGCN/DCRNN preprocessing):
           W[i,j] = exp(-(dist[i,j]**2) / sigma**2)   if dist[i,j] < threshold else 0
    3. slices `speed_matrix` into (X, Y) windows with `make_windows()` below,
       which is dataset-agnostic and does not need to change.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def make_synthetic_dataset(num_nodes: int = 50, num_timesteps: int = 2000,
                            num_clusters: int = 5, seed: int = 0):
    """
    Returns:
        speed_matrix : (num_timesteps, num_nodes) float32, values roughly in [0,70] mph
        static_adj   : (num_nodes, num_nodes) float32, row-normalised, distance-based
        tod_idx      : (num_timesteps,) int64, 5-minute time-of-day slot in [0, 288)
        dow_idx      : (num_timesteps,) int64, day-of-week in [0, 7)
    """
    rng = np.random.default_rng(seed)

    # --- assign sensors to spatial clusters, sensors in the same cluster
    #     share correlated traffic patterns (this is the structure an
    #     adaptive graph should be able to discover) ---
    cluster_id = rng.integers(0, num_clusters, size=num_nodes)
    cluster_phase = rng.uniform(0, 2 * np.pi, size=num_clusters)
    cluster_amp = rng.uniform(10, 20, size=num_clusters)

    t = np.arange(num_timesteps)
    tod_idx = (t % 288).astype(np.int64)          # 288 slots/day at 5-min resolution
    dow_idx = ((t // 288) % 7).astype(np.int64)

    speed_matrix = np.zeros((num_timesteps, num_nodes), dtype=np.float32)
    base_speed = 55.0
    for n in range(num_nodes):
        c = cluster_id[n]
        diurnal = cluster_amp[c] * np.sin(2 * np.pi * tod_idx / 288 + cluster_phase[c])
        node_noise = rng.normal(0, 2.0, size=num_timesteps)
        speed_matrix[:, n] = base_speed - np.abs(diurnal) + node_noise
    speed_matrix = np.clip(speed_matrix, 5, 70)

    # --- static adjacency: sensors in the same cluster are "close" ---
    static_adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                static_adj[i, j] = 1.0
            elif cluster_id[i] == cluster_id[j]:
                static_adj[i, j] = rng.uniform(0.4, 0.8)
            else:
                static_adj[i, j] = rng.uniform(0.0, 0.05)
    row_sums = static_adj.sum(axis=1, keepdims=True)
    static_adj = static_adj / row_sums

    return speed_matrix, static_adj, tod_idx, dow_idx


def make_windows(speed_matrix, tod_idx, dow_idx, in_len=12, horizons=(3, 6, 12)):
    """
    Slices a (T_total, N) speed matrix into (input window, multi-horizon
    targets) pairs. horizons are given in "steps ahead" (e.g. at 5-minute
    sampling, 3/6/12 steps = 15/30/60 minutes).
    """
    max_h = max(horizons)
    T_total, N = speed_matrix.shape
    X, Y, TOD, DOW = [], [], [], []
    for start in range(T_total - in_len - max_h):
        end = start + in_len
        X.append(speed_matrix[start:end])                      # (in_len, N)
        TOD.append(tod_idx[start:end])
        DOW.append(dow_idx[start:end])
        targets = [speed_matrix[end + h - 1] for h in horizons]  # each (N,)
        Y.append(np.stack(targets, axis=-1))                    # (N, num_horizons)
    return (np.stack(X), np.stack(Y), np.stack(TOD), np.stack(DOW))


class TrafficWindowDataset(Dataset):
    def __init__(self, X, Y, TOD, DOW):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.TOD = torch.tensor(TOD, dtype=torch.long)
        self.DOW = torch.tensor(DOW, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.TOD[idx], self.DOW[idx]
