"""
m1_data/dataset.py  --  STEP 3: clean the data and cut it into training samples.

Pipeline (the order matters):
    raw speed CSV
      -> add timestamps (time-of-day, day-of-week)
      -> mark missing readings (sensor reported 0 mph = sensor failure)
      -> chronological split 70% train / 10% val / 20% test
      -> z-score normalisation (mean / std from the TRAIN part only)
      -> sliding windows: 12 past steps  ->  speed 15 / 30 / 60 min later
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config import (BATCH_SIZE, DATA_DIR, DATASETS, HORIZONS, IN_LEN,
                    STEPS_PER_DAY, TRAIN_RATIO, VAL_RATIO)
from m1_data.graph import build_adjacency


# --------------------------------------------------------------------- loading
def load_raw(name: str):
    """Read the downloaded files of one dataset."""
    folder = DATA_DIR / name
    if not (folder / "speed.csv").exists():
        raise FileNotFoundError(
            f"{folder}/speed.csv not found. Run: python -m m1_data.download_data")
    speed = pd.read_csv(folder / "speed.csv", header=None).values.astype(np.float32)
    sensor_ids = (folder / "sensor_ids.txt").read_text().strip().split(",")
    distances = pd.read_csv(folder / "distances.csv", header=None, names=["from", "to", "cost"], dtype=str)
    distances = distances[distances["from"] != "from"]      # the LA file has a header row, BAY does not
    distances["cost"] = distances["cost"].astype(float)
    locations = pd.read_csv(folder / "locations.csv")
    locations["sensor_id"] = locations["sensor_id"].astype(str)
    locations = locations.set_index("sensor_id").loc[sensor_ids].reset_index()
    return speed, sensor_ids, distances, locations[["sensor_id", "latitude", "longitude"]]


def make_timestamps(name: str, num_steps: int) -> pd.DatetimeIndex:
    """The CSV has no time column, so we rebuild it from the known start time."""
    info = DATASETS[name]
    times = pd.date_range(info["start"], periods=num_steps, freq="5min", tz=info["timezone"])
    return times.tz_localize(None) if info["timezone"] else times


# --------------------------------------------------------------------- normaliser
class StandardScaler:
    """z = (x - mean) / std.  Neural networks train much better on values near 0."""

    def __init__(self, mean: float, std: float):
        self.mean, self.std = float(mean), float(std)

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, z):
        return z * self.std + self.mean


# --------------------------------------------------------------------- one bundle with everything
def prepare_dataset(name: str) -> dict:
    """Runs the whole Module-1 pipeline and returns everything the models need."""
    speed, sensor_ids, distances, locations = load_raw(name)
    T, N = speed.shape
    times = make_timestamps(name, T)

    # --- time features (used by the transformer's embeddings)
    tod = (times.hour * 12 + times.minute // 5).values.astype(np.int64)   # 0..287
    dow = times.dayofweek.values.astype(np.int64)                         # 0=Mon..6=Sun

    # --- missing values: a reading of exactly 0 mph means the sensor failed
    missing = speed == 0
    # For the MODEL INPUT we fill a gap with the sensor's last good reading.
    # For the TARGETS we keep the zeros and simply ignore them in the loss/metrics.
    filled = pd.DataFrame(np.where(missing, np.nan, speed)).ffill().bfill().values.astype(np.float32)

    # --- chronological split (by time, never shuffled)
    n_train = int(T * TRAIN_RATIO)
    n_val = int(T * VAL_RATIO)
    splits = {"train": (0, n_train),
              "val": (n_train, n_train + n_val),
              "test": (n_train + n_val, T)}

    # --- normalisation statistics from the TRAINING part only (no peeking at the future)
    train_values = speed[:n_train][~missing[:n_train]]
    scaler = StandardScaler(train_values.mean(), train_values.std())

    # --- graph
    W = build_adjacency(distances, sensor_ids)

    return {
        "name": name,
        "speed_raw": speed,                       # (T, N) with 0 = missing  -> targets
        "speed_input": scaler.transform(filled),  # (T, N) normalised, gaps filled -> inputs
        "tod": tod, "dow": dow, "times": times,
        "missing_ratio": float(missing.mean()),
        "splits": splits, "scaler": scaler,
        "adjacency": W, "sensor_ids": sensor_ids, "locations": locations,
        "num_nodes": N, "num_steps": T,
    }


# --------------------------------------------------------------------- windows
class TrafficWindows(Dataset):
    """
    Sample k  (one training example):
        x   = normalised speed at steps  [s, s+12)            shape (12, N)
        y   = raw speed at steps s+12+h-1 for h in (3, 6, 12) shape (N, 3)
        tod, dow = time features of the 12 input steps        shape (12,)
    Windows are cut on the fly, so we never store a huge copy of the data.
    """

    def __init__(self, data: dict, split: str):
        self.x = torch.from_numpy(data["speed_input"])
        self.y = torch.from_numpy(data["speed_raw"])
        self.tod = torch.from_numpy(data["tod"])
        self.dow = torch.from_numpy(data["dow"])
        start, end = data["splits"][split]
        last_start = end - IN_LEN - max(HORIZONS)          # last target must stay inside the split
        self.starts = np.arange(start, last_start + 1)
        self.horizons = list(HORIZONS)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, k):
        s = int(self.starts[k])
        e = s + IN_LEN
        target_rows = [e + h - 1 for h in self.horizons]   # e.g. 3 steps ahead = row e+2
        x = self.x[s:e]                                    # (12, N)
        y = self.y[target_rows].T                          # (N, 3)
        return x, y, self.tod[s:e], self.dow[s:e]


def get_loaders(data: dict, batch_size: int = BATCH_SIZE):
    """Train loader is shuffled (order of samples), but the split itself is by time."""
    return {
        split: DataLoader(TrafficWindows(data, split), batch_size=batch_size,
                          shuffle=(split == "train"), drop_last=(split == "train"))
        for split in ("train", "val", "test")
    }


if __name__ == "__main__":
    # quick self-check:  python -m m1_data.dataset
    for name in ["metr-la"]:
        d = prepare_dataset(name)
        loaders = get_loaders(d)
        x, y, tod, dow = next(iter(loaders["train"]))
        print(f"{name}: {d['num_steps']} steps x {d['num_nodes']} sensors, "
              f"{d['times'][0]} -> {d['times'][-1]}, missing {100*d['missing_ratio']:.2f}%")
        print("   samples:", {k: len(v.dataset) for k, v in loaders.items()},
              "| batch x", tuple(x.shape), "y", tuple(y.shape))
