"""
m2_models/historical_average.py  --  the simplest possible forecaster (no learning).

Prediction for sensor n at a future time t =
    average speed of sensor n, in the TRAINING data, at the same
    time-of-day and day-of-week as t.
("What is traffic usually like here on a Monday at 8:30?")

If a deep model cannot beat this, it has not learned anything useful.
"""
import numpy as np

from config import HORIZONS, IN_LEN


class HistoricalAverage:
    def fit(self, data: dict):
        start, end = data["splits"]["train"]
        speed = data["speed_raw"][start:end]
        slot = data["dow"][start:end] * 288 + data["tod"][start:end]   # 0..2015 (7 days x 288)
        N = speed.shape[1]
        sums = np.zeros((7 * 288, N))
        counts = np.zeros((7 * 288, N))
        valid = speed > 0                       # ignore missing readings
        np.add.at(sums, slot, np.where(valid, speed, 0))
        np.add.at(counts, slot, valid)
        overall = speed[valid].mean()
        self.table = np.where(counts > 0, sums / np.maximum(counts, 1), overall)  # (2016, N)
        return self

    def predict(self, data: dict, starts: np.ndarray) -> np.ndarray:
        """starts = window start indices  ->  predictions (num_windows, N, 3) in mph."""
        slot = data["dow"] * 288 + data["tod"]
        preds = [self.table[slot[starts + IN_LEN + h - 1]] for h in HORIZONS]   # each (W, N)
        return np.stack(preds, axis=-1)
