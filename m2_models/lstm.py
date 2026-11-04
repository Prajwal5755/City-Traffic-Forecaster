"""
m2_models/lstm.py  --  baseline 1: an LSTM that looks at TIME only.

Each sensor is treated as an independent time series (no graph at all).
The same LSTM weights are shared by every sensor.
This answers: "how far can we get without spatial information?"
"""
import torch.nn as nn


class LSTMBaseline(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_layers: int = 2, num_horizons: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_horizons)   # last hidden state -> 3 forecasts

    def forward(self, x, tod=None, dow=None):
        B, T, N = x.shape
        # put every sensor in the batch: (B, T, N) -> (B*N, T, 1)
        seq = x.permute(0, 2, 1).reshape(B * N, T, 1)
        out, _ = self.lstm(seq)                 # (B*N, T, hidden)
        pred = self.head(out[:, -1])            # use the final time step -> (B*N, 3)
        return pred.reshape(B, N, -1)           # (B, N, 3)
