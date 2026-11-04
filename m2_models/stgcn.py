"""
m2_models/stgcn.py  --  STGCN (Yu, Yin & Zhu, IJCAI 2018), the model named in our proposal.

Idea: no RNN at all. Alternate
    TIME  : gated 1-D convolution along the time axis       ("what is the trend?")
    SPACE : graph convolution over the road graph            ("what are my neighbours doing?")

One ST-Conv block  =  TemporalConv -> GraphConv -> TemporalConv -> LayerNorm
Model              =  ST-Conv block x 2  ->  output layer  ->  3 horizons

Tensor layout inside the model:  (B, C, T, N)
    B = batch, C = channels (features), T = time steps, N = sensors
Note: for simplicity we use the first-order graph convolution
(A_hat @ X @ W) instead of the paper's Chebyshev polynomial version.
"""
import torch
import torch.nn as nn


class TemporalGatedConv(nn.Module):
    """
    1-D convolution over time with a GLU gate:
        out = (P) * sigmoid(Q)      where [P, Q] = Conv(x)
    The sigmoid gate decides how much of each feature passes through.
    Kernel size 3 shrinks the time axis by 2 (no padding).
    """

    def __init__(self, c_in: int, c_out: int, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(c_in, 2 * c_out, kernel_size=(kernel, 1))   # (k x 1): time only
        self.residual = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.kernel = kernel

    def forward(self, x):                          # x: (B, C_in, T, N)
        p, q = self.conv(x).chunk(2, dim=1)        # each (B, C_out, T-k+1, N)
        res = self.residual(x[:, :, self.kernel - 1:, :])   # match the shorter time axis
        return (p + res) * torch.sigmoid(q)


class GraphConv(nn.Module):
    """
    Spatial step:  H' = ReLU( A_hat @ H @ W )
    Each sensor's new feature = weighted average of its neighbours' features,
    then a learned linear transform W.
    """

    def __init__(self, channels: int, a_hat):
        super().__init__()
        self.register_buffer("A", torch.as_tensor(a_hat))   # (N, N), fixed road graph
        self.W = nn.Linear(channels, channels)

    def forward(self, x):                          # x: (B, C, T, N)
        x = torch.einsum("nm,bctm->bctn", self.A, x)        # mix neighbours
        x = self.W(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # linear on channels
        return torch.relu(x)


class STConvBlock(nn.Module):
    """The 'sandwich': time -> space -> time."""

    def __init__(self, c_in: int, c_hidden: int, c_out: int, num_nodes: int, a_hat, dropout=0.1):
        super().__init__()
        self.t1 = TemporalGatedConv(c_in, c_hidden)
        self.graph = GraphConv(c_hidden, a_hat)
        self.t2 = TemporalGatedConv(c_hidden, c_out)
        self.norm = nn.LayerNorm([num_nodes, c_out])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.t2(self.graph(self.t1(x)))        # time shrinks 12 -> 10 -> 8 in block 1
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x)


class STGCN(nn.Module):
    def __init__(self, a_hat, in_len: int = 12, hidden: int = 64, num_horizons: int = 3):
        super().__init__()
        n = a_hat.shape[0]
        self.block1 = STConvBlock(1, hidden // 4, hidden, n, a_hat)        # T: 12 -> 8
        self.block2 = STConvBlock(hidden, hidden // 4, hidden, n, a_hat)   # T: 8  -> 4
        remaining = in_len - 8
        # output layer: one temporal conv that squeezes the remaining time steps to 1
        self.out_time = nn.Conv2d(hidden, hidden, kernel_size=(remaining, 1))
        self.out_fc = nn.Linear(hidden, num_horizons)

    def forward(self, x, tod=None, dow=None):
        x = x.unsqueeze(1)                         # (B, 12, N) -> (B, 1, 12, N)
        x = self.block2(self.block1(x))            # (B, 64, 4, N)
        x = torch.relu(self.out_time(x))           # (B, 64, 1, N)
        x = x.squeeze(2).permute(0, 2, 1)          # (B, N, 64)
        return self.out_fc(x)                      # (B, N, 3)
