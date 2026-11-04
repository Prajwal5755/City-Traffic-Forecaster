"""
model.py
--------
Two models for city-wide traffic speed forecasting:

1. AdaptiveGraphSTTransformer  -- our proposed architecture:
      input+embeddings -> adaptive graph construction -> L x
      [spatial graph-attention sub-layer -> temporal self-attention
      sub-layer -> feed-forward sub-layer] -> multi-horizon head

2. LSTMBaseline -- the standard sequence-model baseline used for comparison.

Shapes follow the convention:
    x        : (B, T_in, N, F)   raw historical features per sensor
    speed    : (B, T_in, N)      the target-relevant channel (traffic speed)
    static_adj: (N, N)           precomputed distance-based adjacency
    output   : (B, N, H)         H = number of forecast horizons (e.g. 3 -> 15/30/60 min)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. Adaptive graph construction module
# --------------------------------------------------------------------------- #
class AdaptiveAdjacency(nn.Module):
    """
    Learns a directed NxN adjacency matrix from trainable node embeddings,
    then fuses it with a fixed, distance-based adjacency via a learnable
    gate alpha. This is the module that directly answers the "static graph"
    limitation identified in the literature review (STGCN/DCRNN/ASTGCN all
    use a graph that cannot adapt to changing traffic correlations).
    """

    def __init__(self, num_nodes: int, embed_dim: int = 16):
        super().__init__()
        self.e1 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.01)
        self.e2 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.01)
        # learnable scalar gate in [0, 1] deciding how much to trust the
        # learned graph vs. the physical distance graph
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, static_adj: torch.Tensor) -> torch.Tensor:
        # (N, N) learned affinity -> row-normalised with softmax
        adaptive_adj = F.softmax(F.relu(self.e1 @ self.e2.T), dim=-1)
        gate = torch.sigmoid(self.alpha)
        combined = gate * adaptive_adj + (1 - gate) * static_adj
        return combined  # (N, N)


# --------------------------------------------------------------------------- #
# 2. Spatial sub-layer: graph convolution driven by the adaptive adjacency
# --------------------------------------------------------------------------- #
class SpatialGraphSubLayer(nn.Module):
    """
    Graph-convolution sub-layer: H' = A_combined @ H @ W (+ self-loop),
    applied independently at every timestep. Residual + LayerNorm wrap it,
    matching standard Transformer sub-layer conventions.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.lin_neighbor = nn.Linear(hidden_dim, hidden_dim)
        self.lin_self = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h: (B, T, N, D), adj: (N, N)
        B, T, N, D = h.shape
        neighbor_msg = torch.einsum("nm,btmd->btnd", adj, h)
        out = self.lin_neighbor(neighbor_msg) + self.lin_self(h)
        out = F.relu(out)
        out = self.dropout(out)
        return self.norm(h + out)  # residual + LN


# --------------------------------------------------------------------------- #
# 3. Temporal sub-layer: self-attention across the time dimension
# --------------------------------------------------------------------------- #
class TemporalAttentionSubLayer(nn.Module):
    """
    Standard multi-head self-attention applied across time, independently
    per node (nodes are folded into the batch dimension for this step).
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, N, D) -> fold nodes into batch: (B*N, T, D)
        B, T, N, D = h.shape
        h_flat = h.permute(0, 2, 1, 3).reshape(B * N, T, D)
        attn_out, _ = self.attn(h_flat, h_flat, h_flat, need_weights=False)
        out = self.norm(h_flat + self.dropout(attn_out))
        return out.reshape(B, N, T, D).permute(0, 2, 1, 3)  # back to (B, T, N, D)


# --------------------------------------------------------------------------- #
# 4. Feed-forward sub-layer
# --------------------------------------------------------------------------- #
class FeedForwardSubLayer(nn.Module):
    def __init__(self, hidden_dim: int, ff_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.norm(h + self.dropout(self.net(h)))


# --------------------------------------------------------------------------- #
# 5. One full spatio-temporal encoder block (spatial -> temporal -> FFN)
# --------------------------------------------------------------------------- #
class STEncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.spatial = SpatialGraphSubLayer(hidden_dim, dropout)
        self.temporal = TemporalAttentionSubLayer(hidden_dim, num_heads, dropout)
        self.ffn = FeedForwardSubLayer(hidden_dim, hidden_dim * 4, dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.spatial(h, adj)
        h = self.temporal(h)
        h = self.ffn(h)
        return h


# --------------------------------------------------------------------------- #
# 6. Input embedding layer: speed + time-of-day + day-of-week + node identity
# --------------------------------------------------------------------------- #
class InputEmbedding(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int,
                 num_tod_slots: int = 288, num_dow_slots: int = 7):
        super().__init__()
        self.value_proj = nn.Linear(1, hidden_dim)
        self.node_embed = nn.Embedding(num_nodes, hidden_dim)
        self.tod_embed = nn.Embedding(num_tod_slots, hidden_dim)  # 5-min slots/day
        self.dow_embed = nn.Embedding(num_dow_slots, hidden_dim)

    def forward(self, speed: torch.Tensor, tod_idx: torch.Tensor,
                dow_idx: torch.Tensor) -> torch.Tensor:
        # speed: (B, T, N), tod_idx/dow_idx: (B, T)
        B, T, N = speed.shape
        val = self.value_proj(speed.unsqueeze(-1))                    # (B,T,N,D)
        node = self.node_embed.weight.view(1, 1, N, -1)                # (1,1,N,D)
        tod = self.tod_embed(tod_idx).unsqueeze(2)                     # (B,T,1,D)
        dow = self.dow_embed(dow_idx).unsqueeze(2)                     # (B,T,1,D)
        return val + node + tod + dow


# --------------------------------------------------------------------------- #
# 7. Full model
# --------------------------------------------------------------------------- #
class AdaptiveGraphSTTransformer(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int = 32, num_layers: int = 2,
                 num_heads: int = 4, num_horizons: int = 3, dropout: float = 0.1):
        super().__init__()
        self.embedding = InputEmbedding(num_nodes, hidden_dim)
        self.graph_module = AdaptiveAdjacency(num_nodes, embed_dim=16)
        self.blocks = nn.ModuleList(
            [STEncoderBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_dim, num_horizons)

    def forward(self, speed, tod_idx, dow_idx, static_adj):
        h = self.embedding(speed, tod_idx, dow_idx)          # (B,T,N,D)
        adj = self.graph_module(static_adj)                  # (N,N)
        for block in self.blocks:
            h = block(h, adj)
        last_step = h[:, -1, :, :]                           # (B,N,D) most recent timestep
        return self.head(last_step)                          # (B,N,H)


# --------------------------------------------------------------------------- #
# 8. LSTM baseline (the model this project benchmarks against)
# --------------------------------------------------------------------------- #
class LSTMBaseline(nn.Module):
    """
    Runs an independent LSTM per sensor (nodes folded into batch dimension),
    then projects the final hidden state to the forecast horizons.
    """

    def __init__(self, hidden_dim: int = 32, num_layers: int = 2,
                 num_horizons: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, num_horizons)

    def forward(self, speed, tod_idx=None, dow_idx=None, static_adj=None):
        # speed: (B, T, N) -> fold nodes into batch: (B*N, T, 1)
        B, T, N = speed.shape
        x = speed.permute(0, 2, 1).reshape(B * N, T, 1)
        x = self.input_proj(x)
        out, _ = self.lstm(x)
        last = out[:, -1, :]                 # (B*N, D)
        pred = self.head(last)                # (B*N, H)
        return pred.reshape(B, N, -1)          # (B, N, H)
