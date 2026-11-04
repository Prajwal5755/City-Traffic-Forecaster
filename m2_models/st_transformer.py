"""
m2_models/st_transformer.py  --  OUR MODEL: Adaptive-Graph Spatio-Temporal Transformer.

Pipeline (same picture as the methodology slide):
    1. Input embedding   : speed value + sensor id + time-of-day + day-of-week
    2. Graph             : A = g * A_learned + (1 - g) * A_static      (g = sigmoid(alpha), learned)
    3. Encoder block x L : spatial graph conv -> temporal self-attention -> feed-forward
                           (each with residual connection + LayerNorm)
    4. Prediction head   : last time step -> 3 horizons

adaptive=False gives the "Static-Graph ST Transformer" ablation: identical model,
but it may only use the physical road graph. Comparing the two isolates the
value of the learned graph -- the key experiment of this project.

Tensor layout:  h is (B, T, N, D)  = batch, time, sensors, hidden features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveAdjacency(nn.Module):
    """
    Learns which sensors influence each other, directly from data.
        A_learned = softmax( ReLU( E1 @ E2^T ) )      E1, E2: (N, k) trainable
    Then a learnable gate mixes it with the physical graph.
    """

    def __init__(self, num_nodes: int, embed_dim: int = 10):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(num_nodes, embed_dim))   # "source" embedding
        self.E2 = nn.Parameter(torch.randn(num_nodes, embed_dim))   # "target" embedding
        self.alpha = nn.Parameter(torch.tensor(0.0))                # sigmoid(0) = 0.5 at start

    def learned(self):
        return F.softmax(F.relu(self.E1 @ self.E2.T), dim=1)       # each row sums to 1

    def gate(self):
        return torch.sigmoid(self.alpha)                            # trust in learned graph, 0..1

    def forward(self, A_static):
        g = self.gate()
        return g * self.learned() + (1 - g) * A_static


class SpatialLayer(nn.Module):
    """Graph convolution: every sensor gathers a weighted average of its neighbours."""

    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.lin_neighbour = nn.Linear(d, d)
        self.lin_self = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, A):
        neighbour = torch.einsum("nm,btmd->btnd", A, h)      # A @ h for every batch and time step
        out = F.relu(self.lin_neighbour(neighbour) + self.lin_self(h))
        return self.norm(h + self.drop(out))                 # residual + LayerNorm


class TemporalLayer(nn.Module):
    """Multi-head self-attention along time: each step can look at every other step."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h):
        B, T, N, D = h.shape
        seq = h.permute(0, 2, 1, 3).reshape(B * N, T, D)    # one sequence per sensor
        out, _ = self.attn(seq, seq, seq, need_weights=False)
        seq = self.norm(seq + self.drop(out))
        return seq.reshape(B, N, T, D).permute(0, 2, 1, 3)


class FeedForward(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h):
        return self.norm(h + self.net(h))


class EncoderBlock(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.spatial = SpatialLayer(d, dropout)
        self.temporal = TemporalLayer(d, heads, dropout)
        self.ffn = FeedForward(d, dropout)

    def forward(self, h, A):
        return self.ffn(self.temporal(self.spatial(h, A)))


class STTransformer(nn.Module):
    def __init__(self, num_nodes: int, A_static, adaptive: bool = True,
                 d: int = 32, layers: int = 2, heads: int = 4,
                 num_horizons: int = 3, dropout: float = 0.0):
        # d=32 and no dropout keep training fast enough for a laptop CPU;
        # early stopping on the validation set is our guard against over-fitting.
        super().__init__()
        self.register_buffer("A_static", torch.as_tensor(A_static))
        self.adaptive = adaptive
        # 1) embeddings
        self.value_proj = nn.Linear(1, d)
        self.node_emb = nn.Embedding(num_nodes, d)   # "who am I" (each sensor behaves differently)
        self.tod_emb = nn.Embedding(288, d)          # time of day (rush hour vs night)
        self.dow_emb = nn.Embedding(7, d)            # weekday vs weekend
        # 2) graph
        self.graph = AdaptiveAdjacency(num_nodes) if adaptive else None
        # 3) encoder
        self.blocks = nn.ModuleList([EncoderBlock(d, heads, dropout) for _ in range(layers)])
        # 4) head
        self.head = nn.Linear(d, num_horizons)

    def adjacency(self):
        """The graph the model actually uses (handy for the heat-map plot)."""
        return self.graph(self.A_static) if self.adaptive else self.A_static

    def forward(self, x, tod, dow):
        B, T, N = x.shape
        h = (self.value_proj(x.unsqueeze(-1))                 # (B, T, N, d)
             + self.node_emb.weight[None, None]               # (1, 1, N, d)
             + self.tod_emb(tod)[:, :, None]                  # (B, T, 1, d)
             + self.dow_emb(dow)[:, :, None])                 # (B, T, 1, d)
        A = self.adjacency()
        for block in self.blocks:
            h = block(h, A)
        return self.head(h[:, -1])                            # last time step -> (B, N, 3)
