"""
Module 2 -- Forecasting models.

Every deep model has the same interface so Module 3 can train any of them:
    forward(x, tod, dow) -> prediction
        x   : (B, 12, N)  normalised speed of the last hour
        tod : (B, 12)     time-of-day slot 0..287
        dow : (B, 12)     day-of-week 0..6
        out : (B, N, 3)   normalised speed 15 / 30 / 60 minutes ahead
"""
from m2_models.lstm import LSTMBaseline
from m2_models.stgcn import STGCN
from m2_models.st_transformer import STTransformer


def build_model(name: str, num_nodes: int, adjacency):
    """Factory: model name (from config.MODEL_NAMES) -> PyTorch model."""
    from m1_data.graph import normalize_for_gcn, normalize_rows
    if name == "lstm":
        return LSTMBaseline()
    if name == "stgcn":
        return STGCN(normalize_for_gcn(adjacency))
    if name == "static_st":
        return STTransformer(num_nodes, normalize_rows(adjacency), adaptive=False)
    if name == "adaptive_st":
        return STTransformer(num_nodes, normalize_rows(adjacency), adaptive=True)
    raise ValueError(f"unknown model {name}")
