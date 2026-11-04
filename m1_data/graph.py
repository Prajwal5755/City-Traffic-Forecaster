"""
m1_data/graph.py  --  STEP 2: turn the road network into a graph.

Nodes  = traffic sensors.
Edges  = how close two sensors are along the road network.

We use the thresholded Gaussian kernel from the STGCN / DCRNN papers:

        W[i, j] = exp( -(dist(i, j) / sigma)^2 )    if that value >= 0.1
        W[i, j] = 0                                  otherwise

  * dist(i, j) is the DRIVING distance from sensor i to sensor j (metres),
    so the graph is directed: i -> j can differ from j -> i.
  * sigma is the standard deviation of all known distances.
  * Close sensors get a weight near 1, far-away sensors get 0 (no edge).
"""
import numpy as np
import pandas as pd


def build_adjacency(distances: pd.DataFrame, sensor_ids: list, threshold: float = 0.1):
    """
    distances  : DataFrame with columns [from, to, cost]  (cost = metres)
    sensor_ids : list of sensor ids, in the same order as the speed columns
    returns    : (N, N) float32 weighted adjacency matrix W
    """
    n = len(sensor_ids)
    position = {str(s): i for i, s in enumerate(sensor_ids)}   # id -> row number

    # 1) fill a distance matrix; "inf" means "no road path recorded"
    dist = np.full((n, n), np.inf, dtype=np.float32)
    for frm, to, cost in distances.itertuples(index=False):
        i, j = position.get(str(frm)), position.get(str(to))
        if i is not None and j is not None:
            dist[i, j] = cost

    # 2) Gaussian kernel: turns distance into similarity in (0, 1]
    sigma = dist[~np.isinf(dist)].std()
    W = np.exp(-np.square(dist / sigma))       # exp(-inf) = 0 automatically

    # 3) sparsify: drop weak edges so each sensor only talks to its neighbours
    W[W < threshold] = 0.0
    return W.astype(np.float32)


def normalize_for_gcn(W: np.ndarray) -> np.ndarray:
    """
    Symmetric normalisation used by graph convolution (Kipf & Welling, STGCN):
        A_hat = D^(-1/2) (A + I) D^(-1/2)
    We first make the graph undirected (A = max(W, W^T)) and add self-loops
    so every sensor also keeps its own information.
    """
    A = np.maximum(W, W.T) + np.eye(len(W), dtype=np.float32)
    d_inv_sqrt = 1.0 / np.sqrt(A.sum(axis=1))
    return (A * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]).astype(np.float32)


def normalize_rows(W: np.ndarray) -> np.ndarray:
    """
    Row normalisation used by our transformer: each row sums to 1, so
    "neighbour message = weighted AVERAGE of the neighbours' features".
    Self-loops are added so a sensor with no neighbours still sees itself.
    """
    A = W + np.eye(len(W), dtype=np.float32)
    return (A / A.sum(axis=1, keepdims=True)).astype(np.float32)


def graph_summary(W: np.ndarray) -> dict:
    """Small numbers to quote on a slide."""
    n = len(W)
    edges = int((W > 0).sum() - (np.diag(W) > 0).sum())   # ignore self-loops
    return {
        "nodes": n,
        "edges": edges,
        "avg_neighbours": round(edges / n, 2),
        "density_percent": round(100 * edges / (n * (n - 1)), 2),
    }
