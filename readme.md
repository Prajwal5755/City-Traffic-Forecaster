# Baseline code — adaptive-graph spatio-temporal transformer vs. LSTM

This is a runnable, end-to-end sanity check of the architecture proposed for
the mid-review: an adaptive-graph spatio-temporal transformer, benchmarked
against the standard LSTM baseline.

## Files
- `model.py` — both models: `LSTMBaseline` and `AdaptiveGraphSTTransformer`
  (adaptive adjacency + stacked spatial/temporal/feed-forward blocks).
- `data.py` — synthetic data generator (clustered sensors with a shared
  diurnal pattern, so a graph-aware model has something real to exploit)
  plus the windowing function you'll reuse for real data.
- `train.py` — trains both models, evaluates MAE/RMSE at 15/30/60-minute
  horizons, and saves the learned adjacency matrix for inspection.

## Run it
```bash
pip install torch --break-system-packages   # if not already installed
python train.py
```

## What this proves right now
- The full pipeline (embeddings → adaptive graph → spatial attention →
  temporal attention → feed-forward → multi-horizon head) runs without
  shape errors and trains — loss decreases every epoch for both models.
- The learned gate `alpha` moves away from 0.5 during training, meaning the
  model is actually using the *learned* graph rather than only the static
  one (printed at the end of the run, and saved in `learned_adjacency.npy`
  vs `static_adjacency.npy` for a side-by-side heatmap if you want one for
  the slides).

**Important — this is a synthetic sanity check, not a real result.** The
synthetic data was deliberately built with strong sensor clustering, so the
large gap between the two models here is partly an artifact of that design
(the LSTM has no way to see other sensors at all), not evidence that the
real architecture will beat LSTM by this margin on METR-LA/PeMS-BAY. Treat
this purely as "the code works, the model learns" — not as a preliminary
result to put on the accuracy slide.

## Swapping in the real datasets
METR-LA and PeMS-BAY are usually distributed as:
- a `(T_total, N)` speed matrix (pandas DataFrame with a DatetimeIndex),
  commonly shipped as `metr-la.h5` / `pems-bay.h5`
- a sensor distance/graph file used to build the static adjacency

To switch over:
1. Replace `make_synthetic_dataset()` in `data.py` with a loader that reads
   the `.h5` file into a numpy array `speed_matrix` of shape `(T_total, N)`.
2. Build `static_adj` from the distance file with the standard thresholded
   Gaussian kernel used by STGCN/DCRNN:
   `W[i,j] = exp(-(dist[i,j]**2) / sigma**2)` if `dist[i,j] < threshold` else `0`,
   then row-normalize.
3. Derive `tod_idx` / `dow_idx` from the DataFrame's DatetimeIndex instead
   of the synthetic clock.
4. `make_windows()` and everything in `train.py` is dataset-agnostic and
   does not need to change.
5. Increase `hidden_dim`, `num_layers`, and `EPOCHS` once you're on a GPU —
   the settings here are deliberately small so this runs in seconds on CPU.

## Where these numbers/artifacts feed into the mid-review
- The training curves above are your "implementation status" evidence —
  proof the model builds and trains, ahead of running on the real datasets.
- `learned_adjacency.npy` vs `static_adjacency.npy` gives you a heatmap for
  the "uniqueness/novelty" slide — visually showing the learned graph found
  structure the distance matrix didn't have.
- Once you swap in METR-LA/PeMS-BAY, the same `evaluate()` function reports
  MAE/RMSE at exactly the 15/30/60-minute horizons the guidelines ask for.
