"""
m3_training/metrics.py  --  how we measure error (all in mph, on real speeds).

"Masked" = readings of 0 mph are sensor failures, so they are left out of
both the loss and the metrics (we cannot be right or wrong about a broken sensor).

  MAE  = mean |pred - true|            average error in mph       (easy to explain)
  RMSE = sqrt(mean (pred - true)^2)    punishes big misses more   (e.g. missed jams)
  MAPE = mean |pred - true| / true     error as a percentage
"""
import torch


def masked_mae(pred, true):
    mask = (true > 0).float()
    return (torch.abs(pred - true) * mask).sum() / mask.sum().clamp(min=1)


def masked_rmse(pred, true):
    mask = (true > 0).float()
    return torch.sqrt(((pred - true) ** 2 * mask).sum() / mask.sum().clamp(min=1))


def masked_mape(pred, true):
    mask = (true > 0).float()
    pct = torch.abs(pred - true) / true.clamp(min=1e-3)
    return 100 * (pct * mask).sum() / mask.sum().clamp(min=1)


def all_metrics(pred, true) -> dict:
    """pred, true: (..., 3) -> metrics per horizon. Works on numpy or torch input."""
    pred, true = torch.as_tensor(pred, dtype=torch.float32), torch.as_tensor(true, dtype=torch.float32)
    out = {}
    for i, name in enumerate(("15 min", "30 min", "60 min")):
        p, t = pred[..., i], true[..., i]
        out[name] = {"MAE": masked_mae(p, t).item(),
                     "RMSE": masked_rmse(p, t).item(),
                     "MAPE": masked_mape(p, t).item()}
    return out
