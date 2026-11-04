"""
m3_training/evaluate.py  --  test every trained model on the unseen 20% (the future).

Run:   python -m m3_training.evaluate --dataset metr-la

Writes:
  results/<dataset>_results.json   all metrics, per model and horizon
  results/<dataset>_results.csv    the same as a table (goes on the results slide)
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from config import CHECKPOINT_DIR, MODEL_LABELS, MODEL_NAMES, RESULTS_DIR
from m1_data.dataset import TrafficWindows, get_loaders, prepare_dataset
from m2_models import build_model
from m2_models.historical_average import HistoricalAverage
from m3_training.metrics import all_metrics


def load_trained_model(name: str, data: dict):
    """Rebuild the architecture and load the best saved weights."""
    model = build_model(name, data["num_nodes"], data["adjacency"])
    ckpt = torch.load(CHECKPOINT_DIR / f"{data['name']}_{name}.pt", map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


@torch.no_grad()
def predict(model, loader, scaler):
    """Run the model over a whole split -> (predictions, targets), both in mph."""
    preds, trues = [], []
    for x, y, tod, dow in loader:
        preds.append(scaler.inverse(model(x, tod, dow)))
        trues.append(y)
    return torch.cat(preds).numpy(), torch.cat(trues).numpy()


def evaluate_all(dataset: str, data: dict = None) -> dict:
    data = data or prepare_dataset(dataset)
    test_loader = get_loaders(data)["test"]
    test_starts = TrafficWindows(data, "test").starts
    results = {}

    for name in MODEL_NAMES:
        if name == "ha":
            ha = HistoricalAverage().fit(data)
            pred = ha.predict(data, test_starts)
            true = np.stack([data["speed_raw"][test_starts + 12 + h - 1] for h in (3, 6, 12)], -1)
        else:
            if not (CHECKPOINT_DIR / f"{dataset}_{name}.pt").exists():
                print(f"  [skip] {name}: not trained yet")
                continue
            pred, true = predict(load_trained_model(name, data), test_loader, data["scaler"])
        results[name] = all_metrics(pred, true)
        r = results[name]
        print(f"  {MODEL_LABELS[name]:32s} " + "  ".join(
            f"{h}: MAE {r[h]['MAE']:.2f} RMSE {r[h]['RMSE']:.2f}" for h in r))

    with open(RESULTS_DIR / f"{dataset}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    table = pd.DataFrame({
        MODEL_LABELS[m]: {f"{h} {k}": round(v, 3) for h, d in r.items() for k, v in d.items()}
        for m, r in results.items()}).T
    table.to_csv(RESULTS_DIR / f"{dataset}_results.csv")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="metr-la", choices=["metr-la", "pems-bay"])
    args = parser.parse_args()
    evaluate_all(args.dataset)
