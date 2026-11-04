"""
m3_training/train.py  --  train one model on one dataset.

Run:   python -m m3_training.train --dataset metr-la --model adaptive_st
       (models: lstm, stgcn, static_st, adaptive_st)

Training recipe (identical for every model, so the comparison is fair):
  * loss       : masked MAE in mph (the metric we report)
  * optimiser  : Adam, learning rate 1e-3, halved when validation stops improving
  * grad clip  : 5.0   (keeps attention/LSTM training stable)
  * early stop : keep the epoch with the best VALIDATION MAE, stop after 5 bad epochs
  * epoch      : 100 random training batches (see config.BATCHES_PER_EPOCH); validation
                 always uses the FULL validation set
"""
import argparse
import json
import time

import numpy as np
import torch

from config import (BATCHES_PER_EPOCH, CHECKPOINT_DIR, LEARNING_RATE, MAX_EPOCHS, PATIENCE,
                    RESULTS_DIR, SEED)
from m1_data.dataset import get_loaders, prepare_dataset
from m2_models import build_model
from m3_training.metrics import masked_mae

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_epoch(model, loader, scaler, optimiser=None, max_batches=None):
    """One pass over the data. If an optimiser is given we train, otherwise we only evaluate."""
    training = optimiser is not None
    model.train(training)
    total, batches = 0.0, 0
    with torch.set_grad_enabled(training):
        for x, y, tod, dow in loader:
            x, y, tod, dow = x.to(DEVICE), y.to(DEVICE), tod.to(DEVICE), dow.to(DEVICE)
            pred = scaler.inverse(model(x, tod, dow))      # back to mph
            loss = masked_mae(pred, y)
            if training:
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimiser.step()
            total += loss.item()
            batches += 1
            if max_batches and batches >= max_batches:
                break
    return total / batches


def train(dataset: str, model_name: str, epochs: int = MAX_EPOCHS,
          max_batches: int = BATCHES_PER_EPOCH, data: dict = None):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    data = data or prepare_dataset(dataset)
    loaders = get_loaders(data)
    scaler = data["scaler"]

    model = build_model(model_name, data["num_nodes"], data["adjacency"]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=2)
    print(f"\n=== {model_name} on {dataset}: {n_params:,} parameters, device={DEVICE} ===")

    ckpt_path = CHECKPOINT_DIR / f"{dataset}_{model_name}.pt"
    history, best_val, bad_epochs = [], float("inf"), 0
    start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_mae = run_epoch(model, loaders["train"], scaler, optimiser, max_batches)
        val_mae = run_epoch(model, loaders["val"], scaler)
        scheduler.step(val_mae)
        row = {"epoch": epoch, "train_mae": train_mae, "val_mae": val_mae,
               "seconds": time.time() - t0}
        if model_name == "adaptive_st":
            row["gate"] = model.graph.gate().item()      # how much the learned graph is trusted
        history.append(row)
        print(f"epoch {epoch:2d} | train MAE {train_mae:.3f} | val MAE {val_mae:.3f} | "
              f"{row['seconds']:.0f}s" + (f" | gate {row['gate']:.3f}" if "gate" in row else ""))

        if val_mae < best_val:                 # new best model -> save it
            best_val, bad_epochs = val_mae, 0
            torch.save({"model": model_name, "dataset": dataset,
                        "state_dict": model.state_dict()}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"early stopping (no improvement for {PATIENCE} epochs)")
                break

    summary = {"model": model_name, "dataset": dataset, "parameters": n_params,
               "best_val_mae": best_val, "epochs_run": len(history),
               "train_seconds": time.time() - start, "history": history}
    with open(RESULTS_DIR / f"{dataset}_{model_name}_history.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved best model to {ckpt_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="metr-la", choices=["metr-la", "pems-bay"])
    parser.add_argument("--model", default="adaptive_st",
                        choices=["lstm", "stgcn", "static_st", "adaptive_st"])
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--max-batches", type=int, default=BATCHES_PER_EPOCH,
                        help="random training batches per epoch (0 = the whole training set)")
    args = parser.parse_args()
    train(args.dataset, args.model, args.epochs, args.max_batches or None)
