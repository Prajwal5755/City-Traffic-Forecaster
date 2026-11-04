"""
m3_training/plots.py  --  turn results into figures for the slides.

Run:  python -m m3_training.plots --dataset metr-la

Creates in figures/:
  m3_training_curves_<ds>.png   validation MAE per epoch for every model
  m3_horizon_mae_<ds>.png       test MAE at 15 / 30 / 60 min, all models
  m3_forecast_example_<ds>.png  one sensor, one day: truth vs predictions
  m2_gate_<ds>.png              how the learned-graph gate moved during training
  m2_graphs_<ds>.png            physical graph vs learned graph (heat-maps)
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import CHECKPOINT_DIR, FIGURES_DIR, IN_LEN, MODEL_LABELS, RESULTS_DIR
from m1_data.dataset import TrafficWindows, prepare_dataset
from m3_training.evaluate import load_trained_model

COLOURS = {"ha": "#ADB5BD", "lstm": "#4C6EF5", "stgcn": "#2B8A3E",
           "static_st": "#F59F00", "adaptive_st": "#D9480F"}
DEEP = ["lstm", "stgcn", "static_st", "adaptive_st"]


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


def training_curves(ds):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for m in DEEP:
        path = RESULTS_DIR / f"{ds}_{m}_history.json"
        if path.exists():
            h = json.loads(path.read_text())["history"]
            ax.plot([r["epoch"] for r in h], [r["val_mae"] for r in h],
                    marker="o", ms=3, label=MODEL_LABELS[m], color=COLOURS[m])
    ax.set(xlabel="Epoch", ylabel="Validation MAE (mph)", title="Training progress")
    ax.legend(frameon=False, fontsize=8)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"m3_training_curves_{ds}.png", dpi=160)
    plt.close(fig)


def horizon_bars(ds):
    res = json.loads((RESULTS_DIR / f"{ds}_results.json").read_text())
    horizons = ["15 min", "30 min", "60 min"]
    models = list(res)
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, m in enumerate(models):
        vals = [res[m][h]["MAE"] for h in horizons]
        xs = np.arange(3) + i * width
        ax.bar(xs, vals, width, label=MODEL_LABELS[m], color=COLOURS[m])
    ax.set_xticks(np.arange(3) + width * (len(models) - 1) / 2, horizons)
    ax.set(ylabel="Test MAE (mph)", title=f"{ds.upper()}: error grows with horizon")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _clean(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"m3_horizon_mae_{ds}.png", dpi=160)
    plt.close(fig)


@torch.no_grad()
def forecast_example(ds, data, sensor=None, horizon_index=2):
    """One test day for one sensor: ground truth vs each model's 60-min-ahead forecast."""
    win = TrafficWindows(data, "test")
    h = (3, 6, 12)[horizon_index]
    # one full weekday (midnight to midnight) of forecast target times, from the test period
    target_idx = win.starts + IN_LEN + h - 1
    t = data["times"][target_idx]
    first = int(np.where((t.hour == 0) & (t.minute == 0) & (t.dayofweek == 1))[0][0])   # a Tuesday
    day = slice(first, first + 288)
    starts = win.starts[day]
    if sensor is None:                                # pick the sensor with the biggest daily dip
        raw = data["speed_raw"][starts + IN_LEN]
        has_gaps = (raw == 0).mean(0) > 0.01          # skip sensors with missing readings that day
        dip = np.median(raw, 0) - np.percentile(raw, 5, axis=0)
        sensor = int(np.argmax(np.where(has_gaps, -1, dip)))
    x = torch.stack([win[k][0] for k in range(day.start, day.stop)])
    tod = torch.stack([win[k][2] for k in range(day.start, day.stop)])
    dow = torch.stack([win[k][3] for k in range(day.start, day.stop)])
    target_times = data["times"][starts + IN_LEN + h - 1]
    truth = data["speed_raw"][starts + 12 + h - 1, sensor]

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(target_times, np.where(truth > 0, truth, np.nan), color="black", lw=2, label="Ground truth")
    for m in ["lstm", "adaptive_st"]:
        if not (CHECKPOINT_DIR / f"{ds}_{m}.pt").exists():
            continue
        model = load_trained_model(m, data)
        pred = data["scaler"].inverse(model(x, tod, dow))[:, sensor, horizon_index].numpy()
        ax.plot(target_times, pred, color=COLOURS[m], lw=1.5, label=MODEL_LABELS[m])
    ax.set(ylabel="Speed (mph)", title=f"Sensor {data['sensor_ids'][sensor]}: "
           f"{(3, 6, 12)[horizon_index] * 5}-minute-ahead forecast over one day")
    ax.legend(frameon=False, fontsize=8)
    fig.autofmt_xdate()
    _clean(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"m3_forecast_example_{ds}.png", dpi=160)
    plt.close(fig)


@torch.no_grad()
def graph_figures(ds, data):
    path = RESULTS_DIR / f"{ds}_adaptive_st_history.json"
    if not path.exists():
        print("adaptive_st not trained yet: skipping graph figures")
        return
    h = json.loads(path.read_text())["history"]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot([r["epoch"] for r in h], [r["gate"] for r in h], marker="o", color=COLOURS["adaptive_st"])
    ax.axhline(0.5, ls="--", color="grey", lw=1)
    ax.set(xlabel="Epoch", ylabel="gate g", ylim=(0, 1), title="Trust in the learned graph")
    _clean(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"m2_gate_{ds}.png", dpi=160)
    plt.close(fig)

    model = load_trained_model("adaptive_st", data)
    learned = model.graph.learned().numpy()
    static = model.A_static.numpy()
    n = 50                                              # zoom on the first 50 sensors
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, mat, title in [(axes[0], static, "Physical road graph"),
                           (axes[1], learned, "Learned graph")]:
        ax.imshow(mat[:n, :n], cmap="Oranges", vmax=np.percentile(mat[:n, :n], 99))
        ax.set(title=title, xlabel="Sensor j", ylabel="Sensor i")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"m2_graphs_{ds}.png", dpi=160)
    plt.close(fig)
    np.save(RESULTS_DIR / f"{ds}_learned_adjacency.npy", learned)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="metr-la", choices=["metr-la", "pems-bay"])
    args = parser.parse_args()
    d = prepare_dataset(args.dataset)
    for name in [f.name for f in FIGURES_DIR.glob(f"m[23]_*{args.dataset}.png")]:
        (FIGURES_DIR / name).unlink()          # never leave figures from an older run
    training_curves(args.dataset)
    horizon_bars(args.dataset)
    forecast_example(args.dataset, d)
    graph_figures(args.dataset, d)
    print("figures saved to", FIGURES_DIR)
