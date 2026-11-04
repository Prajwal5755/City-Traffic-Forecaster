"""
m1_data/explore.py  --  STEP 4: look at the data before modelling (figures for the slides).

Run:  python -m m1_data.explore

Creates in figures/:
  m1_daily_pattern_<ds>.png   average speed by time of day, weekday vs weekend
  m1_sensor_map_<ds>.png      where the sensors are (coloured by average speed)
  m1_adjacency_<ds>.png       the graph we built (adjacency matrix heat-map)
  m1_correlation_<ds>.png     neighbours really do move together (evidence for the graph)
and results/m1_data_summary.json with the numbers quoted on the slides.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import DATASET, FIGURES_DIR, RESULTS_DIR
from m1_data.dataset import prepare_dataset
from m1_data.graph import graph_summary


def daily_pattern(d):
    speed = np.where(d["speed_raw"] > 0, d["speed_raw"], np.nan)
    weekend = d["dow"] >= 5
    hours = np.arange(288) / 12
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for mask, label, colour in [(~weekend, "Weekday", "#D9480F"), (weekend, "Weekend", "#1C7C54")]:
        avg = [np.nanmean(speed[mask & (d["tod"] == s)]) for s in range(288)]
        ax.plot(hours, avg, label=label, color=colour, lw=2)
    ax.set(xlabel="Hour of day", ylabel="Average speed (mph)",
           title=f"{d['name'].upper()}: average speed through the day")
    ax.set_xticks(range(0, 25, 3))
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"m1_daily_pattern_{d['name']}.png", dpi=160)
    plt.close(fig)


def sensor_map(d):
    loc = d["locations"]
    speed = np.where(d["speed_raw"] > 0, d["speed_raw"], np.nan)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sc = ax.scatter(loc.longitude, loc.latitude, c=np.nanmean(speed, axis=0),
                    cmap="RdYlGn", s=22, edgecolor="k", linewidth=0.3)
    plt.colorbar(sc, label="Average speed (mph)")
    ax.set(xlabel="Longitude", ylabel="Latitude", title=f"{d['name'].upper()}: {d['num_nodes']} sensors")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"m1_sensor_map_{d['name']}.png", dpi=160)
    plt.close(fig)


def adjacency_plot(d):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(d["adjacency"], cmap="Oranges")
    plt.colorbar(im, label="Edge weight")
    ax.set(title="Adjacency matrix W (Gaussian kernel)", xlabel="Sensor j", ylabel="Sensor i")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"m1_adjacency_{d['name']}.png", dpi=160)
    plt.close(fig)


def neighbour_correlation(d):
    """Are connected sensors more correlated than random pairs? (Justifies using a graph.)"""
    start, end = d["splits"]["train"]
    corr = np.corrcoef(d["speed_input"][start:end].T)
    W = d["adjacency"]
    off = ~np.eye(len(W), dtype=bool)
    connected = corr[(W > 0) & off]
    not_connected = corr[(W == 0) & off]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.linspace(-0.2, 1, 40)
    ax.hist(not_connected, bins=bins, density=True, alpha=0.6, label="Not connected", color="#868E96")
    ax.hist(connected, bins=bins, density=True, alpha=0.7, label="Connected in graph", color="#D9480F")
    ax.set(xlabel="Correlation of speed", ylabel="Density",
           title="Connected sensors move together")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"m1_correlation_{d['name']}.png", dpi=160)
    plt.close(fig)
    return float(connected.mean()), float(not_connected.mean())


if __name__ == "__main__":
    summary = {}
    for name in [DATASET]:
        d = prepare_dataset(name)
        daily_pattern(d)
        sensor_map(d)
        adjacency_plot(d)
        c_conn, c_not = neighbour_correlation(d)
        summary[name] = {
            "sensors": d["num_nodes"], "time_steps": d["num_steps"],
            "start": str(d["times"][0]), "end": str(d["times"][-1]),
            "days": round(d["num_steps"] / 288, 1),
            "missing_percent": round(100 * d["missing_ratio"], 2),
            "mean_speed": round(d["scaler"].mean, 2), "std_speed": round(d["scaler"].std, 2),
            "split_steps": {k: v[1] - v[0] for k, v in d["splits"].items()},
            "graph": graph_summary(d["adjacency"]),
            "corr_connected": round(c_conn, 3), "corr_not_connected": round(c_not, 3),
        }
        print(name, json.dumps(summary[name], indent=2))
    with open(RESULTS_DIR / "m1_data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
