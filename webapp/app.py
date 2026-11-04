"""
webapp/app.py  --  interactive traffic forecasting dashboard (Streamlit).

Run from the project folder:
    streamlit run webapp/app.py

What it shows (all predictions are computed LIVE by the trained models):
  1. Forecast map      pick a moment from the test period -> predicted speed of every
                       sensor 15/30/60 min later, next to what really happened
                       (drawn on the road graph, so it works without internet)
  2. Sensor explorer   pick one sensor -> last hour + every model's forecast vs truth
  3. Model comparison  test-set MAE / RMSE / MAPE table and chart
  4. Learned graph     how much the model trusts the learned graph vs the road map
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))   # so we can import the modules
from config import CHECKPOINT_DIR, HORIZONS, IN_LEN, MODEL_LABELS, RESULTS_DIR  # noqa: E402
from m1_data.dataset import TrafficWindows, prepare_dataset  # noqa: E402
from m3_training.evaluate import load_trained_model  # noqa: E402

st.set_page_config(page_title="City Traffic Forecaster", page_icon="🚦", layout="wide")
DEEP_MODELS = ["adaptive_st", "static_st", "stgcn", "lstm"]


# ------------------------------------------------------------------ cached loaders
@st.cache_resource(show_spinner="Loading dataset...")
def get_data(name):
    return prepare_dataset(name)


@st.cache_resource(show_spinner="Loading model...")
def get_model(name, dataset):
    return load_trained_model(name, get_data(dataset))


@torch.no_grad()
def forecast(model_name, dataset, start):
    """Run one model on the hour starting at `start` -> (N, 3) speeds in mph."""
    d = get_data(dataset)
    win = TrafficWindows(d, "test")
    k = int(np.where(win.starts == start)[0][0])
    x, _, tod, dow = win[k]
    pred = get_model(model_name, dataset)(x[None], tod[None], dow[None])[0]
    return d["scaler"].inverse(pred).numpy()


def show(fig, where=st):
    """Full-width chart on both new and older Streamlit versions."""
    try:
        where.plotly_chart(fig, width="stretch")
    except Exception:
        where.plotly_chart(fig, use_container_width=True)


def speed_colour_scale():
    return [[0, "#C92A2A"], [0.45, "#F59F00"], [0.7, "#FAB005"], [1, "#2B8A3E"]]


def network_map(loc, values, title, W, street_map=False):
    """Sensors coloured by speed, drawn on top of the road graph from Module 1.
    Works offline. street_map=True adds an OpenStreetMap background (needs internet)."""
    hover = [f"sensor {sid}<br>{v:.1f} mph" if v == v else f"sensor {sid}<br>no reading"
             for sid, v in zip(loc.sensor_id, values)]
    if street_map:
        fig = px.scatter_map(loc.assign(speed=values), lat="latitude", lon="longitude", color="speed",
                             range_color=(15, 70), color_continuous_scale=speed_colour_scale(),
                             hover_name="sensor_id", zoom=9.3, height=500, map_style="open-street-map")
        fig.update_traces(marker={"size": 10})
    else:
        fig = go.Figure()
        i, j = np.nonzero(np.triu(np.maximum(W, W.T), k=1))          # road-graph edges
        ex = np.column_stack([loc.longitude.values[i], loc.longitude.values[j], np.full(len(i), np.nan)]).ravel()
        ey = np.column_stack([loc.latitude.values[i], loc.latitude.values[j], np.full(len(i), np.nan)]).ravel()
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#CED4DA", width=1),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=loc.longitude, y=loc.latitude, mode="markers", text=hover, hoverinfo="text", showlegend=False,
            marker=dict(size=11, color=values, cmin=15, cmax=70, colorscale=speed_colour_scale(),
                        line=dict(width=0.5, color="#343A40"), colorbar=dict(title="mph"))))
        lat0 = np.deg2rad(loc.latitude.mean())
        fig.update_yaxes(scaleanchor="x", scaleratio=1 / np.cos(lat0), visible=False)
        fig.update_xaxes(visible=False)
        fig.update_layout(height=500, plot_bgcolor="white")
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), title=title)
    return fig


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🚦 Traffic Forecaster")
available = [ds for ds in ("metr-la", "pems-bay")
             if any((CHECKPOINT_DIR / f"{ds}_{m}.pt").exists() for m in DEEP_MODELS)]
if not available:
    st.error("No trained models found. Run `python run_all.py` first.")
    st.stop()
dataset = st.sidebar.selectbox("Dataset", available,  # METR-LA in our project
                               format_func=lambda s: {"metr-la": "METR-LA (Los Angeles)",
                                                      "pems-bay": "PeMS-BAY (Bay Area)"}[s])
models_here = [m for m in DEEP_MODELS if (CHECKPOINT_DIR / f"{dataset}_{m}.pt").exists()]
model_name = st.sidebar.selectbox("Model", models_here, format_func=MODEL_LABELS.get)
horizon_label = st.sidebar.radio("Forecast horizon", ["15 min", "30 min", "60 min"], index=2,
                                 horizontal=True)
h_idx = ["15 min", "30 min", "60 min"].index(horizon_label)
h_steps = HORIZONS[h_idx]

data = get_data(dataset)
win = TrafficWindows(data, "test")
times = data["times"]

# pick "now": a date + a time in the test period (the model never saw these days)
test_days = sorted({t.date() for t in times[win.starts + IN_LEN - 1]})
# default: the first weekday with (almost) no missing readings
day_of = np.array([t.date() for t in times])
good = [k for k, dd in enumerate(test_days)
        if pd.Timestamp(dd).dayofweek < 5 and (data["speed_raw"][day_of == dd] == 0).mean() < 0.02]
day = st.sidebar.selectbox("Day (test period)", test_days, index=good[0] if good else 0,
                           format_func=lambda dd: f"{dd:%a %d %b %Y}")
clock = st.sidebar.slider("Time now", value=pd.Timestamp("08:00").time(),
                          min_value=pd.Timestamp("01:00").time(),
                          max_value=pd.Timestamp("23:00").time(),
                          step=pd.Timedelta(minutes=5))
now = pd.Timestamp.combine(day, clock)
end_idx = int(np.searchsorted(times, now))            # index of the "current" reading
start = int(np.clip(end_idx - IN_LEN + 1, win.starts[0], win.starts[-1]))
now = times[start + IN_LEN - 1]
target_time = times[start + IN_LEN - 1 + h_steps]
st.sidebar.caption(f"Input: {times[start]:%H:%M} to {now:%H:%M}  →  forecast for {target_time:%a %d %b %H:%M}")

# ------------------------------------------------------------------ header
st.title("City-wide traffic speed forecasting")
st.caption(f"{MODEL_LABELS[model_name]} · {dataset.upper()} · {data['num_nodes']} sensors · "
           f"now = {now:%A %d %b %Y, %H:%M}")

pred = forecast(model_name, dataset, start)[:, h_idx]
truth = data["speed_raw"][start + IN_LEN - 1 + h_steps]
current = data["speed_raw"][start + IN_LEN - 1]
valid = truth > 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Average speed now", f"{current[current > 0].mean():.1f} mph")
c2.metric(f"Predicted avg in {horizon_label}", f"{pred.mean():.1f} mph")
c3.metric("Sensors predicted congested (< 40 mph)", int((pred < 40).sum()))
c4.metric("Mean error at this moment",
          f"{np.abs(pred - truth)[valid].mean():.2f} mph" if valid.any() else "no readings")

tab_map, tab_sensor, tab_compare, tab_graph = st.tabs(
    ["🗺️ Forecast map", "📈 Sensor explorer", "📊 Model comparison", "🕸️ Learned graph"])

# ------------------------------------------------------------------ 1. map
with tab_map:
    loc = data["locations"]
    street = st.toggle("Street-map background (needs internet)", value=False)
    left, right = st.columns(2)
    show(network_map(loc, pred, f"Predicted speed at {target_time:%H:%M}",
                     data["adjacency"], street), left)
    show(network_map(loc, np.where(valid, truth, np.nan),
                     f"What really happened at {target_time:%H:%M}",
                     data["adjacency"], street), right)
    st.caption("Each dot is a sensor, grey lines are road-graph links from Module 1. "
               "Red = slow / congested, green = free flow. Hover a dot for the numbers.")

# ------------------------------------------------------------------ 2. one sensor
with tab_sensor:
    worst = int(np.argmin(np.where(valid, truth, 99)))       # default: the slowest sensor
    sensor = st.selectbox("Sensor", range(data["num_nodes"]), index=worst,
                          format_func=lambda i: f"{data['sensor_ids'][i]}  (#{i})")
    span = slice(start - 24, start + IN_LEN + 12)              # a bit of context around the window
    fig = go.Figure()
    actual_series = data["speed_raw"][span, sensor]
    fig.add_trace(go.Scatter(x=times[span], y=np.where(actual_series > 0, actual_series, None),
                             name="Actual speed", line=dict(color="black", width=2)))
    fig.add_vrect(x0=times[start], x1=now, fillcolor="#4C6EF5", opacity=0.08,
                  annotation_text="input: last hour", line_width=0)
    future_t = [times[start + IN_LEN - 1 + h] for h in HORIZONS]
    colours = {"adaptive_st": "#D9480F", "static_st": "#F59F00", "stgcn": "#2B8A3E", "lstm": "#4C6EF5"}
    for m in models_here:
        p = forecast(m, dataset, start)[sensor]
        fig.add_trace(go.Scatter(x=future_t, y=p, mode="markers+lines", name=MODEL_LABELS[m],
                                 marker=dict(size=9), line=dict(color=colours[m], dash="dot")))
    fig.update_layout(height=430, yaxis_title="Speed (mph)", margin=dict(t=20),
                      legend=dict(orientation="h", y=-0.15))
    show(fig, st)
    st.caption("Shaded = the hour the model sees. Dots = forecasts for 15, 30 and 60 minutes later.")

# ------------------------------------------------------------------ 3. comparison
with tab_compare:
    path = RESULTS_DIR / f"{dataset}_results.json"
    if path.exists():
        res = json.loads(path.read_text())
        rows = [{"Model": MODEL_LABELS[m], "Horizon": h, **v} for m, r in res.items() for h, v in r.items()]
        df = pd.DataFrame(rows)
        metric = st.radio("Metric", ["MAE", "RMSE", "MAPE"], horizontal=True)
        fig = px.bar(df, x="Horizon", y=metric, color="Model", barmode="group", height=400,
                     color_discrete_sequence=["#ADB5BD", "#4C6EF5", "#2B8A3E", "#F59F00", "#D9480F"])
        show(fig, st)
        table = df.pivot(index="Model", columns="Horizon", values=metric).round(2)
        st.dataframe(table.loc[[MODEL_LABELS[m] for m in res]])
        st.caption("Test set = the last 20% of the timeline (future data never seen in training). "
                   "MAE/RMSE in mph, MAPE in %.")

# ------------------------------------------------------------------ 4. graph
with tab_graph:
    if "adaptive_st" in models_here:
        m = get_model("adaptive_st", dataset)
        g = m.graph.gate().item()
        st.metric("Gate g (0 = only road map, 1 = only learned graph)", f"{g:.3f}")
        learned = m.graph.learned().detach().numpy()
        static = m.A_static.numpy()
        n = st.slider("Show first N sensors", 20, data["num_nodes"], 60)
        a, b = st.columns(2)
        show(px.imshow(static[:n, :n], color_continuous_scale="Oranges", title="Physical road graph"), a)
        show(px.imshow(learned[:n, :n], color_continuous_scale="Oranges", title="Learned graph"), b)
        st.caption("The learned graph finds sensor pairs that behave alike even when they "
                   "are not neighbours on the road network.")
