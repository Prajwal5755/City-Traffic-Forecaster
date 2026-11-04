# City-Traffic-Forecaster

**City-wide urban traffic forecasting with an Adaptive-Graph iTPGT-former**

22AIE304 Deep Learning · B.Tech CSE (AI), Amrita School of Computing, Amritapuri
Team: Prajwal G N (AM.SC.U4AIE24137) · Ridhwajit Bose (AM.SC.U4AIE24141) · Srijita Sarkar (AM.SC.U4AIE24150)

From the last hour of traffic speed at 207 highway sensors in Los Angeles (METR-LA), predict the speed
at every sensor **15, 30 and 60 minutes later**.

Our final model is the **iTPGT-former** graph transformer (Ouyang et al., *Graph Transformer-Based Dynamic
Edge Interaction Encoding for Traffic Prediction*, IEEE T-ITS 2025) **plus our gated adaptive graph**.
It reaches a **60-minute test MAE of 3.41 mph**. That beats the paper's model (3.45), our first adaptive
ST transformer (3.50), STGCN (3.74) and LSTM (4.90).

---

## Results (METR-LA test set: last 20% of the timeline, never seen in training)

| Model | 15 min MAE | 30 min MAE | 60 min MAE | 60 min RMSE | 60 min MAPE |
|---|---|---|---|---|---|
| Historical Average | 4.19 | 4.19 | 4.19 | 7.86 | 13.1% |
| LSTM | 3.08 | 3.78 | 4.90 | 9.68 | 14.9% |
| STGCN (Yu et al. 2018) | 2.88 | 3.25 | 3.74 | 7.64 | 11.1% |
| Static-graph ST Transformer | 2.82 | 3.18 | 3.57 | 7.51 | 10.8% |
| Adaptive-graph ST Transformer (our v1) | 2.79 | 3.14 | 3.50 | 7.32 | 10.4% |
| iTPGT-former (Ouyang et al. 2025) | 2.71 | 3.05 | 3.45 | 7.22 | 10.1% |
| **Adaptive-graph iTPGT-former (ours)** | **2.70** | **3.04** | **3.41** | **7.12** | **9.8%** |

MAE / RMSE in mph. Every model uses the same data, split and training recipe (Colab T4 GPU, up to 50 epochs).

**Congestion check** (a reading is congested if it is below 40 mph, 60 minutes ahead): ours F1 0.795 and recall 0.767,
paper model F1 0.784 and recall 0.735, LSTM F1 0.653 and recall 0.599. Ours catches 77% of real jams an hour ahead.

Full details are in [`RESULTS.md`](RESULTS.md), with plots in [`figures/`](figures/).

---

## How our final model works

Each **sensor** is one token: its last 12 readings go through a linear layer into one vector. Attention then runs
**across sensors**. The attention score between sensors *i* and *j* is

```
score(i, j) = q_i·k_j/√d  +  dynamic_edge(i, j) + position(i, j) + paths(i, j)  +  λ·log A_mix(i, j)
              content        └──────────── iTPGT-former (paper) ────────────┘     └─── ours ───┘
```

* **dynamic edge**: a 1-D convolution reads each sensor's trend; a road edge scores high when both ends behave alike right now
* **position**: a Gaussian kernel of the hop distance between *i* and *j* on the road graph
* **paths**: how many 1, 2 and 3-step road paths connect *i* and *j*
* **ours**: `A_mix = g·A_learned + (1−g)·A_road`, our learned graph mixed with the road graph by a learned gate *g*.
  Because `softmax(log A) = A`, attention starts out following our graph. The final gate is g = 0.72.

Code: [`m2_models/itpgt_former.py`](m2_models/itpgt_former.py) (paper) and
[`m2_models/adaptive_itpgt.py`](m2_models/adaptive_itpgt.py) (our extra ~20 lines).

---

## Quick start

```bash
git clone https://github.com/Prajwal5755/City-Traffic-Forecaster.git
cd City-Traffic-Forecaster
python3 -m venv dl && source dl/bin/activate      # the dl/ folder is git-ignored
pip install -r requirements.txt

streamlit run webapp/app.py                       # web app, opens http://localhost:8501
```

The trained weights in `checkpoints/` and the data in `data/metr-la/` are included, so the web app and
evaluation work without retraining.

```bash
python -m m3_training.evaluate --dataset metr-la              # test every trained model
python -m m3_training.congestion                              # congestion accuracy / F1
python -m m3_training.plots --dataset metr-la                 # all result figures
python -m m3_training.train --dataset metr-la --model adaptive_itpgt   # retrain one model
python run_all.py                                             # everything in one go
```

**GPU training:** open `colab_train_adaptive_itpgt.ipynb` (our final model) or `colab_train_itpgt.ipynb`
(the paper's model) in Google Colab with a T4 GPU and choose Run all. `colab_train.ipynb` and `train_local.ipynb` train every model.

---

## Project layout (one module per student)

| Module | Owner | Folder | What it covers |
|---|---|---|---|
| 1. Data pipeline and road graph | Prajwal G N | `m1_data/` | Download, cleaning (0 mph = broken sensor), 70/10/20 split by time, z-score, sliding windows, Gaussian-kernel road graph |
| 2. Models | Ridhwajit Bose | `m2_models/` | Historical Average, LSTM, STGCN, static and adaptive ST transformer, iTPGT-former, **adaptive iTPGT-former (ours)** |
| 3. Training, evaluation, web app | Srijita Sarkar | `m3_training/`, `webapp/` | Shared training recipe, MAE / RMSE / MAPE, congestion F1, plots, Streamlit dashboard |

Other files:

| Path | What it is |
|---|---|
| `config.py` | Every setting: window length, horizons, split, learning rate, model names |
| `checkpoints/` | Trained weights (`metr-la_<model>.pt`) |
| `results/`, `figures/` | Metrics (JSON / CSV), training histories and plots |
| `data.py`, `model.py`, `train.py`, `readme.md` | Early sanity-check version on synthetic data (mid-review); `python train.py` runs the LSTM, our v1 and our final model in about 3 minutes on CPU |

---

## Data and references

* **METR-LA**: 207 loop detectors, 1 Mar to 27 Jun 2012, one reading every 5 minutes (Li et al., DCRNN, ICLR 2018; github.com/liyaguang/DCRNN). The speed CSV mirrors `metr-la.h5` (github.com/hazdzz/STGCN).
* Yu et al., STGCN, IJCAI 2018 · Li et al., DCRNN, ICLR 2018 · Guo et al., ASTGCN, AAAI 2019 · Wu et al., Graph WaveNet, IJCAI 2019
* Ouyang et al., Graph Transformer-Based Dynamic Edge Interaction Encoding for Traffic Prediction (iTPGT-former), IEEE Transactions on Intelligent Transportation Systems, 2025
