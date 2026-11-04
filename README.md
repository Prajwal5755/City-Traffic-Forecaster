# City-Wide Urban Traffic Forecasting with an Adaptive-Graph Spatio-Temporal Transformer

22AIE304 Deep Learning · B.Tech CSE (AI), Amrita School of Computing, Amritapuri
Team: Prajwal G N (AM.SC.U4AIE24137) · Ridhwajit Bose (AM.SC.U4AIE24141) · Srijita Sarkar (AM.SC.U4AIE24150)

**Task.** Given the last 1 hour of traffic speed from every sensor on a highway network
(12 readings, one every 5 minutes), predict the speed at every sensor **15, 30 and 60
minutes later**.

**Data.** The real **METR-LA** benchmark named in our proposal: 207 loop detectors on Los Angeles
highways, 1 Mar to 27 Jun 2012, one reading every 5 minutes (34,272 time steps). (The code can also
load PeMS-BAY with `python -m m1_data.download_data --all`, but this project uses METR-LA only.)

Speeds are the same numbers as the official `metr-la.h5` file (mirrored as CSV
by github.com/hazdzz/STGCN); sensor distances and GPS locations come from the DCRNN authors
(github.com/liyaguang/DCRNN).

---

## The three modules (one per student)

| Module | Owner | Folder | Question it answers |
|---|---|---|---|
| 1. Data pipeline and graph construction | Prajwal G N | `m1_data/` | How do raw sensor readings become clean training samples and a road graph? |
| 2. Forecasting models | Ridhwajit Bose | `m2_models/` | Which architectures do we compare, and how does our adaptive graph work? |
| 3. Training, evaluation and web app | Srijita Sarkar | `m3_training/`, `webapp/` | How are models trained fairly, how good are they, and how would a user use them? |

(The owner column is a suggestion; swap names freely. Only this table and the slides mention them.)

### Module 1: `m1_data/`
| File | What it does |
|---|---|
| `download_data.py` | Downloads the speed CSV, sensor distances, sensor locations |
| `dataset.py` | Timestamps, missing values (0 mph = broken sensor), chronological 70/10/20 split, z-score normalisation, sliding windows (12 in, 3 horizons out), PyTorch DataLoaders |
| `graph.py` | Weighted adjacency matrix with the thresholded Gaussian kernel `W_ij = exp(-(d_ij/σ)²)` if ≥ 0.1, plus the normalisations each model needs |
| `explore.py` | Data-exploration figures and numbers for the slides |

### Module 2: `m2_models/`
| File | Model | What it tests |
|---|---|---|
| `historical_average.py` | Historical Average | "Usual speed at this time on this weekday" (no learning) |
| `lstm.py` | LSTM | Time only, no graph |
| `stgcn.py` | STGCN (Yu et al., 2018) | Gated temporal convolutions + graph convolution, fixed graph (the model in our proposal) |
| `st_transformer.py` (`adaptive=False`) | Static-Graph ST Transformer | Our architecture with only the physical road graph |
| `st_transformer.py` (`adaptive=True`) | **Adaptive-Graph ST Transformer (ours)** | Physical graph + a graph learned from data, mixed by a learned gate |

### Module 3: `m3_training/` and `webapp/`
| File | What it does |
|---|---|
| `metrics.py` | Masked MAE, RMSE, MAPE (missing readings ignored) |
| `train.py` | Same recipe for every model: masked-MAE loss, Adam, LR halving, gradient clipping, early stopping on validation MAE |
| `evaluate.py` | Test-set metrics per model and horizon, saved as JSON + CSV |
| `plots.py` | Training curves, horizon chart, forecast example, learned graph |
| `webapp/app.py` | Streamlit dashboard with live predictions on a map |

`config.py` holds every setting (window length, horizons, split, learning rate...).

---

## How to run

```bash
pip install -r requirements.txt

python -m m1_data.download_data        # Module 1: get METR-LA (already included in data/)
python -m m1_data.explore              # Module 1: figures + data summary
python -m m3_training.train --dataset metr-la --model adaptive_st   # train one model
python -m m3_training.evaluate --dataset metr-la                    # test all trained models
python -m m3_training.plots --dataset metr-la                       # result figures

python run_all.py                      # or: everything in one go
python run_all.py --quick              # 2-minute smoke test

streamlit run webapp/app.py            # the web app
```

**GPU training:** open `train_local.ipynb` (your own CUDA machine) or `colab_train.ipynb` (Google Colab)
and run all cells. `checkpoints/` already holds the final GPU-trained weights (Colab T4, 50 epochs),
so the web app and `evaluate.py` work without retraining.

## Results

See `results/` (numbers) and `figures/` (plots); a summary is in `RESULTS.md`.
