"""
config.py  --  settings shared by every module.

Keeping all the "magic numbers" in one file means each student can point to
this file when asked "why 12 steps?" or "why 70/10/20?".
"""
from pathlib import Path

# ---------------------------------------------------------------- folders
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"            # downloaded + processed datasets
RESULTS_DIR = ROOT / "results"      # metrics (json / csv) and training histories
FIGURES_DIR = ROOT / "figures"      # every plot used in the slides
CHECKPOINT_DIR = ROOT / "checkpoints"  # trained model weights (.pt)

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, CHECKPOINT_DIR):
    _d.mkdir(exist_ok=True)

# ---------------------------------------------------------------- datasets
# Both datasets record traffic speed (mph) every 5 minutes.
DATASETS = {
    "metr-la": {
        "start": "2012-03-01 00:00",   # first reading (Thursday)
        "timezone": None,              # stored as plain local clock time
        "city": "Los Angeles",
    },
    # PeMS-BAY also works with this code (python -m m1_data.download_data --all),
    # but our project uses METR-LA only.
    "pems-bay": {
        "start": "2017-01-01 00:00",   # first reading (Sunday)
        "timezone": "America/Los_Angeles",  # this file skips the DST hour
        "city": "San Francisco Bay Area",
    },
}
DATASET = "metr-la"        # the dataset used in this project

STEPS_PER_DAY = 288        # 24 h * 60 min / 5 min

# ---------------------------------------------------------------- forecasting task
IN_LEN = 12                # input  = last 12 readings  = 1 hour of history
HORIZONS = (3, 6, 12)      # output = 3, 6, 12 steps ahead = 15, 30, 60 minutes
HORIZON_NAMES = ("15 min", "30 min", "60 min")

# chronological split -- never shuffled, the test set is always the future
TRAIN_RATIO, VAL_RATIO = 0.7, 0.1   # test = remaining 0.2

# ---------------------------------------------------------------- training
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 15
BATCHES_PER_EPOCH = 100    # each epoch = 100 random batches (6,400 windows), keeps CPU training short
PATIENCE = 5               # early stopping: stop if val MAE does not improve for 5 epochs
SEED = 42

# the models we compare (names used on the command line and in result files)
MODEL_NAMES = ["ha", "lstm", "stgcn", "static_st", "adaptive_st"]
MODEL_LABELS = {
    "ha": "Historical Average",
    "lstm": "LSTM",
    "stgcn": "STGCN",
    "static_st": "Static-Graph ST Transformer",
    "adaptive_st": "Adaptive-Graph ST Transformer",
}
