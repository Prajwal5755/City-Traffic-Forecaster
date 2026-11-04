"""
run_all.py  --  reproduce every number and figure in the presentation.

    python run_all.py                       # METR-LA, all models
    python run_all.py --quick               # 2 epochs x 50 batches: checks everything runs

Order = the order of the three modules:
    Module 1  data + graph   ->  Module 2  models   ->  Module 3  train, evaluate, plot
"""
import argparse
import subprocess
import sys

from config import BATCHES_PER_EPOCH, MAX_EPOCHS
from m1_data.dataset import prepare_dataset
from m3_training.evaluate import evaluate_all
from m3_training.train import train

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="metr-la", choices=["metr-la", "pems-bay"])
parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
parser.add_argument("--batches", type=int, default=BATCHES_PER_EPOCH, help="training batches per epoch")
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()

epochs, max_batches = (2, 20) if args.quick else (args.epochs, args.batches)

subprocess.run([sys.executable, "-m", "m1_data.download_data"], check=True)   # Module 1
subprocess.run([sys.executable, "-m", "m1_data.explore"], check=True)         # Module 1
data = prepare_dataset(args.dataset)
for model_name in ["lstm", "stgcn", "static_st", "adaptive_st"]:              # Modules 2 + 3
    train(args.dataset, model_name, epochs, max_batches, data=data)
evaluate_all(args.dataset, data=data)                                         # Module 3
subprocess.run([sys.executable, "-m", "m3_training.plots", "--dataset", args.dataset], check=True)
