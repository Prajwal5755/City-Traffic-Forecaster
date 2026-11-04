import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data import make_synthetic_dataset, make_windows, TrafficWindowDataset
from model import AdaptiveGraphSTTransformer, LSTMBaseline

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

HORIZONS = (3, 6, 12)          # steps ahead -> 15 / 30 / 60 min at 5-min sampling
IN_LEN = 12                    # 1 hour of history
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def masked_mae(pred, target):
    
    mask = (target != 0).float()
    mask = mask / (mask.mean() + 1e-8)
    loss = torch.abs(pred - target) * mask
    return loss.mean()


def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))


def evaluate(model, loader, static_adj):
    model.eval()
    total_mae, total_rmse, n_batches = 0.0, 0.0, 0
    per_horizon_mae = np.zeros(len(HORIZONS))
    with torch.no_grad():
        for X, Y, TOD, DOW in loader:
            X, Y, TOD, DOW = X.to(DEVICE), Y.to(DEVICE), TOD.to(DEVICE), DOW.to(DEVICE)
            pred = model(X, TOD, DOW, static_adj)
            total_mae += masked_mae(pred, Y).item()
            total_rmse += rmse(pred, Y).item()
            per_horizon_mae += torch.abs(pred - Y).mean(dim=(0, 1)).cpu().numpy()
            n_batches += 1
    return (total_mae / n_batches, total_rmse / n_batches,
            (per_horizon_mae / n_batches).tolist())


def train_one_model(model, train_loader, val_loader, static_adj, name):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    history = []
    print(f"\n=== Training {name} ===")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for X, Y, TOD, DOW in train_loader:
            X, Y, TOD, DOW = X.to(DEVICE), Y.to(DEVICE), TOD.to(DEVICE), DOW.to(DEVICE)
            opt.zero_grad()
            pred = model(X, TOD, DOW, static_adj)
            loss = masked_mae(pred, Y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        val_mae, val_rmse, _ = evaluate(model, val_loader, static_adj)
        history.append({"epoch": epoch, "train_mae": epoch_loss,
                         "val_mae": val_mae, "val_rmse": val_rmse})
        print(f"epoch {epoch:2d}  train_mae={epoch_loss:.3f}  "
              f"val_mae={val_mae:.3f}  val_rmse={val_rmse:.3f}")
    return history


def main():
    print("Generating synthetic dataset (stand-in for METR-LA/PeMS-BAY)...")
    speed_matrix, static_adj_np, tod_idx, dow_idx = make_synthetic_dataset(
        num_nodes=50, num_timesteps=2000, num_clusters=5, seed=SEED
    )
    X, Y, TOD, DOW = make_windows(speed_matrix, tod_idx, dow_idx,
                                   in_len=IN_LEN, horizons=HORIZONS)
    dataset = TrafficWindowDataset(X, Y, TOD, DOW)

    n = len(dataset)
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    static_adj = torch.tensor(static_adj_np, dtype=torch.float32).to(DEVICE)
    num_nodes = speed_matrix.shape[1]

    lstm = LSTMBaseline(hidden_dim=32, num_layers=2, num_horizons=len(HORIZONS))
    ours = AdaptiveGraphSTTransformer(
        num_nodes=num_nodes, hidden_dim=32, num_layers=2,
        num_heads=4, num_horizons=len(HORIZONS),
    )

    t0 = time.time()
    hist_lstm = train_one_model(lstm, train_loader, val_loader, static_adj, "LSTM baseline")
    t1 = time.time()
    hist_ours = train_one_model(ours, train_loader, val_loader, static_adj,
                                 "Adaptive-graph ST transformer")
    t2 = time.time()

    print("\n=== Final test-set metrics ===")
    lstm_mae, lstm_rmse, lstm_ph = evaluate(lstm, test_loader, static_adj)
    ours_mae, ours_rmse, ours_ph = evaluate(ours, test_loader, static_adj)
    print(f"LSTM baseline                : MAE={lstm_mae:.3f}  RMSE={lstm_rmse:.3f}  "
          f"(train time {t1 - t0:.1f}s)")
    print(f"Adaptive-graph ST transformer: MAE={ours_mae:.3f}  RMSE={ours_rmse:.3f}  "
          f"(train time {t2 - t1:.1f}s)")
    for i, h in enumerate(HORIZONS):
        mins = h * 5
        print(f"  horizon {mins:>2d} min  ->  LSTM MAE={lstm_ph[i]:.3f}   "
              f"Ours MAE={ours_ph[i]:.3f}")


    np.save("history_lstm.npy", hist_lstm, allow_pickle=True)
    np.save("history_ours.npy", hist_ours, allow_pickle=True)

    with torch.no_grad():
        learned_adj = ours.graph_module(static_adj).cpu().numpy()
        gate = torch.sigmoid(ours.graph_module.alpha).item()
    print(f"\nLearned graph gate alpha (0=trust distance graph, 1=trust learned graph): {gate:.3f}")
    np.save("learned_adjacency.npy", learned_adj)
    np.save("static_adjacency.npy", static_adj_np)


if __name__ == "__main__":
    main()
