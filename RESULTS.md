# Results: METR-LA test set

Trained on a Google Colab T4 GPU, 50 epochs over the full training set for every model.
Test set = last 20% of the timeline (6,832 windows), never seen in training. MAE / RMSE in mph, MAPE in %.

| Model | 15 min MAE / RMSE / MAPE | 30 min | 60 min |
|---|---|---|---|
| Historical Average | 4.19 / 7.86 / 13.1% | 4.19 / 7.86 / 13.1% | 4.19 / 7.86 / 13.1% |
| LSTM | 3.08 / 6.10 / 8.1% | 3.78 / 7.67 / 10.6% | 4.90 / 9.68 / 14.9% |
| STGCN | 2.88 / 5.48 / 7.6% | 3.25 / 6.47 / 9.1% | 3.74 / 7.64 / 11.1% |
| Static-graph ST Transformer | 2.82 / 5.49 / 7.6% | 3.18 / 6.51 / 9.1% | 3.57 / 7.51 / 10.8% |
| **Adaptive-graph ST Transformer (ours)** | 2.79 / 5.41 / 7.4% | 3.14 / 6.39 / 8.8% | 3.50 / 7.32 / 10.4% |

## Key points

- Our adaptive-graph model is best on every metric at every horizon.
- 60-min MAE: 3.50 mph, 29% lower than LSTM and 6% lower than STGCN.
- Ablation: adding the learned graph lowers 60-min MAE from 3.57 to 3.50 (2%) and RMSE from 7.51 to 7.32, for about 4k extra parameters.
- Final gate g = 0.51: the model uses the road graph and the learned graph about equally.
- Historical Average ignores the last hour, so its error is the same at every horizon (4.19).
