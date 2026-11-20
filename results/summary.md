Mean over 8 patient-level splits (seeds 0-7). RMSE, CRPS and interval score in mg/dL.

| Model | RMSE 60 | CRPS 60 | Interval score (conformal) | Coverage | AUROC hypo | Brier hypo | ECE hypo | Energy dist. | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|
| Ridge regression | 25.4 | – | – | – | 0.969 | – | – | – | 0.05 |
| GRU | 24.7 | – | – | – | 0.968 | – | – | – | 0.42 |
| Gaussian MLP | 25.1 | 12.9 | 106.5 | 0.911 | 0.971 | 0.022 | 0.029 | 2.39 | 0.18 |
| Transformer, Gaussian head | 25.0 | 12.7 | 105.2 | 0.911 | 0.975 | 0.022 | 0.030 | 2.28 | 0.60 |
| Quantile network + CQR | 24.8 | – | 101.9 | 0.895 | 0.941 | – | – | – | 0.11 |
| CVAE twin | 31.3 | 16.6 | 143.4 | 0.907 | 0.934 | 0.023 | 0.028 | 1.46 | 0.21 |
| Diffusion twin | 25.2 | 12.8 | 107.1 | 0.908 | 0.954 | 0.016 | 0.008 | 0.59 | 4.68 |
| Ridge + CVAE | 25.5 | 13.4 | 117.4 | 0.905 | 0.961 | 0.019 | 0.019 | 0.85 | 0.34 |
| Ridge + diffusion | 25.5 | 13.0 | 110.8 | 0.907 | 0.959 | 0.016 | 0.007 | 0.55 | 4.76 |
| Transformer + CVAE | 25.0 | 13.0 | 115.3 | 0.904 | 0.956 | 0.019 | 0.016 | 0.92 | 0.88 |
| Transformer + diffusion | 25.0 | 12.8 | 109.2 | 0.907 | 0.954 | 0.017 | 0.006 | 0.58 | 5.27 |
