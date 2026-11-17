**California Housing (regression)**: 12,384 train / 4,128 val / 4,128 test rows, 8 features, depth 6 (at most 64 leaves), 400 rounds, learning rate 0.1, mean ± std over 3 split seeds.

| Library | RMSE | MAE | R² | Train CPU time (s) |
|---|---|---|---|---|
| ours (depth-wise) | 0.4599 ± 0.0061 | 0.3007 ± 0.0017 | 0.8398 ± 0.0023 | 0.62 |
| ours (leaf-wise) | 0.4588 ± 0.0070 | 0.3003 ± 0.0026 | 0.8407 ± 0.0026 | 0.65 |
| XGBoost | 0.4612 ± 0.0069 | 0.3010 ± 0.0018 | 0.8389 ± 0.0029 | 0.33 |
| LightGBM | 0.4556 ± 0.0052 | 0.2985 ± 0.0015 | 0.8428 ± 0.0014 | 0.29 |
| sklearn HistGB | 0.4554 ± 0.0106 | 0.2981 ± 0.0053 | 0.8430 ± 0.0051 | 0.48 |

**Adult (binary)**: 29,304 train / 9,769 val / 9,769 test rows, 14 features, depth 6 (at most 64 leaves), 300 rounds, learning rate 0.1, mean ± std over 3 split seeds.

| Library | Accuracy | AUC | Log-loss | Train CPU time (s) |
|---|---|---|---|---|
| ours (depth-wise) | 0.8727 ± 0.0009 | 0.9287 ± 0.0020 | 0.2767 ± 0.0032 | 0.62 |
| ours (leaf-wise) | 0.8682 ± 0.0008 | 0.9246 ± 0.0017 | 0.2873 ± 0.0032 | 0.80 |
| XGBoost | 0.8721 ± 0.0008 | 0.9286 ± 0.0020 | 0.2769 ± 0.0032 | 0.30 |
| LightGBM | 0.8698 ± 0.0017 | 0.9253 ± 0.0016 | 0.2860 ± 0.0035 | 0.44 |
| sklearn HistGB | 0.8691 ± 0.0017 | 0.9250 ± 0.0018 | 0.2862 ± 0.0038 | 0.79 |

**Higgs (binary)**: 58,830 train / 19,610 val / 19,610 test rows, 28 features, depth 6 (at most 64 leaves), 300 rounds, learning rate 0.1, mean ± std over 3 split seeds.

| Library | Accuracy | AUC | Log-loss | Train CPU time (s) |
|---|---|---|---|---|
| ours (depth-wise) | 0.7226 ± 0.0039 | 0.8021 ± 0.0033 | 0.5364 ± 0.0037 | 3.06 |
| ours (leaf-wise) | 0.7225 ± 0.0028 | 0.8016 ± 0.0032 | 0.5371 ± 0.0038 | 2.73 |
| XGBoost | 0.7255 ± 0.0042 | 0.8028 ± 0.0042 | 0.5358 ± 0.0047 | 1.77 |
| LightGBM | 0.7244 ± 0.0035 | 0.8019 ± 0.0041 | 0.5369 ± 0.0049 | 1.76 |
| sklearn HistGB | 0.7254 ± 0.0052 | 0.8023 ± 0.0039 | 0.5364 ± 0.0043 | 2.07 |

**Covertype, 200k rows (7 classes)**: 120,000 train / 40,000 val / 40,000 test rows, 54 features, depth 8 (at most 256 leaves), 200 rounds, learning rate 0.1, mean ± std over 2 split seeds.

| Library | Accuracy | Log-loss | Train CPU time (s) |
|---|---|---|---|
| ours (depth-wise) | 0.9128 ± 0.0010 | 0.2332 ± 0.0000 | 46.74 |
| ours (leaf-wise) | 0.9434 ± 0.0000 | 0.1577 ± 0.0016 | 53.79 |
| XGBoost | 0.8804 ± 0.0005 | 0.3084 ± 0.0018 | 11.72 |
| LightGBM | 0.9423 ± 0.0006 | 0.1625 ± 0.0005 | 8.24 |
| sklearn HistGB | 0.9446 ± 0.0006 | 0.1593 ± 0.0003 | 57.27 |

Train CPU time: single thread, best of the runs above, binning included.


**Test quality at the validation-optimal round (split seed 0, early-stopping style)**


California Housing (regression)

| Library | Best round | rmse | mae |
|---|---|---|---|
| ours (depth-wise) | 400 | 0.4524 | 0.2991 |
| ours (leaf-wise) | 364 | 0.4517 | 0.2979 |
| XGBoost | 398 | 0.4525 | 0.2993 |
| LightGBM | 349 | 0.4499 | 0.2965 |
| sklearn HistGB | 400 | 0.4457 | 0.2940 |

Adult (binary)

| Library | Best round | accuracy | auc | logloss |
|---|---|---|---|---|
| ours (depth-wise) | 161 | 0.8741 | 0.9276 | 0.2780 |
| ours (leaf-wise) | 107 | 0.8721 | 0.9264 | 0.2802 |
| XGBoost | 153 | 0.8740 | 0.9275 | 0.2779 |
| LightGBM | 99 | 0.8717 | 0.9269 | 0.2792 |
| sklearn HistGB | 100 | 0.8727 | 0.9263 | 0.2803 |

Higgs (binary)

| Library | Best round | accuracy | auc | logloss |
|---|---|---|---|---|
| ours (depth-wise) | 195 | 0.7211 | 0.7994 | 0.5390 |
| ours (leaf-wise) | 176 | 0.7212 | 0.7988 | 0.5398 |
| XGBoost | 163 | 0.7212 | 0.7989 | 0.5395 |
| LightGBM | 147 | 0.7214 | 0.7993 | 0.5390 |
| sklearn HistGB | 245 | 0.7195 | 0.7988 | 0.5397 |

Covertype, 200k rows (7 classes)

| Library | Best round | accuracy | logloss |
|---|---|---|---|
| ours (depth-wise) | 200 | 0.9119 | 0.2332 |
| ours (leaf-wise) | 200 | 0.9434 | 0.1561 |
| XGBoost | 200 | 0.8799 | 0.3066 |
| LightGBM | 200 | 0.9416 | 0.1620 |
| sklearn HistGB | 200 | 0.9439 | 0.1596 |
