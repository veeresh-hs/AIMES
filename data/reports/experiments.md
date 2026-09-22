# Experimental results

Model version `v1`, runtime mode ADVANCED, split strategy time.

## E1 - individual layers

| Layer | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | Latency (ms) | Coverage |
|---|---|---|---|---|---|---|---|---|
| lexical | 0.9492 | 0.5055 | 1.0000 | 0.6715 | 0.9969 | 0.0536 | 2.25 | 1.00 |
| semantic | 0.8893 | 0.3060 | 0.8913 | 0.4556 | 0.9707 | 0.1108 | 16.22 | 1.00 |
| anomaly | 0.9525 | 0.5385 | 0.6087 | 0.5714 | 0.8853 | 0.0286 | 8.09 | 1.00 |
| url | 0.9492 | 1.0000 | 0.0217 | 0.0426 | 0.4364 | 0.0000 | 0.07 | 1.00 |
| header | 0.8497 | 0.0000 | 0.0000 | 0.0000 | 0.3405 | 0.1037 | 0.02 | 0.99 |
| attachment | 0.9480 | 0.0000 | 0.0000 | 0.0000 | 0.4994 | 0.0000 | 0.00 | 1.00 |
| impersonation | 0.9480 | 0.0000 | 0.0000 | 0.0000 | 0.4923 | 0.0000 | 0.05 | 0.99 |
| social_engineering | 0.9480 | 0.0000 | 0.0000 | 0.0000 | 0.7956 | 0.0000 | 0.99 | 1.00 |

## E1b - lexical baselines (validation split, multiclass)

| Algorithm | Accuracy | Macro F1 | Train (s) | Inference (ms) |
|---|---|---|---|---|
| MultinomialNB | 0.9819 | 0.9719 | 0.02 | 0.0081 |
| LogisticRegression | 0.9683 | 0.9529 | 0.5 | 0.0031 |
| LinearSVC | 0.9819 | 0.9724 | 0.84 | 0.0073 |
| RandomForest | 0.8247 | 0.7769 | 1.58 | 0.179 |

## E2 - fusion

| Strategy | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |
|---|---|---|---|---|---|---|---|
| fusion:weighted | 0.9853 | 0.8367 | 0.8913 | 0.8632 | 0.9966 | 0.8883 | 0.0095 |
| fusion:stacking | 0.9740 | 0.6667 | 1.0000 | 0.8000 | 0.9967 | 0.8886 | 0.0274 |
| ENSEMBLE:end_to_end | 0.9740 | 0.6667 | 1.0000 | 0.8000 | 0.9967 | 0.9107 | 0.0274 |

## E3 - leave-one-layer-out ablation

Positive delta means removing the layer made the ensemble worse.

| Layer removed | ROC-AUC without | F1 without | FPR without | Delta ROC-AUC |
|---|---|---|---|---|
| lexical | 0.9030 | 0.0392 | 0.0048 | +0.0936 |
| transformer | 0.9966 | 0.8632 | 0.0095 | +0.0000 |
| url | 0.9966 | 0.8660 | 0.0107 | +0.0000 |
| header | 0.9966 | 0.8660 | 0.0107 | +0.0000 |
| attachment | 0.9966 | 0.8632 | 0.0095 | +0.0000 |
| semantic | 0.9967 | 0.8632 | 0.0095 | -0.0000 |
| social_engineering | 0.9967 | 0.8627 | 0.0143 | -0.0000 |
| impersonation | 0.9967 | 0.8542 | 0.0107 | -0.0001 |
| anomaly | 0.9967 | 0.8485 | 0.0131 | -0.0001 |

## E4 - cross-corpus generalisation

No held-out corpus available. Add a second source under data/raw (for example the Nazario phishing corpus) to run this experiment.

## E5 - operating points

Cost model: 5 x false negatives + 1 x false positives.

| Target recall | Threshold | Achieved recall | FPR |
|---|---|---|---|
| 0.90 | 0.840 | 0.9130 | 0.0107 |
| 0.95 | 0.790 | 0.9565 | 0.0155 |
| 0.99 | 0.730 | 1.0000 | 0.0179 |

Cost-optimal threshold: 0.70 (recall 1.0000, FPR 0.0191).

## E6 - latency per layer

| Layer | Mean (ms) | p95 (ms) |
|---|---|---|
| semantic | 16.22 | 17.99 |
| anomaly | 8.09 | 10.78 |
| lexical | 2.25 | 4.34 |
| social_engineering | 0.99 | 2.92 |
| url | 0.07 | 0.13 |
| impersonation | 0.05 | 0.02 |
| header | 0.02 | 0.02 |
| transformer | 0.00 | 0.00 |
| attachment | 0.00 | 0.00 |
