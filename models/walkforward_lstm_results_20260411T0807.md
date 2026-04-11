# LSTM+CatBoost Walk-Forward Training Report

**Generated**: 20260411T0807  
**probability_threshold**: 0.65

---

## 1. Model Configuration

| Parameter | Value |
|-----------|-------|
| neutral_scale (K) | 1.3 |
| lookforward_bars | 6 |
| seq_len | 48 |
| hidden_dim | 128 |
| lstm_output_dim | 64 |
| probability_threshold | 0.65 |

---

## 2. Dataset Split

### Fold 0
- **Training**: 2024-03 ~ 2025-06 (16 months)
- **Validation**: 2025-07
- **Test**: 2025-08

### Fold 1
- **Training**: 2024-03 ~ 2025-08 (18 months)
- **Validation**: 2025-09
- **Test**: 2025-10

---

## 3. Key Metrics Summary

| Metric | Fold 0 | Fold 1 | Average |
|--------|--------|--------|---------|
| **Macro F1** | 0.7091 | 0.7589 | **0.7340** |
| **Accuracy** | 0.7493 | 0.7967 | **0.7730** |
| **Composite Score** | 1.0641 | 1.3938 | **1.2289** |
| **Long Recall** | 0.6872 | 0.7196 | **0.7034** |
| **Short Recall** | 0.7558 | 0.7378 | **0.7468** |
| **Long Precision** | 0.5927 | 0.6866 | **0.6396** |
| **Short Precision** | 0.6376 | 0.7278 | **0.6827** |
| **Neutral Ratio** | 59.3% | 63.5% | **61.4%** |

---

## 4. Prediction Distribution

### Fold 0 (Test: 2025-08)
| Class | Count | Percentage |
|-------|-------|------------|
| DOWN (Short) | 665 | 22.3% |
| NEUTRAL | 1766 | 59.3% |
| UP (Long) | 545 | 18.3% |

### Fold 1 (Test: 2025-10)
| Class | Count | Percentage |
|-------|-------|------------|
| DOWN (Short) | 518 | 17.4% |
| NEUTRAL | 1890 | 63.5% |
| UP (Long) | 568 | 19.1% |

---

## 5. 18-Probability Evaluation System

### Fold 0

**P(actual|predicted)**
| Predicted | Actual Long | Actual Neutral | Actual Short |
|------------|-------------|----------------|--------------|
| Long | 63.8% | 36.1% | 0.2% |
| Neutral | 7.8% | 84.0% | 8.3% |
| Short | 0.0% | 40.7% | 59.3% |

**P(predicted|actual)**
| Actual | Pred Long | Pred Neutral | Pred Short |
|--------|-----------|--------------|------------|
| Long | 75.6% | 42.8% | 0.2% |
| Neutral | 7.0% | 76.2% | 7.5% |
| Short | 0.0% | 47.2% | 68.7% |

### Fold 1

**P(actual|predicted)**
| Predicted | Actual Long | Actual Neutral | Actual Short |
|------------|-------------|----------------|--------------|
| Long | 72.8% | 27.2% | 0.0% |
| Neutral | 7.1% | 84.9% | 8.0% |
| Short | 0.0% | 31.3% | 68.7% |

**P(predicted|actual)**
| Actual | Pred Long | Pred Neutral | Pred Short |
|--------|-----------|--------------|------------|
| Long | 73.8% | 27.6% | 0.0% |
| Neutral | 7.0% | 83.4% | 7.9% |
| Short | 0.0% | 32.8% | 72.0% |

---

## 6. Confusion Matrix

### Fold 0 (Average)
```
         Predicted
         DOWN    NEUTRAL  UP
Actual DOWN   424      137       0
       NEUTRAL 240      1483      222
       UP     1       146       323
```

### Fold 1 (Average)
```
         Predicted
         DOWN    NEUTRAL  UP
Actual DOWN   377      134       0
       NEUTRAL 141      1604      178
       UP     0       152       390
```

---

## 7. Key Findings

- Direction flip penalty is effective: P(aS|pL) and P(aL|pS) are both ~0%
- Neutral ratio with threshold=0.65: Fold0=59.3%, Fold1=63.5%
- Average Macro F1: 0.7340
- Average Composite Score: 1.2289

---

## 8. Model Storage

```
models/lstm_catboost/
├── fold_0/
│   ├── lstm_model.pt
│   ├── catboost_model.cbm
│   └── scaler.joblib
└── fold_1/
    ├── lstm_model.pt
    ├── catboost_model.cbm
    └── scaler.joblib
```
