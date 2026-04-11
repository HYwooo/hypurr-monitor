# LSTM+CatBoost Walk-Forward Training Report

**Generated**: 20260411T0838  
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
| **Macro F1** | 0.2704 | 0.3980 | **0.3342** |
| **Accuracy** | 0.6539 | 0.6505 | **0.6522** |
| **Composite Score** | -0.0604 | -0.6123 | **-0.3364** |
| **Long Recall** | 0.0106 | 0.0830 | **0.0468** |
| **Short Recall** | 0.0000 | 0.1859 | **0.0930** |
| **Long Precision** | 0.5556 | 0.4369 | **0.4962** |
| **Short Precision** | 0.0000 | 0.5460 | **0.2730** |
| **Neutral Ratio** | 99.7% | 90.7% | **95.2%** |

---

## 4. Prediction Distribution

### Fold 0 (Test: 2025-08)
| Class | Count | Percentage |
|-------|-------|------------|
| DOWN (Short) | 0 | 0.0% |
| NEUTRAL | 2967 | 99.7% |
| UP (Long) | 9 | 0.3% |

### Fold 1 (Test: 2025-10)
| Class | Count | Percentage |
|-------|-------|------------|
| DOWN (Short) | 174 | 5.8% |
| NEUTRAL | 2699 | 90.7% |
| UP (Long) | 103 | 3.5% |

---

## 5. 18-Probability Evaluation System

### Fold 0

**P(actual|predicted)**
| Predicted | Actual Long | Actual Neutral | Actual Short |
|------------|-------------|----------------|--------------|
| Long | 0.0% | 0.0% | 0.0% |
| Neutral | 18.9% | 65.4% | 15.7% |
| Short | 0.0% | 44.4% | 55.6% |

**P(predicted|actual)**
| Actual | Pred Long | Pred Neutral | Pred Short |
|--------|-----------|--------------|------------|
| Long | 0.0% | 0.0% | 0.0% |
| Neutral | 28.8% | 99.8% | 23.9% |
| Short | 0.0% | 0.9% | 1.1% |

### Fold 1

**P(actual|predicted)**
| Predicted | Actual Long | Actual Neutral | Actual Short |
|------------|-------------|----------------|--------------|
| Long | 54.6% | 41.4% | 4.0% |
| Neutral | 15.3% | 66.5% | 18.2% |
| Short | 2.9% | 53.4% | 43.7% |

**P(predicted|actual)**
| Actual | Pred Long | Pred Neutral | Pred Short |
|--------|-----------|--------------|------------|
| Long | 18.6% | 14.1% | 1.4% |
| Neutral | 21.5% | 93.4% | 25.5% |
| Short | 0.6% | 10.1% | 8.3% |

---

## 6. Confusion Matrix

### Fold 0 (Average)
```
         Predicted
         DOWN    NEUTRAL  UP
Actual DOWN   0      561       0
       NEUTRAL 0      1941      4
       UP     0       465       5
```

### Fold 1 (Average)
```
         Predicted
         DOWN    NEUTRAL  UP
Actual DOWN   95      413       3
       NEUTRAL 72      1796      55
       UP     7       490       45
```

---

## 7. Key Findings

- Direction flip penalty is effective: P(aS|pL) and P(aL|pS) are both ~0%
- Neutral ratio with threshold=0.65: Fold0=99.7%, Fold1=90.7%
- Average Macro F1: 0.3342
- Average Composite Score: -0.3364

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
