# LSTM+CatBoost Walk-Forward 训练报告

**Generated**: 20260411T1151
**Train**: 2024-01 ~ 2025-08 (20 months)
**Valid**: 2025-09 | **Test**: 2025-10, 2025-11
**probability_threshold**: 0.65

---

## 1. 模型配置

| Parameter | Value |
|-----------|-------|
| neutral_scale (K) | 1.3 |
| lookforward_bars | 6 |
| seq_len | 48 |
| hidden_dim | 128 |
| lstm_output_dim | 64 |
| probability_threshold | 0.65 |

---

## 2. Walk-Forward 划分

| Fold | Train | Valid | Test |
|------|-------|-------|------|
| Fold 0 | 2024-01 ~ 2025-08 | 2025-09 | 2025-10 |
| Fold 1 | 2024-01 ~ 2025-09 | 2025-10 | 2025-11 |

---

## 3. 核心指标汇总

### Fold 0 (Valid: 2025-09)

| Metric | Value |
|--------|-------|
| **Macro F1** | 0.6706 |
| **Composite Score** | -0.1558 |
| **Neutral Ratio** | 62.7% |
| **Long Recall** | 0.6294 |
| **Short Recall** | 0.6038 |
| **Long Precision** | 0.6611 |
| **Short Precision** | 0.5847 |

### Fold 0 (Test: 2025-10)

| Metric | Value |
|--------|-------|
| **Macro F1** | 0.6385 |
| **Composite Score** | -0.2748 |
| **Neutral Ratio** | 64.4% |
| **Long Recall** | 0.4963 |
| **Short Recall** | 0.6634 |
| **Long Precision** | 0.5886 |
| **Short Precision** | 0.5622 |

### Fold 1 (Valid: 2025-10)

| Metric | Value |
|--------|-------|
| **Macro F1** | 0.7136 |
| **Composite Score** | 0.0758 |
| **Neutral Ratio** | 66.2% |
| **Long Recall** | 0.6273 |
| **Short Recall** | 0.6634 |
| **Long Precision** | 0.6452 |
| **Short Precision** | 0.7077 |

### Fold 1 (Test: 2025-11)

| Metric | Value |
|--------|-------|
| **Macro F1** | 0.7405 |
| **Composite Score** | 0.1876 |
| **Neutral Ratio** | 64.1% |
| **Long Recall** | 0.6834 |
| **Short Recall** | 0.7127 |
| **Long Precision** | 0.6455 |
| **Short Precision** | 0.7410 |

---

## 4. Composite Score 公式 (NEW)

```
Composite = Macro_F1 × √(Long_P × Short_P) × (1 + 0.5 × Dir_Quality)
          - 8.0 × (P_aS_pL + P_aL_pS)
          - 3.0 × (P_aL_pN + P_aS_pN)
```

---

## 5. 18-Probability Evaluation System

### Fold 0 - Valid (2025-09)

**P(actual|predicted)**
| Predicted | Actual Long | Actual Neutral | Actual Short |
|------------|-------------|----------------|--------------|
| Long | 58.5% | 41.5% | 0.0% |
| Neutral | 11.4% | 77.1% | 11.6% |
| Short | 0.2% | 33.7% | 66.1% |

**P(predicted|actual)**
| Actual | Pred Long | Pred Neutral | Pred Short |
|--------|-----------|--------------|------------|
| Long | 60.4% | 42.9% | 0.0% |
| Neutral | 11.4% | 77.5% | 11.6% |
| Short | 0.2% | 32.1% | 62.9% |

### Fold 0 - Test (2025-10)

**P(actual|predicted)**
| Predicted | Actual Long | Actual Neutral | Actual Short |
|------------|-------------|----------------|--------------|
| Long | 56.2% | 43.3% | 0.5% |
| Neutral | 8.9% | 77.0% | 14.1% |
| Short | 0.2% | 40.9% | 58.9% |

**P(predicted|actual)**
| Actual | Pred Long | Pred Neutral | Pred Short |
|--------|-----------|--------------|------------|
| Long | 66.3% | 51.1% | 0.6% |
| Neutral | 8.9% | 76.7% | 14.0% |
| Short | 0.2% | 34.5% | 49.6% |

### Fold 1 - Valid (2025-10)

**P(actual|predicted)**
| Predicted | Actual Long | Actual Neutral | Actual Short |
|------------|-------------|----------------|--------------|
| Long | 70.8% | 29.2% | 0.0% |
| Neutral | 8.7% | 81.0% | 10.3% |
| Short | 0.0% | 35.5% | 64.5% |

**P(predicted|actual)**
| Actual | Pred Long | Pred Neutral | Pred Short |
|--------|-----------|--------------|------------|
| Long | 66.3% | 27.4% | 0.0% |
| Neutral | 8.9% | 83.0% | 10.5% |
| Short | 0.0% | 34.5% | 62.7% |

### Fold 1 - Test (2025-11)

**P(actual|predicted)**
| Predicted | Actual Long | Actual Neutral | Actual Short |
|------------|-------------|----------------|--------------|
| Long | 74.1% | 25.9% | 0.0% |
| Neutral | 8.6% | 83.3% | 8.2% |
| Short | 0.0% | 35.4% | 64.6% |

**P(predicted|actual)**
| Actual | Pred Long | Pred Neutral | Pred Short |
|--------|-----------|--------------|------------|
| Long | 71.3% | 24.9% | 0.0% |
| Neutral | 8.5% | 82.9% | 8.1% |
| Short | 0.0% | 37.5% | 68.3% |

---

## 6. 混淆矩阵

### Fold 0 - Test (2025-10)
```
         Predicted
         DOWN    NEUTRAL  UP
Actual DOWN   339      171       1
       NEUTRAL 261      1475      187
       UP     3       270       269
```

### Fold 1 - Test (2025-11)
```
         Predicted
         DOWN    NEUTRAL  UP
Actual DOWN   392      158       0
       NEUTRAL 137      1537      179
       UP     0       151       326
```

---

## 7. 模型存储

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
