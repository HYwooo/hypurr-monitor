# LSTM+CatBoost Walk-Forward 训练报告

**Generated**: 20260411T1211
**Train**: 2024-08 ~ 2025-08 (13个月)
**Valid**: 2025-09 | **Test**: 2025-10
**probability_threshold**: 0.12

---

## 1. 模型配置

| 参数 | 值 |
|------|-----|
| neutral_scale (K) | 1.3 |
| lookforward_bars | 6 |
| seq_len | 48 |
| hidden_dim | 128 |
| lstm_output_dim | 64 |
| probability_threshold | 0.12 |

---

## 2. 核心指标

### Valid (2025-09)

| 指标 | 值 |
|------|-----|
| **Macro F1** | 0.4316 |
| **Composite Score** | -1.8660 |
| **Neutral Ratio** | 79.7% |
| **Long Recall** | 0.1631 |
| **Short Recall** | 0.2808 |
| **Long Precision** | 0.3485 |
| **Short Precision** | 0.4548 |

### Test (2025-10)

| 指标 | 值 |
|------|-----|
| **Macro F1** | 0.4880 |
| **Composite Score** | -1.9414 |
| **Neutral Ratio** | 69.7% |
| **Long Recall** | 0.3118 |
| **Short Recall** | 0.3601 |
| **Long Precision** | 0.3095 |
| **Short Precision** | 0.5154 |

---

## 3. Composite Score 公式

```
Composite = Macro_F1 × √(Long_P × Short_P) × (1 + 0.5 × Dir_Quality)
          - 8.0 × (P_aS_pL + P_aL_pS)
          - 3.0 × (P_aL_pN + P_aS_pN)
```

---

## 4. 18-Probability Evaluation System

### Test (2025-10)

**P(actual|predicted)**
| Predicted | Actual Long | Actual Neutral | Actual Short |
|------------|-------------|----------------|--------------|
| Long | 51.5% | 46.5% | 2.0% |
| Neutral | 12.1% | 70.3% | 17.7% |
| Short | 14.1% | 54.9% | 31.0% |

**P(predicted|actual)**
| Actual | Pred Long | Pred Neutral | Pred Short |
|--------|-----------|--------------|------------|
| Long | 36.0% | 32.5% | 1.4% |
| Neutral | 13.0% | 75.8% | 19.0% |
| Short | 14.2% | 55.4% | 31.2% |

---

## 5. 混淆矩阵

### Test (2025-10)
```
         Predicted
         DOWN    NEUTRAL  UP
Actual DOWN   184      250       77
       NEUTRAL 166      1457      300
       UP     7       366       169
```

---

## 6. 模型存储

```
models/lstm_catboost/fold_0/
├── lstm_model.pt
├── catboost_model.cbm
└── scaler.joblib
```
