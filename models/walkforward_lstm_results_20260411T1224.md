# LSTM+CatBoost Walk-Forward 训练报告

**Generated**: 20260411T1224
**Train**: 2024-09 ~ 2025-08 (12个月)
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
| **Macro F1** | 0.6220 |
| **TCS (New)** | 0.5949 |
| **Neutral Ratio** | 40.6% |
| **Long Recall** | 0.7163 |
| **Short Recall** | 0.8712 |
| **Long Precision** | 0.5549 |
| **Short Precision** | 0.4604 |

### Test (2025-10)

| 指标 | 值 |
|------|-----|
| **Macro F1** | 0.5491 |
| **TCS (New)** | 0.4990 |
| **Neutral Ratio** | 34.1% |
| **Long Recall** | 0.7491 |
| **Short Recall** | 0.8278 |
| **Long Precision** | 0.4486 |
| **Short Precision** | 0.4009 |

---

## 3. TCS 公式 (NEW)

```
TCS = (Long_P × Short_P)^0.5 + 0.2 × (Long_R + Short_R) / 2 - 2.0 × (P_aS_pL + P_aL_pS) - 0.3 × (P_aL_pN + P_aS_pN)
```

| 组成部分 | 说明 |
|---------|------|
| (Long_P × Short_P)^0.5 | 几何平均 - Long/Short Precision 都必须高 |
| + 0.2 × (Long_R + Short_R) / 2 | Recall 贡献 |
| - 2.0 × (P_aS_pL + P_aL_pS) | 方向翻转惩罚（最严厉） |
| - 0.3 × (P_aL_pN + P_aS_pN) | False Break 惩罚 |

---

## 4. 18-Probability Evaluation System

### Test (2025-10)

**P(actual|predicted)**
| Predicted | Actual Long | Actual Neutral | Actual Short |
|------------|-------------|----------------|--------------|
| Long | 40.1% | 59.1% | 0.8% |
| Neutral | 8.5% | 78.9% | 12.6% |
| Short | 0.2% | 54.9% | 44.9% |

**P(predicted|actual)**
| Actual | Pred Long | Pred Neutral | Pred Short |
|--------|-----------|--------------|------------|
| Long | 82.8% | 122.1% | 1.6% |
| Neutral | 4.5% | 41.7% | 6.7% |
| Short | 0.4% | 91.7% | 74.9% |

---

## 5. 混淆矩阵

### Test (2025-10)
```
         Predicted
         DOWN    NEUTRAL  UP
Actual DOWN   423      86       2
       NEUTRAL 624      802      497
       UP     8       128       406
```

---

## 6. 模型存储

```
models/lstm_catboost/fold_0/
├── lstm_model.pt
├── catboost_model.cbm
└── scaler.joblib
```
