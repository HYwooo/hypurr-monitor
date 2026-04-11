# LSTM+CatBoost Walk-Forward 训练报告

**Generated**: 20260411T1259
**Train**: 2024-09 ~ 2025-08 (12个月)
**Valid**: 202509 | **Test**: 202510
**probability_threshold**: 0.65

---

## 1. 模型配置

| 参数 | 值 |
|------|-----|
| neutral_scale (K) | 1.3 |
| lookforward_bars | 6 |
| seq_len | 48 |
| hidden_dim | 128 |
| lstm_output_dim | 64 |
| probability_threshold | 0.65 |

---

## 2. 核心指标

### Valid (202509)

| 指标 | 值 |
|------|-----|
| **Macro F1** | 0.5907 |
| **TCS (New)** | 0.5722 |
| **Neutral Ratio** | 70.0% |
| **Long Recall** | 0.5089 |
| **Short Recall** | 0.4077 |
| **Long Precision** | 0.6364 |
| **Short Precision** | 0.5146 |

### Test (202510)

| 指标 | 值 |
|------|-----|
| **Macro F1** | 0.5596 |
| **TCS (New)** | 0.4747 |
| **Neutral Ratio** | 66.6% |
| **Long Recall** | 0.5166 |
| **Short Recall** | 0.4090 |
| **Long Precision** | 0.5176 |
| **Short Precision** | 0.4604 |

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

### Valid (202509)

**P(actual|predicted)**
| Predicted | Actual Long | Actual Neutral | Actual Short |
|------------|-------------|----------------|--------------|
| Long | 51.5% | 48.3% | 0.2% |
| Neutral | 15.3% | 71.0% | 13.7% |
| Short | 0.0% | 36.4% | 63.6% |

**P(predicted|actual)**
| Actual | Pred Long | Pred Neutral | Pred Short |
|--------|-----------|--------------|------------|
| Long | 40.8% | 38.3% | 0.2% |
| Neutral | 17.1% | 79.8% | 15.4% |
| Short | 0.0% | 29.1% | 50.9% |

### Test (202510)

**P(actual|predicted)**
| Predicted | Actual Long | Actual Neutral | Actual Short |
|------------|-------------|----------------|--------------|
| Long | 46.0% | 53.1% | 0.9% |
| Neutral | 15.2% | 71.8% | 13.0% |
| Short | 0.2% | 48.1% | 51.8% |

**P(predicted|actual)**
| Actual | Pred Long | Pred Neutral | Pred Short |
|--------|-----------|--------------|------------|
| Long | 40.9% | 47.2% | 0.8% |
| Neutral | 15.7% | 73.9% | 13.4% |
| Short | 0.2% | 48.0% | 51.7% |

---

## 5. 混淆矩阵

### Valid (202509)
```
         Predicted
         DOWN    NEUTRAL  UP
Actual DOWN   212      199       1
       NEUTRAL 308      1433      276
       UP     0        164       287
```

### Test (202510)
```
         Predicted
         DOWN    NEUTRAL  UP
Actual DOWN   209      241       4
       NEUTRAL 301      1422      258
       UP     1        260       280
```

---

## 6. Threshold Comparison

| Threshold | Neutral% | Macro F1 | TCS |
|-----------|---------|----------|-----|
| 0.55 | 42.4% | 0.5605 | 0.4900 |
| 0.60 | 52.7% | 0.5697 | 0.4787 |
| 0.65 | 66.6% | 0.5596 | 0.4747 |
| 0.70 | 81.3% | 0.4876 | 0.4665 |
| 0.75 | 93.2% | 0.3790 | 0.5080 |

---

## 7. 模型存储

- Timestamp: 20260411T1259
- Config: models/walkforward_lstm_results_20260411T1259.json
