# LSTM+CatBoost Walk-Forward 多 Fold 训练报告

**Generated**: 20260411T1333
**Valid**: 202509 | **Test**: 202510
**Folds**: 3-month, 6-month, 12-month, 24-month

---

## 1. TCS 公式 (v2 - Optimized for Macro F1)

```
TCS = 0.35×√(Long_P×Short_P) + 0.35×√(Long_R×Short_R) + 0.15×F1_N + 0.15×(F1_L+F1_S)/2 - 1.0×Flip - 0.2×FalseBreak
```

| 组成部分 | 说明 |
|---------|------|
| 0.35×√(Long_P×Short_P) | 几何平均 - Long/Short Precision 平衡 |
| 0.35×√(Long_R×Short_R) | 几何平均 - Long/Short Recall 平衡 |
| 0.15×F1_N | Neutral F1 贡献 |
| 0.15×(F1_L+F1_S)/2 | 方向 F1 平均 |
| -1.0×Flip | 方向翻转惩罚 (P_aS_pL + P_aL_pS) |
| -0.2×FalseBreak | False Break 惩罚 (P_aL_pN + P_aS_pN) |

---

## 2. 模型配置 (All Folds)

| 参数 | 值 |
|------|-----|
| neutral_scale (K) | 1.3 |
| lookforward_bars | 6 |
| seq_len | 48 |
| hidden_dim | 128 |
| lstm_output_dim | 64 |
| focal_gamma | 2.0 |
| class_weights | [2.5, 1.0, 2.5] |

---

## 3. Fold Summary

| Fold | Train Period | Months | Threshold | Train Neutral% |
|------|-------------|--------|-----------|---------------|
| 3-month | 202507~202509 | 3 | 0.55 | 62.9% |
| 6-month | 202504~202509 | 6 | 0.55 | 63.0% |
| 12-month | 202410~202509 | 12 | 0.65 | 63.6% |
| 24-month | 202310~202509 | 24 | 0.65 | 64.6% |

---

## 4. Test Metrics by Fold

### Macro F1 & TCS

| Fold | Valid Macro F1 | Valid TCS | Test Macro F1 | Test TCS | Test Neutral% |
|------|----------------|-----------|---------------|----------|---------------|
| 3-month | 0.5149 | 0.4191 | 0.4236 | 0.1378 | 72.3% |
| 6-month | 0.3668 | 0.2804 | 0.3927 | 0.0998 | 88.5% |
| 12-month | 0.6480 | 0.5583 | 0.6037 | 0.5097 | 66.7% |
| 24-month | 0.7908 | 0.7347 | 0.7658 | 0.7078 | 63.4% |

### Direction Metrics (Test)

| Fold | Long Recall | Short Recall | Long Precision | Short Precision | P(aS\|pL) | P(aL\|pS) |
|------|------------|-------------|----------------|----------------|----------|-----------|
| 3-month | 0.2657 | 0.2427 | 0.3165 | 0.3370 | 0.0951 | 0.0505 |
| 6-month | 0.0959 | 0.1781 | 0.3714 | 0.4527 | 0.1045 | 0.0643 |
| 12-month | 0.5609 | 0.4658 | 0.5400 | 0.5548 | 0.0000 | 0.0018 |
| 24-month | 0.7915 | 0.6830 | 0.7647 | 0.6610 | 0.0019 | 0.0036 |

---

## 5. Threshold Comparison by Fold

### 3-month

| Threshold | Neutral% | Macro F1 | TCS |
|-----------|---------|----------|-----|
| 0.55 | 72.3% | 0.4236 | 0.1378 |
| 0.60 | 86.0% | 0.3791 | 0.1373 |
| 0.65 | 94.7% | 0.3270 | 0.1840 |
| 0.70 | 97.7% | 0.2888 | 0.1215 |
| 0.75 | 99.0% | 0.2802 | 0.0515 |

### 6-month

| Threshold | Neutral% | Macro F1 | TCS |
|-----------|---------|----------|-----|
| 0.55 | 88.5% | 0.3927 | 0.0998 |
| 0.60 | 95.9% | 0.3399 | 0.1561 |
| 0.65 | 98.5% | 0.2997 | 0.2439 |
| 0.70 | 99.3% | 0.2824 | 0.0025 |
| 0.75 | 99.9% | 0.2644 | 0.0477 |

### 12-month

| Threshold | Neutral% | Macro F1 | TCS |
|-----------|---------|----------|-----|
| 0.55 | 49.4% | 0.6028 | 0.5408 |
| 0.60 | 58.2% | 0.6057 | 0.5239 |
| 0.65 | 66.7% | 0.6037 | 0.5097 |
| 0.70 | 76.2% | 0.5660 | 0.4645 |
| 0.75 | 87.2% | 0.4832 | 0.4051 |

### 24-month

| Threshold | Neutral% | Macro F1 | TCS |
|-----------|---------|----------|-----|
| 0.55 | 57.8% | 0.7513 | 0.7023 |
| 0.60 | 60.7% | 0.7596 | 0.7062 |
| 0.65 | 63.4% | 0.7658 | 0.7078 |
| 0.70 | 67.3% | 0.7646 | 0.7004 |
| 0.75 | 71.0% | 0.7594 | 0.6927 |

---

## 6. Best Folds Analysis

### Highest Test Macro F1
- **24-month**: Macro F1 = 0.7658, TCS = 0.7078

### Highest Test TCS
- **24-month**: TCS = 0.7078, Macro F1 = 0.7658

---

## 7. JSON Results

- Timestamp: 20260411T1333
- Path: models/walkforward_lstm_results_20260411T1333.json