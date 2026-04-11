# LSTM+CatBoost Walk-Forward Training Report

**Generated**: 20260411T0849
**Training Period**: 2024-01 ~ 2025-12 (24 months)
**Valid**: 2026-01 | **Test**: 2026-02
**probability_threshold**: 0.65

---

## Model Configuration

| Parameter | Value |
|-----------|-------|
| neutral_scale (K) | 1.3 |
| lookforward_bars | 6 |
| seq_len | 48 |
| hidden_dim | 128 |
| lstm_output_dim | 64 |
| probability_threshold | 0.65 |

---

## Results Summary

| Metric | Value |
|--------|-------|
| **Macro F1** | 0.7804 |
| **Composite Score** | 0.3717 |
| **Neutral Ratio** | 60.9% |
| **Long Recall** | 0.7740 |
| **Short Recall** | 0.7886 |
| **Long Precision** | 0.6836 |
| **Short Precision** | 0.7462 |

---

## Composite Score Formula (NEW)

```
Composite = Macro_F1 × √(Long_P × Short_P) × (1 + 0.5 × Dir_Quality)
          - 8.0 × (P_aS_pL + P_aL_pS)
          - 3.0 × (P_aL_pN + P_aS_pN)
```

| Component | Value |
|-----------|-------|
| √(Long_P × Short_P) | 0.7142 |
| Dir_Quality | 0.7149 |
| P_aS_pL | 0.0000 |
| P_aL_pS | 0.0000 |
| P_aL_pN | 0.0635 |
| P_aS_pN | 0.0648 |

---

## Prediction Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| DOWN (Short) | 520 | 19.3% |
| NEUTRAL | 1637 | 60.9% |
| UP (Long) | 531 | 19.8% |

---

## Confusion Matrix

```
         Predicted
         DOWN    NEUTRAL  UP
Actual DOWN   388      104       0
       NEUTRAL 132      1427      168
       UP     0       106       363
```

---

## Model Storage

```
models/lstm_catboost/fold_0/
├── lstm_model.pt
├── catboost_model.cbm
└── scaler.joblib
```
