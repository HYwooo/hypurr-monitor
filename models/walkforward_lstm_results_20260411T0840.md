# LSTM+CatBoost Walk-Forward Training Report

**Generated**: 20260411T0840  
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

| Fold | Training Period | Valid | Test |
|------|-----------------|-------|------|
| 0 | 2024-01 ~ 2024-03 (3 months) | 2026-01 | 2026-02 |
| 1 | 2024-01 ~ 2024-06 (6 months) | 2026-01 | 2026-02 |
| 2 | 2024-01 ~ 2024-09 (9 months) | 2026-01 | 2026-02 |
| 3 | 2024-01 ~ 2024-12 (12 months) | 2026-01 | 2026-02 |

---

## 3. Key Metrics Summary

| Metric | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Average |
|--------|--------|--------|--------|--------|---------|
| **Macro F1** | 0.2618 | 0.3939 | 0.2608 | 0.6162 | **0.3832** |
| **Composite Score** | -1.0738 | -1.4414 | -1.0725 | -0.3646 | **-0.9881** |
| **Long Recall** | 0.0021 | 0.0874 | 0.0000 | 0.5181 | **0.1519** |
| **Short Recall** | 0.0000 | 0.1992 | 0.0000 | 0.5610 | **0.1900** |
| **Neutral Ratio** | 99.8% | 88.1% | 100.0% | 65.2% | **88.3%** |

---

## 4. Composite Score Formula (NEW)

```
Composite = Macro_F1 × √(Long_P × Short_P) × (1 + 0.5 × Dir_Quality)
          - 8.0 × (P_aS_pL + P_aL_pS)
          - 3.0 × (P_aL_pN + P_aS_pN)
```

| Component | Description |
|-----------|--------------|
| √(Long_P × Short_P) | Geometric mean of Long/Short Precision - both must be high |
| (1 + 0.5 × Dir_Quality) | Direction quality boost |
| 8.0 × (P_aS_pL + P_aL_pS) | Direction flip penalty (most severe) |
| 3.0 × (P_aL_pN + P_aS_pN) | False break Neutral penalty |

---

## 5. Prediction Distribution

| Fold | DOWN | NEUTRAL | UP |
|------|------|---------|-----|
| Fold 0 | 0 (0.0%) | 2682 (99.8%) | 6 (0.2%) |
| Fold 1 | 235 (8.7%) | 2369 (88.1%) | 84 (3.1%) |
| Fold 2 | 0 (0.0%) | 2688 (100.0%) | 0 (0.0%) |
| Fold 3 | 508 (18.9%) | 1752 (65.2%) | 428 (15.9%) |

---

## 6. Model Storage

```
models/lstm_catboost/
├── fold_0/
├── fold_1/
├── fold_2/
└── fold_3/
```

---

## 7. Key Findings

- Fold 3 (12 months training) shows best performance with Macro F1=0.6162
- Composite Score is negative due to strict penalties for false breaks
- Direction flip errors (P_aS_pL, P_aL_pS) are near zero across all folds
