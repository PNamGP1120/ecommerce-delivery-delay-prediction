# Model Card — Delivery Delay Risk Model

## Model Summary

| Field | Value |
|---|---|
| Model family | XGBoost binary classifier |
| Selected configuration | `xgboost_03` |
| Prediction target | Late delivery |
| Prediction point | `order_approved_at` |
| Modeling grain | One row per order |
| Feature count | 41 |
| Development rows | 81,982 |
| Model artifact | `models/best_tuned_candidate.joblib` |

The model predicts a positive-class **risk score** for late delivery.

The score is suitable for ranking orders by risk, but Phase 5 showed that it is not sufficiently calibrated to be interpreted as a literal probability.

---

## Intended Use

The model is intended to support an operations workflow that prioritizes orders for additional review or intervention after order approval.

Possible interventions could include:

- checking fulfillment readiness;
- verifying seller/carrier handoff;
- prioritizing customer communication;
- increasing monitoring for high-risk orders.

The model should support human/operational prioritization rather than autonomously making high-impact customer decisions.

---

## Prediction Target

```text
late_delivery = (
    order_delivered_customer_date
    > order_estimated_delivery_date
)
```

Only orders with a valid target, prediction point and promised delivery window are included in the modeling population.

---

## Feature Availability

Only features available at or before `order_approved_at` may be used.

Post-outcome features are explicitly excluded.

Examples of forbidden inputs:

```text
order_delivered_carrier_date
order_delivered_customer_date
delivery_delay_days
review information
final order status
```

---

## Development Strategy

The project uses chronological evaluation.

Original split:

| Split | Rows | Late rate |
|---|---:|---:|
| Train | 67,515 | 9.02% |
| Validation | 14,467 | 5.38% |
| Test | 14,468 | 6.58% |

Phase 4B combines train + validation into an 81,982-row development dataset and evaluates 22 configurations using four expanding-window temporal folds.

Selection criterion:

```text
highest mean temporal-CV PR-AUC
```

with lower temporal PR-AUC variability as a tie-break.

---

## Selected Hyperparameters

```text
n_estimators      = 300
max_depth         = 5
learning_rate     = 0.03
min_child_weight  = 5
subsample         = 0.85
colsample_bytree  = 0.85
reg_lambda        = 2.0
scale_pos_weight  = automatic per training fold
```

---

## Performance

### Temporal cross-validation

```text
Mean PR-AUC = 0.2267
Std PR-AUC  = 0.1145
Mean ROC-AUC = 0.7198
```

### Pooled development OOF

```text
Prevalence   = 0.1008
PR-AUC       = 0.1907
PR-AUC lift  = 1.89x
ROC-AUC      = 0.6919
```

### Observed test diagnostic

```text
Prevalence   = 0.0658
PR-AUC       = 0.1080
PR-AUC lift  = 1.64x
ROC-AUC      = 0.6796
```

The observed test period is not described as a pristine final holdout because it was already observed during Phase 4A and influenced the decision to conduct Phase 4B.

---

## Thresholds

Development OOF operating points:

```text
Best F1:
threshold = 0.5650
precision = 19.55%
recall    = 44.12%
F1        = 27.09%

Recall >= 50%:
threshold = 0.5260
precision = 18.25%
recall    = 50.02%
F1        = 26.75%
```

These thresholds did not remain stable on the later observed test period.

Therefore the deployment treats thresholding as a configurable operational policy.

---

## Calibration

The model is not calibrated.

Phase 5 found systematic probability overestimation and a mean absolute calibration gap of approximately:

```text
0.2135
```

The production API therefore returns:

```json
"calibrated_probability": false
```

and uses `risk_score` rather than probability terminology.

---

## Explainability

Three methods are provided:

### Native XGBoost importance

Highlights model-specific split importance across transformed features.

### Permutation importance

The strongest raw feature on the observed test diagnostic was:

```text
promised_delivery_days
```

### SHAP

Global SHAP importance emphasizes:

```text
purchase_month_cos
promised_delivery_days
customer-state features
distance features
same_state_seller_share
approval_lag_hours
```

These explanations describe model behavior, not causal relationships.

---

## Drift

Important development-to-test PSI signals:

```text
promised_delivery_days ≈ 0.581  → high drift
total_freight          ≈ 0.238  → moderate drift
product category       ≈ 0.169  → moderate drift
```

Calendar features show very large PSI by construction because the comparison period contains later months and years.

`promised_delivery_days` deserves particular attention because it is both highly predictive and highly shifted.

---

## Known Limitations

- Historical dataset from one marketplace.
- Limited warehouse/carrier operational features.
- Strong temporal changes in late-delivery prevalence.
- Probability scores are poorly calibrated.
- Fixed thresholds are temporally unstable.
- The observed test period is not a pristine final holdout.
- Model performance weakens in the latest observed month.
- Production serving assumes exact upstream feature reproduction.
- Runtime telemetry is currently in-memory.

---

## Monitoring Recommendations

Monitor:

```text
request failures
latency
risk-score distribution
review rate
risk-band distribution
feature missingness
promised_delivery_days drift
realized positive prevalence
future PR-AUC / PR-AUC lift
future calibration
```

A future production version should connect these metrics to a persistent monitoring backend.

---

## Ethical / Operational Considerations

The model should be used to prioritize operational review, not to make punitive decisions about customers or sellers without additional evidence.

Segment-level metrics vary materially, so performance should be monitored across geography, time and operational conditions.

Model explanations should not be interpreted as proof of causal responsibility for delays.
