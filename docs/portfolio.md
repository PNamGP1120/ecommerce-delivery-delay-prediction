# Portfolio Notes

Use this document when presenting the project on a CV, GitHub profile, application form or interview.

---

## One-Line Project Description

Built an end-to-end machine learning system that predicts e-commerce delivery-delay risk at order approval time using leakage-safe feature engineering, temporal cross-validation, XGBoost, explainability, drift analysis, FastAPI and Docker.

---

## GitHub Repository Description

```text
End-to-end ML system for e-commerce delivery-delay risk prediction using temporal CV, XGBoost, SHAP, FastAPI, Docker, testing and drift monitoring.
```

---

## CV Version — Compact

**E-commerce Delivery Delay Prediction — Machine Learning / MLOps Project**

- Built an end-to-end binary-classification pipeline on 96K+ e-commerce orders with leakage-safe feature engineering and chronological evaluation.
- Evaluated 22 Logistic Regression, Random Forest, Gradient Boosting and XGBoost configurations using 4-fold expanding-window temporal cross-validation.
- Selected an XGBoost candidate with mean temporal-CV PR-AUC of **0.2267** and analyzed temporal drift, calibration, SHAP and permutation importance.
- Productionized inference with **FastAPI, Pydantic, Docker, model versioning, structured logging, batch prediction and monitoring hooks**.
- Implemented automated tests across feature engineering, modeling, temporal tuning, evaluation and serving; verified **23 tests passing**.

---

## CV Version — Very Short

```text
Built and Dockerized an end-to-end XGBoost delivery-risk prediction system on 96K+ e-commerce orders with temporal CV, SHAP, FastAPI and automated testing.
```

---

## LinkedIn / Portfolio Summary

This project explores what happens when a machine-learning problem is treated as a production system rather than only a notebook experiment.

The model predicts whether an e-commerce order is at risk of being delivered after its promised date, using only information available at order approval time.

I implemented the complete workflow: relational data understanding, leakage prevention, feature engineering, chronological splitting, imbalanced-class evaluation, expanding-window temporal cross-validation, hyperparameter tuning, XGBoost, SHAP, permutation importance, feature-drift analysis, FastAPI serving, Docker deployment and runtime monitoring hooks.

A key finding was that the strongest feature, the promised delivery window, also experienced substantial temporal drift. The final evaluation therefore documents where the model generalizes, where performance weakens, and why its output should be treated as an uncalibrated risk score rather than a literal probability.

---

## Interview Story — 60 Seconds

> I built an end-to-end delivery-delay prediction system using the Olist e-commerce dataset. The prediction happens at order approval time, so I first designed a leakage-safe feature contract and aggregated the relational tables to one row per order. Because the positive class was only about 8%, I used PR-AUC instead of accuracy, and because delivery behavior changed over time, I used chronological splits and expanding-window temporal cross-validation rather than a random split. I compared several model families and selected an XGBoost candidate. In evaluation, I found that the model still ranked risk above baseline on the latest period, but probability calibration and threshold stability were weak under drift. I therefore exposed the production output as a risk score, not a probability, and deployed it through FastAPI and Docker with validation, versioning, logging, tests and monitoring hooks.

---

## Interview Story — Technical Deep Dive

### Why not random split?

A random split mixes historical and future operating regimes.

The observed late-delivery rate changed materially over time, so a random split would produce an overly optimistic estimate of future performance.

### Why PR-AUC?

The final population has only about 8.1% late deliveries.

Accuracy can remain above 90% while detecting almost no late orders.

PR-AUC directly focuses on minority-class ranking quality.

### Why XGBoost?

It achieved the highest mean PR-AUC across the declared expanding-window temporal CV search.

The margin over other boosting candidates was small, so I describe it as the selected development candidate rather than claiming it is universally superior.

### What was the biggest model risk?

Temporal drift.

`promised_delivery_days` was both the strongest raw permutation feature and a high-PSI feature between development and the latest observed period.

### Why is the API response called risk score?

The Phase 5 calibration analysis showed systematic probability overestimation.

Calling a score of 0.60 a 60% chance of late delivery would therefore be misleading.

### Why not tune the threshold on test?

Because the test period had already been observed and should not become another validation set.

The project documents that limitation instead of repeatedly optimizing against the same future period.

---

## Strongest Technical Talking Points

1. **Prediction-time discipline**
   - Every feature is checked against the approval-time prediction boundary.

2. **Relational feature engineering**
   - Items, payments, products, sellers and geolocation are aggregated without breaking the one-order grain.

3. **Temporal validation**
   - Chronological split plus expanding-window cross-validation.

4. **Metric choice**
   - PR-AUC and PR-AUC lift rather than headline accuracy.

5. **Honest evaluation**
   - Documents poor calibration and threshold instability instead of hiding them.

6. **Explainability**
   - Native tree importance, permutation importance and SHAP.

7. **Drift**
   - PSI diagnostics linked back to temporal model behavior.

8. **Deployment**
   - FastAPI + Pydantic + Docker + model fingerprint + structured logs.

9. **Monitoring**
   - Score distribution, review rate, missingness, drift and later label-based evaluation.

10. **Reproducibility**
   - `uv.lock`, modular source code and automated tests.

---

## Possible Recruiter Questions

### “What was the business goal?”

Identify orders at elevated risk of late delivery early enough for operations to intervene.

### “What was the hardest problem?”

Maintaining realistic evaluation under temporal distribution shift.

### “What would you do with more data?”

Add warehouse/carrier state, fulfillment queue information, route/carrier performance, seller recent-history features and external logistics signals.

### “Would you deploy this exact model?”

As a portfolio prototype, yes. For a real business, I would first validate on a truly untouched future period, introduce persistent monitoring, improve calibration, and define intervention capacity/costs.

### “What would Phase 8 be?”

CI/CD, model registry, external metrics, scheduled drift/performance evaluation and controlled retraining/promotion.

---

## GitHub Topics

Suggested repository topics:

```text
machine-learning
xgboost
fastapi
docker
mlops
data-science
feature-engineering
time-series-validation
model-monitoring
shap
python
scikit-learn
```

---

## Suggested Pinned-Repo Description

```text
🚚 End-to-end delivery-delay risk prediction: temporal CV, XGBoost, SHAP, FastAPI, Docker & monitoring.
```

---

## Portfolio Project Status

```text
Data pipeline            ✓
EDA                      ✓
Feature engineering      ✓
Leakage prevention       ✓
Temporal validation      ✓
Model baselines          ✓
Hyperparameter tuning    ✓
XGBoost                  ✓
Explainability           ✓
Drift analysis           ✓
FastAPI                  ✓
Docker                   ✓
Automated tests          ✓
Documentation            ✓
```
