# Credit Card Fraud Detection

**A model that never predicts fraud is 99.83% accurate. That is the whole problem.**

492 of 284,807 transactions in this dataset are fraudulent — 0.17%. Any classifier can
score near-perfect accuracy by calling everything legitimate, so this project compares
five approaches on the metrics that actually matter when the positive class is rare:
precision, recall, F1 and AUC. On top of the notebooks sits a Streamlit app that scores
an uploaded batch of transactions and hands back a ranked risk list.

---

## The dataset

Kaggle's [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
set — two days of European card transactions from September 2013.

| | |
|---|---|
| Rows | 284,807 |
| Frauds | 492 (0.17%) |
| Features | `Time`, `Amount`, and `V1`–`V28` — PCA components, anonymised for privacy |

`creditcard.csv` is not in this repo (it is ~150 MB). Download it from Kaggle and drop
it beside `app.py` before running the app.

## Models compared

Each notebook is self-contained: load, scale, split, train, evaluate, plot the
confusion matrix, ROC and precision–recall curves.

| Model | Evaluated on | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| **XGBoost** (`scale_pos_weight` for imbalance) | full imbalanced test set — 56,962 rows, 98 frauds | 0.872 | 0.837 | 0.854 | 0.970 |
| **SVM** (RBF) | undersampled subset | 0.989 | 0.861 | 0.921 | 0.976 |
| **Decision Tree** (`max_depth=8`) | balanced 492/492 subset | 0.875 | 0.929 | 0.901 | — |
| **Decision Tree** (entropy, unpruned) | full imbalanced test set | 0.76 | 0.77 | 0.76 | — |
| **K-Nearest Neighbours** (k=5) | full imbalanced test set | 0.87 | 0.78 | 0.82 | — |
| **Logistic Regression** | balanced 492/492 subset | — | — | — | — |

Logistic regression reached 93.4% accuracy on the balanced subset (94.9% on train) —
and that row is deliberately left half-empty, because accuracy on a 50/50 subset says
almost nothing about behaviour on live traffic.

**Reading the table honestly:** the rows are not all the same experiment. Models
evaluated on the full imbalanced test set faced the real 0.17% base rate; those
evaluated on a `RandomUnderSampler` subset faced a rebalanced one, which flatters
precision. XGBoost's numbers are the ones to trust for production behaviour — it was
trained on the full data with the class imbalance handled by weighting rather than by
throwing away 99.8% of the legitimate transactions.

**The precision/recall trade is the actual decision.** Recall is the fraud you catch;
precision is how often you cry wolf. A bank that misses a third of its fraud has a loss
problem, one that blocks thousands of real customers has a churn problem, and the
threshold between them is a business call, not a modelling one — which is why the app
exposes fraud *probability*, not just a label.

## The app

`app.py` is a Streamlit interface over the exported model:

- upload a CSV of transactions (`Time`, `V1`–`V28`, `Amount`)
- get a predicted class and a fraud probability for every row
- totals, average risk, and a probability distribution across the batch
- if the file includes the true `Class` column, it scores itself: accuracy, precision,
  recall, F1 and a confusion matrix
- filter to fraudulent, legitimate, or high/medium-risk bands, and export the scored
  rows back out as CSV

## Running it

```bash
pip install streamlit pandas numpy scikit-learn xgboost joblib plotly
# place creditcard.csv next to app.py
streamlit run app.py
```

## Structure

```
Credit Card Fraud Detection - Data Visualization.ipynb   class balance, distributions
Credit Card Fraud Detection - Logistic Regression.ipynb  baseline
Credit Card Fraud Detection - K-Nearest Neighbor.ipynb   k sweep + evaluation
Credit Card Fraud Detection - Decision Tree.ipynb        entropy tree, depth-limited tree
Credit Card Fraud Detection - Support Vector Machines.ipynb  RBF SVM on undersampled data
Credit Card Fraud Detection- XGBOOST.ipynb               final model + exported artefact
app.py                                                   Streamlit scoring interface
best_fraud_model.joblib                                  exported XGBoost classifier
best_fraud_model_no_svm.joblib                           logistic-regression baseline
best_fraud_model_with_xgboost.joblib                     logistic-regression baseline
```

## Known limitations

- The scaler is re-fit from `creditcard.csv` at startup instead of being exported
  alongside the model, so the app needs the training data present to score anything.
- Undersampling discards most of the legitimate transactions. It makes the models
  comparable on a small balanced set, but the full-data XGBoost run is the honest
  measure.
- `Time` is kept as a raw feature; on a two-day capture it carries little signal.

---

Tech: Python · scikit-learn · XGBoost · imbalanced-learn · pandas · Streamlit · Plotly

More of my work at [harshgupta.co.in](https://www.harshgupta.co.in)
