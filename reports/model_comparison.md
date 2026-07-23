# Model comparison — APS predictive maintenance

Validation = stratified 20% of official training set. Cost = 10*FP + 500*FN (official metric). Threshold tuned on validation only.

| model         |   val_cost |   val_cost_default_thr |   val_pr_auc |   val_recall_at_thr |   threshold |   fit_seconds |
|:--------------|-----------:|-----------------------:|-------------:|--------------------:|------------:|--------------:|
| xgboost       |       5800 |                  16870 |     0.906019 |               0.965 |  0.00855415 |          25.1 |
| random_forest |       5910 |                  38640 |     0.856213 |               0.97  |  0.0633333  |          21.6 |
| logreg        |      11140 |                  11450 |     0.72659  |               0.925 |  0.461631   |           6.7 |


## Final test-set result (xgboost)

Total cost **14,800** (FP=280 -> 2,800; FN=24 -> 12,000) at threshold 0.0086. 
Reference points: all-negative baseline costs 187,500 on this test set; the IDA 2016 challenge winner reported 9,920.
