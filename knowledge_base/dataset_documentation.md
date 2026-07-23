# APS Failure at Scania Trucks — dataset documentation

## Origin

Released by Scania CV AB for the Industrial Challenge at the 15th
International Symposium on Intelligent Data Analysis (IDA) 2016, and
hosted in the UCI Machine Learning Repository. The data was collected
from heavy Scania trucks in everyday use.

## Structure

- Training set: 60,000 rows — 59,000 negative, 1,000 positive (1.67%).
- Test set: 16,000 rows — 15,625 negative, 375 positive.
- 170 anonymized numeric features plus a binary class label.
- The positive class means the component failure is related to the APS;
  the negative class means the failure relates to some other system.

## Feature anonymity and histogram bins

All attribute names are anonymized for proprietary reasons (e.g. aa_000,
ag_001). Features are a mix of single numerical counters and histogram
bins. Seven identifiable histogram groups each have 10 bins (ag, ay, az,
ba, cn, cs, ee) — a histogram represents one physical quantity (such as
time spent in different ambient temperature or load ranges) binned into
operating ranges. Bin values within a group are therefore related and can
be aggregated (sums, distribution shape) for feature engineering.

## Missing values

Missing values are encoded as the string 'na'. 169 of the 170 features
have at least some missingness; the worst column (br_000) is 82% missing,
and 7 columns exceed 70%. Missingness is often systematic (a counter not
present on a vehicle variant), so whether a value is missing can itself
carry signal — this project adds binary missing-indicator features for
columns with at least 5% missingness.

## Official cost metric

Total cost = 10 x FP + 500 x FN.

- False positive: an unnecessary workshop check — cost 10.
- False negative: a missed APS failure that may cause a breakdown —
  cost 500.

Accuracy is meaningless at 1.7% prevalence (predicting "no failure" for
every truck reaches 98.3% accuracy but costs 187,500 on the test set).
The IDA 2016 challenge winner reported a total test cost of 9,920 with
9 missed failures and 542 false alarms.

## This project's results

- v1 (single validation split, median imputation + missing indicators,
  XGBoost, threshold tuned on the split): test cost 14,800.
- v2 (5-fold out-of-fold threshold selection and imputation ablation):
  see reports/final_result_v2.json for the committed numbers.
