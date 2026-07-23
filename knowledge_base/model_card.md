# Model card — APS failure classifier

## Intended use

Rank trucks by probability that a reported fault is APS-related so that
workshop checks can be prioritized. Decision threshold is tuned for the
official 50:1 cost asymmetry, i.e. the model deliberately over-flags:
false alarms are 50x cheaper than missed failures.

## Training data

UCI "APS Failure at Scania Trucks" training set: 60,000 heavy-truck
operational records, 1.67% positive. Features are anonymized counters and
histogram bins of vehicle operating data. See dataset_documentation.md.

## Model

Gradient-boosted trees (XGBoost), class imbalance handled with
scale_pos_weight, 400 estimators, max depth 6. Preprocessing: columns
with >70% missingness dropped (7 columns), binary missing-indicators
added for columns with >=5% missingness, median imputation (v2 also
evaluates XGBoost's native missing-value routing). Experiments tracked
in MLflow; final configuration committed on out-of-fold cost before any
test-set contact.

## Performance

- v1: test cost 14,800 (FP=280, FN=24; recall 93.6% at threshold 0.0086).
- v2 (final): test cost 11,010 (FP=401, FN=14; recall 96.3%, PR-AUC 0.929).
  Changes: threshold selected on 5-fold out-of-fold probabilities instead
  of a single split; XGBoost native missing-value routing replaced median
  imputation after winning the ablation (OOF cost 37,920 vs 40,420).
- Reference: all-negative baseline 187,500; IDA 2016 winner 9,920.

## Limitations

- Feature names are anonymized, so per-feature explanations name opaque
  sensors (e.g. "ag_002"); the agent layer maps groups to their physical
  meaning only where documented (histogram groups).
- Trained on Scania trucks from before 2016; distribution shift to other
  fleets, vehicle generations, or telematics stacks is unquantified.
- The classifier predicts whether a *reported failure* is APS-related;
  it is not a remaining-useful-life model and does not forecast time to
  failure.
- Threshold assumes the 10:500 cost ratio; a fleet with different
  economics must re-tune it.
