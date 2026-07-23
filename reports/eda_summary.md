# EDA summary — APS Failure training set

Rows: 60,000. Features: 170. Positives: 1,000 (1.67%) — heavy class imbalance.

Columns with any missing values: 169 of 170. Worst column: `br_000` at 82.1%. Columns >70% missing: 7 (`br_000`, `bq_000`, `bp_000`, `bo_000`, `cr_000`, `ab_000`, `bn_000`).

Feature name prefixes: 107. Histogram-style groups (>=10 bins): {'ag': 10, 'ay': 10, 'az': 10, 'ba': 10, 'cn': 10, 'cs': 10, 'ee': 10} — these are binned distributions of single physical quantities (e.g. time spent in load ranges), useful for aggregate features later.

Baseline costs (train): predict all-negative = 500,000; predict all-positive = 590,000. Any useful model must land far below both; accuracy is meaningless at 1.7% prevalence.
