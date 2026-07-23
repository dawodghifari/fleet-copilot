# Retrieval evaluation

25 labeled questions; k=4; embeddings: all-MiniLM-L6-v2 (local).

## fixed (610 chunks)

hit@4: **0.56** — MRR: **0.46**

Missed questions:

- What are the most common ways the APS fails?
- Why is a missed APS failure so much more expensive than a false alarm?
- How many rows are in the training and test sets?
- What does a positive class label mean in this dataset?
- What is the formula for the official cost metric?
- Why is accuracy a misleading metric for this problem?
- What is the intended use of the failure prediction model?
- What test cost did the v1 model achieve?
- Does the model predict when a component will fail in the future?
- Why does the system produce so many false alarms?
- Why does it matter to log whether a flag was a false alarm?

## sections (19 chunks)

hit@4: **1.0** — MRR: **0.813**

