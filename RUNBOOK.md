# Runbook — running Fleet Copilot locally

## Setup (once)

```bash
cd ~/Documents/Claude/Projects/Job\ Hunt/GitHub/fleet-copilot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_data.py        # downloads the UCI dataset (~55 MB)
```

`.env` must contain `ANTHROPIC_API_KEY=...` (already set up; gitignored).

## Reproduce the ML results

```bash
python scripts/run_eda.py                      # EDA report + figures
python scripts/train.py                        # v1: single-split baseline
for i in 0 1 2 3 4; do python scripts/train_v2_stage.py fold native  $i; done
for i in 0 1 2 3 4; do python scripts/train_v2_stage.py fold imputed $i; done
python scripts/train_v2_stage.py select        # OOF threshold + ablation
python scripts/train_v2_stage.py final         # single test-set evaluation
mlflow ui --backend-store-uri file:mlruns      # browse experiments
```

Expected: v1 test cost 14,800 → v2 test cost 11,010 (FP=401, FN=14).

## RAG retrieval evals (no API key needed)

```bash
python scripts/eval_rag.py     # chunking comparison -> reports/rag_eval.md
```

Expected: sections hit@4 = 1.00 / MRR ≈ 0.81; fixed ≈ 0.56 / 0.46.

## Agent demo (needs API key)

```bash
python scripts/demo_agent.py
# or your own question:
python scripts/demo_agent.py "Is vehicle 100 at risk? What should we check?"
```

Vehicle 42 is a confirmed APS failure in the test fleet (model predicts
P≈0.97); vehicle 0 is healthy (P≈0.00). Good demo pair.

## Web UI

```bash
cd src && uvicorn fleet_copilot.api:app --reload
# open http://127.0.0.1:8000
```

## Notes

- macOS: if a script segfaults where torch and xgboost load together, run
  with `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1` (demo_agent.py sets
  this automatically).

- First RAG run downloads the MiniLM embedding model (~90 MB) and builds
  the Chroma index in `chroma_db/` (gitignored).
- The trained model artifact is `models/best_model_v2.joblib` (gitignored;
  reproduce with the training commands above).
