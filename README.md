# Tower Brain - complete project (Person 1 + Person 2 + Person 3)

All results are SIMULATED. No real building data.

## Folder layout
| Path | Owner | What it is |
|---|---|---|
| `simulator.py` | Person 1 | 11-floor RC building simulator, fixed-schedule baseline, tuned timer |
| `data/timetable.csv`, `attendance_history.csv`, `baseline_*.csv` | Person 1 | Timetable and baseline results |
| `person2/model.py` | Person 2 | Random Forest 1-hour-ahead occupancy predictor (needs `person2/timetable.csv` beside it) |
| `person2/` (csv, json, png, joblib) | Person 2 | Outputs of `model.py`: accuracy report, feature importance, plots, trained model |
| `data/random_forest_model.joblib` | Person 2 | Copy of the trained model that the controller loads |
| `controller.py` | Person 3 | Rule-based AC controller + full before/after run |
| `llm_explainer.py` | Person 3 | Optional local-AI (Ollama) plain-English explainer |
| `outputs/` | Person 3 | Everything `controller.py` writes (before/after, decision log, plots) |
| `docs/Person3_Controller_Submission.md` | Person 3 | Technical write-up and report sections 6.4, 6.6, 6.7 |

## Run (Windows / Mac / Linux, Python 3.10+)
```
python -m venv venv
venv\Scripts\activate            (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python controller.py             # about 1-2 minutes, writes outputs/
```
It prints the before/after table. Expected summer result: Tower Brain 7,748 kWh vs 8,103 kWh baseline, peak 180.1 vs 211.7 kW.

## How Person 2's model is used
`controller.py` loads `data/random_forest_model.joblib` and calls it live every 15 minutes (1-hour-ahead forecast per floor).
Each decision-log row is tagged `[prediction: Person 2 model]`, `[... + fallback for gaps]` or `[fallback prediction]`.
Beyond the model's 1-hour horizon the controller uses the timetable.
- Retrain the model: `cd person2`, `python model.py`, then copy the new `random_forest_model.joblib` to `data/`
  (use the same scikit-learn version for training and running; the file was made with scikit-learn 1.9.1).
- Use a specific model: `python controller.py --model path/to/random_forest_model.joblib`
- Force the timetable-only fallback: `python controller.py --no-predictions`
- Alternative input: `data/occupancy_predictions.csv` with columns `timestamp,target_timestamp,floor_id,predicted_people`
  (Monday = 2026-12-14). Used only if no model file is found.

## Local AI explainer (optional)
1. Install Ollama (https://ollama.com/download) and start it.
2. `ollama pull llama3.2:3b` (8 GB RAM) or `ollama pull qwen2.5:7b` (16 GB RAM).
3. `python llm_explainer.py` -> `outputs/plain_english_report.md`; add `--no-llm` to run without Ollama.

## Note
`controller_summary.json` / `tower_brain_report.md` describe the summer run. Full three-scenario table: `outputs/before_after.csv`.
