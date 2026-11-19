PitchPilot AI - Step 02: FastAPI backend (HTML/CSS/JS frontend comes next)

1) Extract ZIP contents into your existing Football/ project ROOT.
   This ADDS backend/main.py and REPLACES backend/requirements.txt.
   The archive also includes the previous backend/check_setup.py.
   DO NOT overwrite, retrain, or publish your .pkl files without your permission.

2) From the Football/ root in Windows PowerShell:
   .\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
   .\.venv\Scripts\python.exe backend/check_setup.py
   .\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

3) Open:
   http://127.0.0.1:8000/health
   http://127.0.0.1:8000/api/teams
   http://127.0.0.1:8000/docs

4) In /docs > POST /api/predict > Try it out, enter:
   {
     "team": "Arsenal",
     "manager": "Mikel Arteta",
     "formation": "4-3-3",
     "bench": { "GK": 0, "DEF": 2, "MID": 2, "FWD": 1 }
   }

5) If you see a startup error about required columns in player_data.csv,
   that means this CSV differs from the newer notebook's merged dataset.
   Do not create dummy columns or fake predictions. Export the updated
   notebook's final_merged into data/player_data.csv after verifying the source.

6) Reply with the /health JSON and /api/predict status/output (or error) so
   the next step can connect your frontend without breaking existing files.

API notes:
- Feature-alias Match_played -> Match played is checked; models receive 15 features
  in exact saved order as numpy arrays, following a 35-column scaler.
- Predictions use saved Stacking models, not older XGBoost-only squad cell.
- Starting XI and bench are selected by position; invalid roster sizes cause 422.
- Manager names unseen during training are encoded as -1, with an API warning.
- Source is historical player_data.csv; this is not live match fixture prediction.
- The old notebook labels are minutes-played proxies; model scores are not
  calibrated probabilities of actual starting/substitute events.
- The backend has NOT been validated against your local CSV, which was not uploaded.
- Run python backend/check_setup.py first to verify saved artifacts.

Security / GitHub:
- Never load unknown .pkl files: pickle can execute code when loading.
- Add .venv/ to .gitignore. Review dataset rights before pushing raw data.
- GitHub Pages hosts static frontend, NOT FastAPI; backend needs separate host.
