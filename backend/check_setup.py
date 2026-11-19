"""PitchPilot AI — read-only deployment readiness check.

Run from anywhere: python backend/check_setup.py
Requires: numpy, pandas, scikit-learn, xgboost, lightgbm, openpyxl.
Only open pickle files that you personally created and trust.
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATA = ROOT / "data"
FILES = ["stack_starter.pkl", "stack_substitute.pkl", "scaler.pkl", "label_encoders.pkl"]


def report(title: str, ok: bool, details: str = "") -> None:
    mark = "OK" if ok else "CHECK"
    print(f"[{mark}] {title}" + (f" — {details}" if details else ""))


def canonical_feature(name: object) -> str:
    # One known naming inconsistency between exported Stacking and XGBoost.
    return str(name).replace("Match_played", "Match played")


def validate_data() -> None:
    print("\n=== DATASET CHECK ===")
    csv_path = DATA / "player_data.csv"
    excel_path = DATA / "Cleaned_Football_Data.xlsx"

    if csv_path.is_file():
        try:
            columns = list(pd.read_csv(csv_path, nrows=0).columns)
            required = {"P_name", "C_name", "PS"}
            report("player_data.csv", required.issubset(columns),
                   f"{len(columns)} columns; required present: {sorted(required.intersection(columns))}")
        except Exception as exc:
            report("player_data.csv readable", False, str(exc))
    else:
        report("player_data.csv", False, "Missing (optional if XLSX source available)")

    if not excel_path.is_file():
        report("Cleaned_Football_Data.xlsx", False, "Missing")
        return

    try:
        with pd.ExcelFile(excel_path) as workbook:
            sheets = workbook.sheet_names
            report("Excel workbook", len(sheets) >= 3,
                   f"{len(sheets)} sheets: {', '.join(sheets[:5])}")
            if len(sheets) >= 3:
                columns = [set(pd.read_excel(workbook, sheet_name=s, nrows=0).columns)
                           for s in sheets[:3]]
                join_ok = all("M_id" in cols for cols in columns)
                report("Excel M_id join key", join_ok,
                       "Needed for notebook's three-sheet merge")
                union = set.union(*columns)
                required = {"P_name", "C_name", "PS"}
                report("Required player fields", required.issubset(union),
                       f"Found: {sorted(union.intersection(required))}")
    except Exception as exc:
        report("Excel readable", False, str(exc))


def main() -> int:
    print("PitchPilot AI — Backend Readiness Check")
    print(f"Project root: {ROOT}")
    print(f"Python: {sys.version.split()[0]} | scikit-learn: {sklearn.__version__}")
    print("NOTE: load only your own trusted .pkl files.\n")

    missing = [name for name in FILES if not (MODELS / name).is_file()]
    for name in FILES:
        report(name, name not in missing)
    if missing:
        print("\nModel files are missing; place all four in Football/models/.")
        validate_data()
        return 1

    objects = {}
    try:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            for name in FILES:
                with (MODELS / name).open("rb") as fp:
                    objects[name] = pickle.load(fp)
        version_warnings = [w for w in records if "Trying to unpickle estimator" in str(w.message)]
        if version_warnings:
            print("\n[VERSION WARNING] Saved sklearn objects may require scikit-learn 1.6.1.")
            print(f"Current environment produced {len(version_warnings)} compatibility warnings.")
    except Exception as exc:
        report("Model deserialization", False, f"{type(exc).__name__}: {exc}")
        return 1

    scaler = objects["scaler.pkl"]
    encoders = objects["label_encoders.pkl"]
    scaler_columns = [str(x) for x in getattr(scaler, "feature_names_in_", [])]
    report("Scaler input metadata", bool(scaler_columns),
           f"{len(scaler_columns)} columns")
    report("Categorical encoders", isinstance(encoders, dict),
           f"{len(encoders) if isinstance(encoders, dict) else 0} columns")

    print("\n=== MODEL FEATURE CONSISTENCY ===")
    for filename in ["stack_starter.pkl", "stack_substitute.pkl"]:
        model = objects[filename]
        outer = [str(x) for x in getattr(model, "feature_names_in_", [])]
        try:
            booster_names = [str(x) for x in model.named_estimators_["xgb"].get_booster().feature_names]
        except Exception as exc:
            report(f"{filename} XGBoost metadata", False, str(exc))
            continue

        exact = outer == booster_names
        aligned = [canonical_feature(x) for x in outer] == booster_names
        available = all(canonical_feature(c) in scaler_columns for c in outer)
        report(f"{filename} features", len(outer) == 15,
               f"{len(outer)} features")
        report("Outer vs XGBoost feature names", exact,
               "Exact match" if exact else "Mismatch detected; see names below")
        if not exact:
            for a, b in zip(outer, booster_names):
                if a != b:
                    print(f"        outer={a!r} | XGBoost={b!r}")
        report("Positional feature order", aligned,
               "Order agrees after known alias normalization" if aligned else "UNEXPECTED mismatch")
        report("All prediction features known to scaler", available)

        # Smoke test checks only whether the loaded estimator can execute.
        # Zero input is NOT a valid football player and output is not a prediction.
        if aligned and len(outer) == 15:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="X does not have valid feature names")
                    value = model.predict_proba(np.zeros((1, len(outer)), dtype=float))
                ok = (value.shape == (1, 2) and np.isfinite(value).all()
                      and (value >= 0).all() and (value <= 1).all())
                report("Estimator smoke test (synthetic input)", ok,
                       "Inference executes; not a real player prediction")
            except Exception as exc:
                report("Estimator smoke test", False, f"{type(exc).__name__}: {exc}")
        print()

    validate_data()
    print("\nSend the entire terminal output to review before implementing /api/predict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
