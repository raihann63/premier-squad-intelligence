"""PitchPilot AI FastAPI inference server.

Uses the saved notebook preprocessing: train-fitted LabelEncoders -> the 35-feature
StandardScaler -> per-target 15-feature StackingClassifier. Model file format is
pickle, so never load model files from an untrusted source.

Note: these models learn historical >=60 / 1..59 playing-minutes *proxies*, not
verified real Starting XI / substitution events. Scores are not calibrated.
"""

from __future__ import annotations

import os
import pickle
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATASET_PATH = ROOT / "data" / "player_data.csv"

POSITION_NAMES = {"G": "GK", "D": "DEF", "M": "MID", "F": "FWD"}
FORMATION_MAP = {
    "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
    "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
    "4-2-3-1": {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1},
    "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
    "3-4-3": {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},
    "5-3-2": {"GK": 1, "DEF": 5, "MID": 3, "FWD": 2},
}
TEAM_FIX_MAP = {
    "Leiceste": "Leicester",
    "Leicester": "Leicester",
    "Man Utd ": "Man Utd",
    "Man Utd": "Man Utd",
    "Man City": "Man City",
}


def canonical_feature(name: str) -> str:
    # The saved model's outer name differs only for this known feature.
    return "Match played" if str(name) == "Match_played" else str(name)


def clean_team_name(value: object) -> str:
    name = str(value).strip()
    return TEAM_FIX_MAP.get(name, name)


def position_category(value: object) -> str:
    return POSITION_NAMES.get(str(value).strip().upper(), "OTHER")


class BenchSlots(BaseModel):
    GK: int = Field(default=0, ge=0, le=4)
    DEF: int = Field(default=2, ge=0, le=9)
    MID: int = Field(default=2, ge=0, le=9)
    FWD: int = Field(default=1, ge=0, le=9)
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def check_total(self):
        count = self.GK + self.DEF + self.MID + self.FWD
        if not 1 <= count <= 9:
            raise ValueError("Total bench positions must be between 1 and 9")
        return self


class PredictRequest(BaseModel):
    team: str = Field(min_length=1, max_length=100)
    manager: str = Field(min_length=1, max_length=100)
    formation: Literal["4-3-3", "4-4-2", "4-2-3-1", "3-5-2", "3-4-3", "5-3-2"] = "4-3-3"
    bench: BenchSlots = Field(default_factory=BenchSlots)
    model_config = ConfigDict(extra="forbid")

    @field_validator("team", "manager")
    @classmethod
    def trim(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Must not be blank")
        return value


class Engine:
    def __init__(self):
        self.scaler = self._load_model("scaler.pkl")
        self.encoders = self._load_model("label_encoders.pkl")
        self.starter = self._load_model("stack_starter.pkl")
        self.substitute = self._load_model("stack_substitute.pkl")

        self.scaler_features = list(map(str, getattr(self.scaler, "feature_names_in_", [])))
        if len(self.scaler_features) != 35 or not isinstance(self.encoders, dict):
            raise RuntimeError("Saved scaler/encoders do not match the verified notebook artifacts")
        self.model_features = {}
        for label, model in (("starter", self.starter), ("substitute", self.substitute)):
            outer = list(map(str, getattr(model, "feature_names_in_", [])))
            try:
                xgb_feature_names = list(map(str, model.named_estimators_["xgb"].get_booster().feature_names))
            except (KeyError, AttributeError, TypeError) as exc:
                raise RuntimeError(f"{label}: missing XGBoost component metadata") from exc
            mapped = [canonical_feature(name) for name in outer]
            if len(mapped) != 15 or mapped != xgb_feature_names:
                raise RuntimeError(f"{label}: model feature order unexpectedly changed")
            unknown = sorted(set(mapped) - set(self.scaler_features))
            if unknown:
                raise RuntimeError(f"{label}: feature(s) absent from scaler: {unknown}")
            if not np.array_equal(model.classes_, [0, 1]):
                raise RuntimeError(f"{label}: binary class labels not [0, 1]")
            self.model_features[label] = mapped

        if not DATASET_PATH.is_file():
            raise RuntimeError(f"Dataset missing: {DATASET_PATH}")
        self.data = pd.read_csv(DATASET_PATH, encoding="utf-8-sig", low_memory=False)
        self.data.columns = self.data.columns.str.strip()
        mandatory = set(self.scaler_features) - {"Managers"}
        mandatory |= {"C_name", "PS", "P_name"}
        missing = sorted(mandatory - set(self.data.columns))
        if missing:
            raise RuntimeError(
                f"player_data.csv is missing required columns: {missing}. "
                "Use data exported from the updated notebook's final_merged."
            )
        self.data = self.data.dropna(subset=["C_name", "P_name", "PS"]).copy()
        self.data["_team"] = self.data["C_name"].map(clean_team_name)
        self.data["_name"] = self.data["P_name"].astype(str).str.strip()
        self.data["_position"] = self.data["PS"].map(position_category)
        self.data["_sort_date"] = self._sort_dates(self.data["Date"])
        self.teams = sorted(self.data["_team"].dropna().unique().tolist())

    @staticmethod
    def _load_model(filename: str):
        path = MODELS_DIR / filename
        if not path.is_file():
            raise RuntimeError(f"Required model not found: {path}")
        # Pickle is only safe for artifacts created by the user and kept locally.
        with path.open("rb") as f:
            return pickle.load(f)

    @staticmethod
    def _sort_dates(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().all():
            return numeric
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        return pd.Series(np.where(parsed.notna(), parsed.astype("int64") // 10**9, -1), index=series.index)

    def team_rows(self, team: str) -> pd.DataFrame:
        matched = clean_team_name(team)
        rows = self.data[self.data["_team"].str.casefold() == matched.casefold()].copy()
        if rows.empty:
            raise HTTPException(status_code=404, detail=f"Team '{team}' not found in dataset")
        rows = rows.sort_values("_sort_date", ascending=False, kind="stable")
        rows = rows.drop_duplicates(subset=["_name"], keep="first").reset_index(drop=True)
        return rows

    def historical_managers(self, team: str) -> dict:
        """Club-scoped historical manager options, latest observation first.

        Suggestions must come from this team's dataset; never infer a manager
        from global encoder labels (which have no club association).
        """
        club = clean_team_name(team)
        matches = self.data.loc[self.data["_team"].str.casefold() == club.casefold()]
        if matches.empty:
            raise HTTPException(status_code=404, detail=f"Team '{team}' not found in dataset")
        display_team = str(matches.iloc[0]["_team"])

        names = []
        if "Managers" in matches.columns:
            ordered = matches.sort_values("_sort_date", ascending=False, kind="stable")
            seen = set()
            for value in ordered["Managers"].dropna():
                name = str(value).strip()
                if not name or name.casefold() in {"nan", "none", "null", "missing"}:
                    continue
                key = name.casefold()
                if key not in seen:
                    names.append(name)
                    seen.add(key)
        return {
            "team": display_team,
            "count": len(names),
            "managers": names,
            "suggested": names[0] if names else None,
            "source": "historical player_data.csv",
        }

    def transform_inputs(self, rows: pd.DataFrame, manager: str) -> pd.DataFrame:
        data = rows.loc[:, self.scaler_features].copy()
        data["Managers"] = manager
        # Exactly matches notebook encoding on training data, including -1 for unknowns.
        for column, encoder in self.encoders.items():
            if column not in data.columns:
                continue
            original = data[column].fillna("missing").astype(str)
            index = {str(label): i for i, label in enumerate(encoder.classes_)}
            data[column] = original.map(index).fillna(-1).astype("int64")
        for column in data.columns:
            if column == "Date" and not pd.api.types.is_numeric_dtype(data[column]):
                # Notebook converts datetime values to Unix seconds before scaling.
                dates = pd.to_datetime(data[column], errors="coerce", utc=True)
                if dates.isna().any():
                    raise HTTPException(status_code=422, detail="Invalid Date values in player_data.csv")
                data[column] = dates.astype("int64") // 10**9
            data[column] = pd.to_numeric(data[column], errors="coerce")
        invalid = data.columns[data.isna().any()].tolist()
        if invalid:
            raise HTTPException(
                status_code=422,
                detail=f"Dataset contains missing/non-numeric inference features: {invalid}. Re-export final_merged.",
            )
        values = self.scaler.transform(data.loc[:, self.scaler_features])
        return pd.DataFrame(values, columns=self.scaler_features, index=rows.index)

    @staticmethod
    def stats_for_player(data: pd.DataFrame, team: str, player: str) -> dict:
        # Use team-scoped history, rather than combining name collisions across clubs.
        history = data.loc[(data["_team"] == team) & (data["_name"] == player)].copy()
        history = history.sort_values("_sort_date", ascending=False, kind="stable")

        def sum_num(column: str) -> int | None:
            if column not in history:
                return None
            return int(pd.to_numeric(history[column], errors="coerce").fillna(0).sum())

        rating_col = "Rating" if "Rating" in history else None
        rating = pd.to_numeric(history[rating_col], errors="coerce") if rating_col else pd.Series(dtype=float)
        rating = rating.dropna()
        return {
            "goals": sum_num("Goals"),
            "assists": sum_num("Assist"),
            "average_rating": round(float(rating.mean()), 2) if len(rating) else None,
            "last_three_ratings": [round(float(x), 1) for x in rating.head(3).iloc[::-1]],
            "historical_match_rows": int(len(history)),
        }

    def predict(self, request: PredictRequest) -> dict:
        rows = self.team_rows(request.team)
        team = str(rows.iloc[0]["_team"])
        usable = rows[rows["_position"] != "OTHER"].copy().reset_index(drop=True)
        slots = FORMATION_MAP[request.formation]
        bench = request.bench.model_dump()
        needed = {k: slots[k] + bench[k] for k in slots}
        available = usable["_position"].value_counts().to_dict()
        deficit = {k: {"required": v, "available": int(available.get(k, 0))}
                   for k, v in needed.items() if available.get(k, 0) < v}
        if deficit:
            raise HTTPException(
                status_code=422,
                detail={"message": "Not enough eligible players for this formation and bench", "positions": deficit},
            )

        scaled = self.transform_inputs(usable, request.manager)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            starter_x = scaled[self.model_features["starter"]].to_numpy(dtype=float)
            sub_x = scaled[self.model_features["substitute"]].to_numpy(dtype=float)
            starter_probs = self.starter.predict_proba(starter_x)[:, 1]
            sub_probs = self.substitute.predict_proba(sub_x)[:, 1]
        if (not np.isfinite(starter_probs).all() or not np.isfinite(sub_probs).all()):
            raise HTTPException(status_code=500, detail="Model produced non-finite scores")

        usable["starter_score"] = starter_probs
        usable["substitute_score"] = sub_probs
        result_starters = []
        result_subs = []
        for cat in ("GK", "DEF", "MID", "FWD"):
            candidates = usable[usable["_position"] == cat].sort_values("starter_score", ascending=False)
            result_starters.extend(candidates.head(slots[cat]).index.tolist())
        remaining = usable.drop(index=result_starters)
        for cat in ("GK", "DEF", "MID", "FWD"):
            candidates = remaining[remaining["_position"] == cat].sort_values("substitute_score", ascending=False)
            result_subs.extend(candidates.head(bench[cat]).index.tolist())

        def player_record(idx: int, group: Literal["starter", "substitute"]) -> dict:
            row = usable.loc[idx]
            score = row["starter_score"] if group == "starter" else row["substitute_score"]
            name = str(row["_name"])
            return {
                "name": name,
                "position": str(row["_position"]),
                "raw_position": str(row["PS"]).strip(),
                "model_score": round(float(score), 4),
                "model_score_percent": round(100 * float(score), 1),
                **self.stats_for_player(self.data, team, name),
            }

        unknown_manager = request.manager not in set(map(str, self.encoders["Managers"].classes_))
        return {
            "team": team,
            "manager": request.manager,
            "formation": request.formation,
            "formation_positions": slots,
            "bench_positions": bench,
            "available_players": len(rows),
            "starter_count": len(result_starters),
            "substitute_count": len(result_subs),
            "starting_xi": [player_record(idx, "starter") for idx in result_starters],
            "substitutes": [player_record(idx, "substitute") for idx in result_subs],
            "warnings": (["Manager not seen by the training encoder; encoded as -1."] if unknown_manager else []),
            "disclaimer": (
                "Model scores reflect historical minutes-played proxy labels (>=60 min / 1-59 min), "
                "not official Starting XI/Substitution events or calibrated future-match probabilities."
            ),
        }


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return Engine()


app = FastAPI(title="PitchPilot AI API", version="0.2.0", docs_url="/docs")
origins = os.getenv(
    "PITCHPILOT_FRONTEND_ORIGINS",
    "http://127.0.0.1:5500,http://localhost:5500,http://127.0.0.1:5501,http://localhost:5501",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in origins.split(",") if x.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

@app.get("/api/overview")
@app.get("/health")
def health():
    engine = get_engine()
    return {
        "status": "ok",
        "models": ["stack_starter", "stack_substitute"],
        "teams": len(engine.teams),
        "data_rows": len(engine.data),
    }


@app.get("/api/teams")
def teams():
    engine = get_engine()
    return {"count": len(engine.teams), "teams": engine.teams, "formations": list(FORMATION_MAP)}


@app.get("/api/managers")
def managers(team: str):
    """Return known historical managers only for the requested club."""
    return get_engine().historical_managers(team)


@app.get("/api/players")
def players(team: str):
    engine = get_engine()
    rows = engine.team_rows(team)
    team_name = str(rows.iloc[0]["_team"])
    return {
        "team": team_name,
        "count": len(rows),
        "players": [
            {"name": str(r["_name"]), "position": str(r["_position"])}
            for _, r in rows.iterrows()
        ],
    }


@app.post("/api/predict")
def predict(request: PredictRequest):
    return get_engine().predict(request)
