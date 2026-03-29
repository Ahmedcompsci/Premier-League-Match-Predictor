"""
Premier League Match Predictor
================================
An end-to-end ML pipeline that predicts English Premier League match
outcomes (Home Win / Draw / Away Win) using historical match data.

Pipeline stages:
    1. Data ingestion    — load raw CSV match data
    2. Cleaning          — handle missing values, normalize types
    3. Feature engineering — rolling form, goal averages, head-to-head stats
    4. Model training    — Random Forest with class balancing
    5. Evaluation        — k-fold cross-validation + held-out test set
    6. Prediction        — predict outcome for any team matchup

Why Random Forest:
    - Handles non-linear feature interactions (form vs opponent quality)
    - Built-in feature importance (interpretable)
    - Robust to noisy data without heavy tuning
    - Ensemble method reduces overfitting vs single decision tree

Usage:
    # Run full pipeline with built-in demo data
    python predictor.py --demo

    # Run with your own CSV
    python predictor.py --data matches.csv

    # Predict a specific matchup
    python predictor.py --demo --predict "Man City" "Arsenal"

    # Save trained model
    python predictor.py --demo --save model.pkl

Install:
    pip install -r requirements.txt
"""

import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from typing import Optional

warnings.filterwarnings("ignore")  # suppress sklearn convergence warnings in demo

# ── Constants ─────────────────────────────────────────────────────────────────

OUTCOME_MAP = {
    "H": "Home Win",
    "D": "Draw",
    "A": "Away Win",
}

ROLLING_WINDOW = 5  # matches to look back for rolling form features

# ── 1. Data ingestion ─────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """
    Load raw match data from a CSV file.

    Expected columns:
        HomeTeam, AwayTeam  — team names (string)
        FTHG, FTAG          — full-time home/away goals (int)
        FTR                 — full-time result: 'H', 'D', or 'A'
        Date                — match date (parseable string)

    Returns a DataFrame sorted by date ascending.
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"[INFO] Loaded {len(df)} matches from {path}")
    return df


# ── 2. Demo data generator ────────────────────────────────────────────────────

def generate_demo_data(n_seasons: int = 3, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic EPL-style match data for demonstration.
    Simulates realistic outcome distributions (home win bias ~45%).

    Args:
        n_seasons: number of simulated 380-match seasons
        seed:      random seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    teams = [
        "Man City", "Arsenal", "Liverpool", "Chelsea", "Tottenham",
        "Man United", "Newcastle", "Aston Villa", "Brighton", "West Ham",
        "Brentford", "Fulham", "Crystal Palace", "Wolves", "Everton",
        "Nottm Forest", "Bournemouth", "Burnley", "Sheffield Utd", "Luton",
    ]

    rows = []
    # Each season: every team plays every other team home and away (380 matches)
    for season in range(n_seasons):
        season_start = pd.Timestamp(f"{2022 + season}-08-01")
        match_num = 0
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                # Simulate goals: home team has slight advantage
                hg = int(rng.poisson(1.6))
                ag = int(rng.poisson(1.1))
                result = "H" if hg > ag else ("A" if ag > hg else "D")
                rows.append({
                    "Date":     season_start + pd.Timedelta(days=match_num // 10),
                    "HomeTeam": home,
                    "AwayTeam": away,
                    "FTHG":     hg,
                    "FTAG":     ag,
                    "FTR":      result,
                })
                match_num += 1

    df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    print(f"[INFO] Generated {len(df)} synthetic matches ({n_seasons} seasons, {len(teams)} teams)")
    return df


# ── 3. Feature engineering ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create predictive features from raw match data.

    Features generated per match:
        home_form       — home team's win rate in last ROLLING_WINDOW games
        away_form       — away team's win rate in last ROLLING_WINDOW games
        home_goals_avg  — home team's avg goals scored in last N games
        away_goals_avg  — away team's avg goals scored in last N games
        home_concede_avg — home team's avg goals conceded in last N games
        away_concede_avg — away team's avg goals conceded in last N games
        h2h_home_wins   — home team's win rate in last 5 head-to-head matches

    Why rolling features:
        Raw team names carry no signal. Form-based features capture
        current momentum — a key predictor in football outcomes.
    """
    df = df.copy()

    # ── Per-team rolling stats ────────────────────────────────────────────────

    # Track each team's recent results across all matches (home + away)
    team_history: dict = {}  # team → list of (goals_scored, goals_conceded, won)

    home_form_list       = []
    away_form_list       = []
    home_goals_avg_list  = []
    away_goals_avg_list  = []
    home_concede_list    = []
    away_concede_list    = []

    for _, row in df.iterrows():
        ht, at = row["HomeTeam"], row["AwayTeam"]

        def get_rolling(team, stat_idx, n=ROLLING_WINDOW):
            """Return rolling mean of stat_idx from team's last n games."""
            history = team_history.get(team, [])[-n:]
            if not history:
                return 0.5  # neutral prior when no history exists
            return np.mean([h[stat_idx] for h in history])

        # Features BEFORE the current match (no data leakage)
        home_form_list.append(get_rolling(ht, 2))       # win rate
        away_form_list.append(get_rolling(at, 2))
        home_goals_avg_list.append(get_rolling(ht, 0))  # goals scored
        away_goals_avg_list.append(get_rolling(at, 0))
        home_concede_list.append(get_rolling(ht, 1))    # goals conceded
        away_concede_list.append(get_rolling(at, 1))

        # Update team history with this match's result (AFTER features computed)
        h_won = 1 if row["FTR"] == "H" else 0
        a_won = 1 if row["FTR"] == "A" else 0

        team_history.setdefault(ht, []).append((row["FTHG"], row["FTAG"], h_won))
        team_history.setdefault(at, []).append((row["FTAG"], row["FTHG"], a_won))

    df["home_form"]        = home_form_list
    df["away_form"]        = away_form_list
    df["home_goals_avg"]   = home_goals_avg_list
    df["away_goals_avg"]   = away_goals_avg_list
    df["home_concede_avg"] = home_concede_list
    df["away_concede_avg"] = away_concede_list
    df["form_diff"]        = df["home_form"] - df["away_form"]
    df["goal_diff_avg"]    = df["home_goals_avg"] - df["away_goals_avg"]

    # ── Head-to-head feature ─────────────────────────────────────────────────
    h2h_wins = []
    for _, row in df.iterrows():
        ht, at = row["HomeTeam"], row["AwayTeam"]
        # Past meetings between these two teams (before this match)
        past = df[
            (df["HomeTeam"] == ht) & (df["AwayTeam"] == at) &
            (df["Date"]     <  row["Date"])
        ].tail(5)
        if len(past) == 0:
            h2h_wins.append(0.5)  # neutral prior
        else:
            h2h_wins.append((past["FTR"] == "H").mean())
    df["h2h_home_wins"] = h2h_wins

    return df


# ── 4. Model training ─────────────────────────────────────────────────────────

FEATURE_COLS = [
    "home_form", "away_form",
    "home_goals_avg", "away_goals_avg",
    "home_concede_avg", "away_concede_avg",
    "form_diff", "goal_diff_avg",
    "h2h_home_wins",
]


def train(df: pd.DataFrame):
    """
    Train a Random Forest classifier on engineered features.

    Evaluation strategy:
        - 80/20 stratified train/test split (preserves class distribution)
        - 5-fold stratified cross-validation on training set
        - Final accuracy reported on held-out test set

    Returns trained model and label encoder.
    """
    df = engineer_features(df)

    X = df[FEATURE_COLS]
    y = df["FTR"]

    # ── Encode target labels (H/D/A → 0/1/2) ─────────────────────────────────
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # ── Train/test split — stratified to preserve class balance ───────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # ── Model — class_weight balances imbalanced outcomes (home win bias) ──────
    model = RandomForestClassifier(
        n_estimators=200,       # 200 trees — good accuracy/speed tradeoff
        max_depth=8,            # limit depth to reduce overfitting
        min_samples_leaf=5,     # require at least 5 samples per leaf
        class_weight="balanced",# compensates for home win bias in data
        random_state=42,
        n_jobs=-1,              # use all CPU cores
    )
    model.fit(X_train, y_train)

    # ── k-fold cross-validation (k=5) ────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

    # ── Held-out test set evaluation ─────────────────────────────────────────
    y_pred   = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n── Cross-Validation (5-fold) ────────────────────────────")
    print(f"  Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    print("\n── Held-Out Test Set ────────────────────────────────────")
    print(f"  Accuracy: {test_acc:.3f}")
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        zero_division=0
    ))

    print("\n── Feature Importance ───────────────────────────────────")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    for feat, imp in importances.sort_values(ascending=False).items():
        bar = "█" * int(imp * 50)
        print(f"  {feat:<22} {bar} {imp:.3f}")

    return model, le, df


# ── 5. Prediction ─────────────────────────────────────────────────────────────

def predict_match(
    model,
    le: LabelEncoder,
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> dict:
    """
    Predict the outcome of a single match using computed team features.

    Looks up each team's most recent rolling stats from the training data
    and uses them as the feature vector for prediction.

    Returns a dict with predicted outcome and class probabilities.
    """
    # Get the latest feature values for each team from processed data
    def get_team_features(team: str, role: str) -> dict:
        """Extract most recent rolling features for a team."""
        col_map = {
            "home": {
                "form":        "home_form",
                "goals_avg":   "home_goals_avg",
                "concede_avg": "home_concede_avg",
            },
            "away": {
                "form":        "away_form",
                "goals_avg":   "away_goals_avg",
                "concede_avg": "away_concede_avg",
            },
        }
        team_col = "HomeTeam" if role == "home" else "AwayTeam"
        recent = df[df[team_col] == team]
        if recent.empty:
            return {"form": 0.5, "goals_avg": 1.5, "concede_avg": 1.2}
        last = recent.iloc[-1]
        cols = col_map[role]
        return {
            "form":        last[cols["form"]],
            "goals_avg":   last[cols["goals_avg"]],
            "concede_avg": last[cols["concede_avg"]],
        }

    h = get_team_features(home_team, "home")
    a = get_team_features(away_team, "away")

    features = pd.DataFrame([{
        "home_form":        h["form"],
        "away_form":        a["form"],
        "home_goals_avg":   h["goals_avg"],
        "away_goals_avg":   a["goals_avg"],
        "home_concede_avg": h["concede_avg"],
        "away_concede_avg": a["concede_avg"],
        "form_diff":        h["form"] - a["form"],
        "goal_diff_avg":    h["goals_avg"] - a["goals_avg"],
        "h2h_home_wins":    0.5,  # neutral for unknown future matchup
    }])

    pred_enc  = model.predict(features)[0]
    pred_prob = model.predict_proba(features)[0]
    pred_label = le.inverse_transform([pred_enc])[0]

    result = {
        "home_team":   home_team,
        "away_team":   away_team,
        "prediction":  OUTCOME_MAP[pred_label],
        "probabilities": {
            OUTCOME_MAP[cls]: round(float(prob), 3)
            for cls, prob in zip(le.classes_, pred_prob)
        }
    }

    print(f"\n── Match Prediction ─────────────────────────────────────")
    print(f"  {home_team} vs {away_team}")
    print(f"  Predicted outcome: {result['prediction']}")
    print(f"  Probabilities:")
    for outcome, prob in result["probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"    {outcome:<12} {bar} {prob:.1%}")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Premier League Match Predictor")
    parser.add_argument("--demo",    action="store_true", help="Run with synthetic demo data")
    parser.add_argument("--data",    type=str,            help="Path to CSV match data file")
    parser.add_argument("--predict", nargs=2, metavar=("HOME", "AWAY"), help="Predict a matchup")
    parser.add_argument("--save",    type=str,            help="Save model to .pkl file")
    args = parser.parse_args()

    # Load data
    if args.demo:
        df_raw = generate_demo_data(n_seasons=3)
    elif args.data:
        df_raw = load_data(args.data)
    else:
        parser.print_help()
        return

    # Train
    model, le, df_processed = train(df_raw)

    # Predict a matchup if requested
    if args.predict:
        home, away = args.predict
        predict_match(model, le, df_processed, home, away)

    # Save model
    if args.save:
        joblib.dump({"model": model, "label_encoder": le}, args.save)
        print(f"\n[INFO] Model saved to {args.save}")


if __name__ == "__main__":
    main()
