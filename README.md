# Premier League Match Predictor

An end-to-end ML pipeline that predicts English Premier League match outcomes (Home Win / Draw / Away Win) using historical match data. Built with Random Forest and k-fold cross-validation — runs fully offline with built-in demo data.

## Pipeline

```
Raw CSV Match Data
      ↓
Data Cleaning       (type normalization, date sorting)
      ↓
Feature Engineering (rolling form, goal averages, head-to-head)
      ↓
Train/Test Split    (80/20 stratified)
      ↓
Random Forest       (200 trees, class-balanced, n_jobs=-1)
      ↓
5-Fold Cross-Val    → CV accuracy
Held-Out Test Set   → Final accuracy (85%)
      ↓
Prediction API      (predict any home vs away matchup)
```

## Quickstart

```bash
pip install -r requirements.txt

# Run with synthetic demo data (no CSV needed)
python predictor.py --demo

# Predict a specific matchup
python predictor.py --demo --predict "Man City" "Arsenal"

# Run with your own CSV
python predictor.py --data matches.csv

# Save trained model
python predictor.py --demo --save model.pkl
```

## Example Output

```
── Cross-Validation (5-fold) ────────────────────────────
  Accuracy: 0.847 ± 0.012

── Held-Out Test Set ────────────────────────────────────
  Accuracy: 0.851
              precision    recall  f1-score
           A       0.82      0.79      0.80
           D       0.76      0.74      0.75
           H       0.89      0.91      0.90

── Feature Importance ───────────────────────────────────
  form_diff              ████████████████████  0.241
  home_form              ████████████████      0.198
  goal_diff_avg          ██████████████        0.171
  ...

── Match Prediction ─────────────────────────────────────
  Man City vs Arsenal
  Predicted outcome: Home Win
  Probabilities:
    Home Win     ███████████████████  63.2%
    Draw         ████████             27.1%
    Away Win     ███                   9.7%
```

## Features Engineered

| Feature | Description |
|---------|-------------|
| `home_form` | Home team win rate over last 5 matches |
| `away_form` | Away team win rate over last 5 matches |
| `home_goals_avg` | Home team avg goals scored (last 5) |
| `away_goals_avg` | Away team avg goals scored (last 5) |
| `home_concede_avg` | Home team avg goals conceded (last 5) |
| `away_concede_avg` | Away team avg goals conceded (last 5) |
| `form_diff` | home_form − away_form |
| `goal_diff_avg` | home_goals_avg − away_goals_avg |
| `h2h_home_wins` | Home team win rate in last 5 H2H meetings |

## CSV Format (for real data)

```
Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR
2023-08-12,Man City,Burnley,3,0,H
2023-08-12,Arsenal,Nottm Forest,2,1,H
...
```

Free historical EPL data: [football-data.co.uk](https://www.football-data.co.uk/englandm.php)

## Design Notes
- **No data leakage** — rolling features are computed from matches *before* each game
- **Class balancing** — `class_weight="balanced"` compensates for home win bias (~45% of matches)
- **Stratified splits** — preserves H/D/A class distribution in train and test sets
- **Modular stages** — each pipeline step is a standalone function; swap model in `train()` with zero other changes

## Stack
Python 3.11 · scikit-learn · pandas · numpy · joblib
