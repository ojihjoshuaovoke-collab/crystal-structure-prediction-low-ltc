"""
Train surrogate models to predict lattice thermal conductivity (LTC) and
heat capacity from real DFT-calculated ground truth data.

Data source: data/combined_features.csv -- 4706 real crystal structures
with descriptors (atomic weight, volume, density, bond length, electronic
structure properties) and REAL DFT-calculated targets:
    - Log(DFT LTC)     -- already log10-scaled by the original pipeline,
                           matching the log-scale training approach used
                           in both source papers (Ojih et al. 2023, 2024)
    - DFT Heat Capcity -- raw units (J/mol-K)

This supersedes the earlier version of this project, which trained on
ALIGNN's own *predictions* on 646 newly-discovered structures (a copy of
a model's output, not ground truth). This version trains directly on the
DFT-calculated values themselves -- the same target the original papers'
GNNs were trained on.

Two separate Random Forest models are trained (one per property) since
they have different scales and the dataset provides both independently.

Usage:
    python train_real_model.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

FEATURE_COLS = [
    "Total Weight",
    "volume",
    "Average Weight",
    "Number Density",
    "Mass Density",
    "Bond Length",
    "Number of Atom",
    "Number of unpaired Electron",
    "Average Number of Electron",
    "Maximun Principle quantum number",
    "Pauling Electronegativity",
]

LTC_TARGET = "Log(DFT LTC)"       # already log10-scaled
HC_TARGET = "DFT Heat Capcity"    # raw J/mol-K


def train_one(df, target_col, out_path, log_already=False):
    X = df[FEATURE_COLS].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=0, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"\n=== {target_col} ===")
    print(f"Test R^2: {r2:.3f}")
    print(f"Test MAE: {mae:.3f}" + (" (log10 scale)" if log_already else ""))
    if log_already:
        print(f"  -> predictions on average within {10**mae:.2f}x of true value")

    importances = sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1])
    print("Feature importances:")
    for feat, imp in importances:
        print(f"  {feat:35s} {imp:.3f}")

    model.fit(X, y)  # refit on all data for deployment
    bundle = {"model": model, "features": FEATURE_COLS, "target": target_col, "log_scale": log_already, "test_r2": r2, "test_mae": mae}
    joblib.dump(bundle, out_path)
    print(f"Saved to {out_path}")
    return bundle


def train_all(data_path="data/combined_features.csv", ltc_out="model_ltc_real.joblib", hc_out="model_hc_real.joblib"):
    """Callable entry point (used by app.py for first-run auto-training)."""
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} real structures with DFT ground truth")
    ltc_bundle = train_one(df, LTC_TARGET, ltc_out, log_already=True)
    hc_bundle = train_one(df, HC_TARGET, hc_out, log_already=False)
    return ltc_bundle, hc_bundle


def main():
    df = pd.read_csv("data/combined_features.csv")
    print(f"Loaded {len(df)} real structures with DFT ground truth")

    train_one(df, LTC_TARGET, "model_ltc_real.joblib", log_already=True)
    train_one(df, HC_TARGET, "model_hc_real.joblib", log_already=False)

    # Sanity check: how many structures exceed the Dulong-Petit limit at
    # room temperature? (3NR, N = number of atoms). Per Ojih et al. 2022,
    # this should be rare but not impossible -- flag for inspection,
    # don't auto-discard (see MnIn2Se4 case: a real, DFT-verified exception).
    R = 8.314  # J/mol-K
    dulong_petit = 3 * df["Number of Atom"] * R
    exceeds = df["DFT Heat Capcity"] > dulong_petit
    print(f"\n{exceeds.sum()} / {len(df)} structures exceed the Dulong-Petit limit in this dataset")


if __name__ == "__main__":
    main()
