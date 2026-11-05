"""Generate the SYNTHETIC demo dataset used to try the app.

The rows are random numbers produced by this script. They do not describe real
people, and the relationships baked in below are arbitrary choices made so a
classifier has something to learn. They are NOT medical facts or real
statistics. Results from this file are for software demonstration only.

Usage:  python scripts/generate_synthetic_data.py [output.csv]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
N = 800


def generate(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(np.round(21 + rng.gamma(2.2, 9.0, n)), 21, 81).astype(int)
    pregnancies = np.clip(rng.poisson(2.5, n), 0, 14)
    glucose = np.clip(rng.normal(116, 27, n), 55, 210).round(0)
    blood_pressure = np.clip(rng.normal(72, 11, n), 40, 118).round(0)
    bmi = np.clip(rng.normal(31, 6.3, n), 16, 58).round(1)
    pedigree = np.clip(rng.lognormal(-0.9, 0.55, n), 0.07, 2.4).round(3)
    activity = rng.choice(["Low", "Moderate", "High"], size=n, p=[0.4, 0.4, 0.2])
    family = rng.choice(["Yes", "No"], size=n, p=[0.35, 0.65])

    # Arbitrary synthetic link so that the label depends on the inputs.
    logit = (
        -10.2
        + 0.048 * glucose
        + 0.075 * bmi
        + 0.028 * age
        + 0.7 * pedigree
        + 0.04 * pregnancies
        + np.select([activity == "Low", activity == "High"], [0.35, -0.35], 0.0)
        + np.where(family == "Yes", 0.45, 0.0)
    )
    prob = 1 / (1 + np.exp(-logit))
    diabetes = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame(
        {
            "PatientID": np.arange(1, n + 1),
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree,
            "Age": age,
            "PhysicalActivity": activity,
            "FamilyHistory": family,
            "Diabetes": diabetes,
        }
    )
    # Sprinkle a few missing values so imputation is exercised.
    for col, share in (("Glucose", 0.015), ("BloodPressure", 0.03), ("BMI", 0.02), ("PhysicalActivity", 0.01)):
        mask = rng.random(n) < share
        df.loc[mask, col] = np.nan
    return df


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "sample-data" / "synthetic_diabetes_demo.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = generate()
    frame.to_csv(out, index=False)
    print(f"Wrote {len(frame)} synthetic rows to {out} (positive share {frame['Diabetes'].mean():.1%})")
