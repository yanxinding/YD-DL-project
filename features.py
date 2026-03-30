from __future__ import annotations
import os
import pandas as pd
import numpy as np

RAW_DIR = "data/raw"
OUT_PATH = "data/processed.csv"

# open a csv as pd df
def _read_csv(name: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Put Lahman CSVs in {RAW_DIR}/")
    return pd.read_csv(path)

# Build team-season dataset with payroll + basic performance features
def build_processed(start_year: int = 1985, end_year: int = 2016, overwrite: bool = False) -> pd.DataFrame:
    # If processed file already exists, reuse it unless overwrite=True is specified
    if os.path.exists(OUT_PATH) and not overwrite:
        return pd.read_csv(OUT_PATH)

    teams = _read_csv("Teams.csv")
    salaries = _read_csv("Salaries.csv")

    # keep needed columns in Teams.csv
    keep = ["yearID", "teamID", "lgID", "divID", "G", "W", "R", "RA"]
    missing = [c for c in keep if c not in teams.columns]
    if missing:
        raise ValueError(f"Teams.csv missing columns: {missing}")

    teams = teams[keep].copy()
    teams = teams[(teams["yearID"] >= start_year) & (teams["yearID"] <= end_year)]

    # Payroll.csv: aggregate salaries to team-season
    sal_keep = ["yearID", "teamID", "salary"]
    missing = [c for c in sal_keep if c not in salaries.columns]
    if missing:
        raise ValueError(f"Salaries.csv missing columns: {missing}")

    payroll = (
        salaries[sal_keep]
        .dropna(subset=["salary"])
        .groupby(["yearID", "teamID"], as_index=False)
        .agg(payroll=("salary", "sum"))
    )


    df = teams.merge(payroll, on=["yearID", "teamID"], how="left")

    # some early years/teams may have missing payroll; keep row but mark NA
    df["log_payroll"] = np.log(df["payroll"])

    # normalize payroll within each year with z-score
    df["payroll_z_by_year"] = df.groupby("yearID")["log_payroll"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) != 0 else 1.0)
    )

    # win percentage
    df["W_pct"] = df["W"] / df["G"]

    # pythagorean win% 
    denom = (df["R"] ** 2 + df["RA"] ** 2).replace(0, np.nan)
    df["pythag_win_pct"] = (df["R"] ** 2) / denom

    # cleanup
    df = df.sort_values(["teamID", "yearID"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    return df

# loas processed data as pd df
def load_processed() -> pd.DataFrame:
    if not os.path.exists(OUT_PATH):
        raise FileNotFoundError(f"{OUT_PATH} not found. Run build_processed() first.")
    return pd.read_csv(OUT_PATH)
