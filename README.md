# Deep Temporal Models for MLB Payroll Efficiency

## Overview
This project studies how efficiently Major League Baseball (MLB) teams convert payroll into on-field success over time.

We model this as a temporal prediction problem, where past team performance influences future outcomes. The project includes:
- Predicting next-season wins  
- Measuring efficiency as performance above/below expectation (residual)

Models are trained on league-wide team-season data (1985–2016) and evaluated both globally and through a Toronto Blue Jays case study.

---

## Key Features
- End-to-end data pipeline from Lahman dataset  
- Classical baseline: Ridge Regression (lag features)  
- Deep models:
  - LSTM  
  - Transformer  
- Time-aware train/validation/test split (no data leakage)  
- Evaluation metrics: RMSE, MAE, R²  
- Efficiency (residual) analysis  
- Ablation studies (sequence length, feature subsets)

---

## Project Structure
```
.
├── data/
│   ├── raw/                # Lahman CSV files (Teams.csv, Salaries.csv)
│   └── processed.csv       # Generated dataset
├── feature.py              # Data preprocessing pipeline
├── analysis.ipynb          # Main experiments and models
├── analysis.html           # Exported results
├── proposal.pdf
├── README.md
├── requirements.txt
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

---

### 2. Prepare data
Download the Lahman Baseball Database:

https://sabr.org/lahman-database/

Download the CSV version and place the following files into:
```
data/raw/
```

Required files:
- Teams.csv  
- Salaries.csv  

---

### 3. Build dataset
Run:
```python
from feature import build_processed
build_processed()
```

If you want to rebuild from scratch:
```python
build_processed(overwrite=True)
```

This will:
- Aggregate salaries into team payroll  
- Log-transform payroll  
- Normalize payroll within each season  
- Compute features:
  - wins (W)
  - runs scored (R)
  - runs allowed (RA)
  - win percentage
  - Pythagorean expected win rate  

Output:
```
data/processed.csv
```

---

### 4. Run experiments
```bash
jupyter notebook analysis.ipynb
```

Run all cells from top to bottom.

---

## Expected Output
Running the notebook will produce:
- Model performance tables (RMSE, MAE, R²)
- Prediction vs actual plots  
- Ablation comparisons  
- Toronto Blue Jays case study results  

---

## Problem Formulation

For each team i and season t:

- Input: previous L seasons  
- Output: next-season wins  

Efficiency residual:
```
r_{t+1} = W_{t+1} - Ŵ_{t+1}^{ridge}
```

---

## Data Split
To avoid temporal leakage:
- Train: 1985–2008  
- Validation: 2009–2012  
- Test: 2013–2016  

---

## Models

### Baseline
- Ridge regression using lagged wins and payroll  

### Deep Temporal Models
- LSTM: captures sequential dependencies  
- Transformer: captures long-range temporal relationships using attention  

All models:
- Use chronological splits  
- Train with mean squared error  

---

## Results (Summary)
- Transformer achieves best overall performance  
- Longer temporal context improves predictions  
- Payroll-only features underperform full feature set  
- Residual analysis captures team efficiency differences  

---

## Reproducibility
To reproduce results:
1. Install dependencies  
2. Place Lahman data in `data/raw/`  
3. Run `build_processed()`  
4. Run all cells in `analysis.ipynb`  

---

## Notes
This project satisfies course requirements:
- Classical baseline + deep temporal models  
- Controlled variants (sequence length, feature subsets)  
- Evaluation metrics (RMSE, MAE, R²)  
- Ablation experiments and analysis  
- Time-aware modeling to prevent data leakage  