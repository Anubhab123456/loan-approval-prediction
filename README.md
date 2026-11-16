# Loan Approval Prediction System

End-to-end machine learning project: data -> EDA -> preprocessing -> model comparison -> evaluation -> web app.

## Folder structure
```
loan-approval-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/loan_data.csv          # original / synthetic dataset
│   └── processed/loan_clean.csv   # cleaned dataset
├── notebooks/                     # optional exploration notebooks
├── src/
│   ├── config.py                  # paths, columns, constants
│   ├── generate_data.py           # synthetic dataset generator
│   ├── data_loader.py             # load + clean
│   ├── eda.py                     # EDA plots
│   ├── preprocessing.py           # feature engineering + sklearn pipeline
│   ├── train.py                   # CV, model comparison, save best
│   ├── evaluate.py                # metrics + plots
│   └── predict.py                 # single-applicant inference
├── app/app.py                     # Streamlit UI
├── models/loan_model.joblib       # trained pipeline (generated)
├── reports/
│   ├── metrics.json               # results (generated)
│   └── figures/                   # plots for the report (generated)
├── tests/test_pipeline.py         # pytest tests
└── docs/report_outline.md         # technical report structure
```

## How to run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.generate_data     # skip if you use the real Kaggle CSV
python -m src.eda
python -m src.train
python -m src.predict           # quick CLI test
pytest -q
streamlit run app/app.py
```

## Using the real dataset
Download the "Loan Prediction" dataset (Kaggle / Analytics Vidhya), save as
`data/raw/loan_data.csv`. Same columns; `Loan_ID`, `Y/N` and `3+` are handled.

## Models compared
Logistic Regression, Random Forest, Gradient Boosting (5-fold stratified CV; best by ROC-AUC).
