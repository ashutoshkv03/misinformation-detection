# MISINFO_DETECTION

A supervised machine learning project for misinformation / rumor detection using the PHEME dataset.

## Objective
This project classifies text into:
- **0 = Non-Rumor / True**
- **1 = Rumor / Misinformation**

## Dataset
The raw dataset is stored at:

`data/raw/pheme_dataset.csv`

Expected dataset columns:
- `text`
- `is_rumor`
- `user.handle` (optional)
- `topic` (optional)

## Project Structure
```text
MISINFO_DETECTION/
│
├── data/
│   ├── raw/
│   │   └── pheme_dataset.csv
│   └── processed/
│       └── cleaned_dataset.csv
│
├── models/
│   └── best_model.pkl
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── reports/
│   └── project_report.md
│
├── results/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── metrics.json
│   └── top_features.txt
│
├── src/
│   ├── data_preprocessing.py
│   ├── evaluate_model.py
│   ├── feature_engineering.py
│   ├── predict.py
│   ├── train_model.py
│   └── utils.py
│
├── README.md
├── requirements.txt
└── run_project.py