# Predictive Maintenance System

This project implements a predictive maintenance pipeline using time-series machine learning techniques.

## Key Features
- Time-series feature engineering (rolling statistics, EMA, lag features)
- Handling class imbalance using class-weighted models
- Baseline model using Random Forest
- Production model using XGBoost with hyperparameter tuning
- Evaluation using Precision-Recall AUC (PR-AUC)
- Model serialization using joblib

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- imbalanced-learn

## Outcome
The final model predicts machine failures based on sensor degradation patterns and is ready for deployment.
