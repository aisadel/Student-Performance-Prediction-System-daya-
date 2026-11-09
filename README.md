# Student Performance Prediction System

A classroom prototype that estimates whether a student is at risk of failing
the final Mathematics grade, built for the Data Mining laboratory.

## Problem
Academic risk is often noticed only after poor results. This prototype explores
early identification of students who may need additional academic support.

## Dataset
UCI Machine Learning Repository - Student Performance (ID 320)
https://archive.ics.uci.edu/dataset/320/student+performance

## Prediction target
`G3 < 10` = **AT RISK**; `G3 >= 10` = **NOT AT RISK**.
`G3` is the target only and is never used as an input feature.

## Technologies
Python 3.10+, pandas, scikit-learn (Random Forest), joblib, Streamlit, ucimlrepo, Git/GitHub.

## Installation and run steps
```bash
pip install -r requirements.txt
python train_model.py      # downloads data, trains model, creates student_model.pkl
streamlit run app.py       # opens http://localhost:8501
```
In the app, download the sample input CSV, then upload it to see predictions.

## Limitations
- Trained on one Portuguese secondary-school dataset; it may not generalize to
  other institutions or populations.
- Predictions are not diagnoses and must not be the sole basis for real student decisions.
- Includes G1 and G2 (earlier period grades), so it is not a very early warning.
