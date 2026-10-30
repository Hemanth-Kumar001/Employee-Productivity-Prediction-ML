# Employee Productivity Prediction

A regression project that estimates a productivity score from five work-related attributes. It compares three scikit-learn regression algorithms using 5-Fold Cross-Validation.

## Dataset

File: Employee_Productivity.csv  
Rows: 600  
Target: Productivity_Score

Input columns:
- Experience_Years
- Training_Hours
- Tasks_Completed
- Attendance_Percentage
- Work_Hours

The uploaded CSV has no missing values.

## Algorithms

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor (n_estimators=200)

Median imputation is included in each model pipeline.

## Evaluation

The primary comparison uses shuffled 5-Fold KFold cross-validation. It reports fold-wise MAE and MSE and their mean +/- standard deviation. It selects the best comparison row by the lowest mean MAE. The script also attempts optional StratifiedKFold after binning the continuous target into quantiles; this is a supplementary regression evaluation, not the primary result.

## Run locally

Install dependencies with: python -m pip install -r requirements.txt

Run: python employee_productivity_prediction.py

Generated output files:
- employee_productivity_cv_comparison.csv
- employee_productivity_mae_comparison.png

## Notes

- The script uses random_state=42 for reproducibility.
- No performance values are claimed here; run the script to generate results in your environment.
- Dataset source and redistribution terms were not provided with the uploaded file. Confirm the source/license before redistributing the CSV elsewhere.
