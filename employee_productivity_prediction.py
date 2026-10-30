"""Employee productivity regression with 5-fold cross-validation."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Employee_Productivity.csv"
TARGET = "Productivity_Score"
RANDOM_STATE = 42


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' in the dataset.")

    print("Dataset preview:\n", df.head())
    print("\nDataset information:")
    df.info()
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDescriptive statistics:\n", df.describe(include="all"))

    X = pd.get_dummies(df.drop(columns=[TARGET]), drop_first=False)
    y = df[TARGET]
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    scoring = {
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "mse": make_scorer(mean_squared_error, greater_is_better=False),
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for name, estimator in models.items():
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ])
        scores = cross_validate(model, X, y, cv=kf, scoring=scoring, error_score="raise")
        maes, mses = -scores["test_mae"], -scores["test_mse"]
        print(f"\n{name}")
        for i, (mae, mse) in enumerate(zip(maes, mses), start=1):
            print(f"Fold {i}: MAE={mae:.3f}, MSE={mse:.3f}")
        print(f"Mean MAE={maes.mean():.3f} +/- {maes.std():.3f}")
        print(f"Mean MSE={mses.mean():.3f} +/- {mses.std():.3f}")
        rows.append({
            "Model": name,
            "Mean_MAE": maes.mean(),
            "Std_MAE": maes.std(),
            "Mean_MSE": mses.mean(),
            "Std_MSE": mses.std(),
        })

    results = pd.DataFrame(rows).sort_values("Mean_MAE")
    print("\n5-fold KFold comparison:\n", results.to_string(index=False))
    results.to_csv(BASE_DIR / "employee_productivity_cv_comparison.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.bar(results["Model"], results["Mean_MAE"])
    plt.ylabel("Mean 5-Fold MAE (lower is better)")
    plt.title("Employee Productivity Model Comparison")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "employee_productivity_mae_comparison.png", dpi=150)
    plt.close()
    print("\nBest model by lowest average MAE:", results.iloc[0]["Model"])

    # Optional stratified CV for regression by binning the continuous target.
    try:
        y_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
        if y_bins.nunique() >= 2 and y_bins.value_counts().min() >= 5:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            print("\nTarget-binned StratifiedKFold results:")
            for name, estimator in models.items():
                model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", estimator),
                ])
                scores = cross_validate(
                    model, X, y, cv=skf.split(X, y_bins), scoring=scoring, error_score="raise"
                )
                maes, mses = -scores["test_mae"], -scores["test_mse"]
                print(f"{name}: MAE={maes.mean():.3f} +/- {maes.std():.3f}; "
                      f"MSE={mses.mean():.3f} +/- {mses.std():.3f}")
        else:
            print("\nSkipped stratified regression CV: not enough examples in target bins.")
    except ValueError as exc:
        print("\nCould not run target-binned stratification:", exc)


if __name__ == "__main__":
    main()
