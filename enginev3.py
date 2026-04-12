import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Regression models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt


def detect_problem_type(y):
    if y.dtype == 'object':
        return "classification"
    if y.nunique() <= 10:
        return "classification"
    return "regression"


def predictive_engine_v3(df, target_column, problem_type=None):

    df = df.copy()

    # -------- Split FIRST --------
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # -------- Detect problem type --------
    detected_type = detect_problem_type(y)

    if problem_type is not None:
        if problem_type != detected_type:
            print(f"⚠️ Warning: Detected {detected_type}, but using {problem_type}")
        final_type = problem_type
    else:
        final_type = detected_type

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------- Column detection --------
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # -------- Pipelines --------
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    results = {}

    # -------- Regression --------
    if final_type == "regression":

        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42)
        }

        for name, model in models.items():
            pipeline = Pipeline([
                ("preprocessing", preprocessor),
                ("model", model)
            ])

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            score = r2_score(y_test, preds)
            results[name] = score

        best_model = max(results, key=results.get)

    # -------- Classification --------
    else:

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42)
        }

        for name, model in models.items():
            pipeline = Pipeline([
                ("preprocessing", preprocessor),
                ("model", model)
            ])

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            score = accuracy_score(y_test, preds)
            results[name] = score

        best_model = max(results, key=results.get)

    return {
        "Problem Type": final_type,
        "Best Model": best_model,
        "Scores": results
    }


df = pd.DataFrame({
    'Feature': np.random.randn(100),
    'Category1': np.random.choice(['X', 'M', 'F'], size=100),
    'Category2': np.random.choice(['A', 'B', 'C'], size=100),
    'Salary': np.random.randn(100),
    'Purchase': np.random.choice(['Yes', 'No'], size=100)
})

print(predictive_engine_v3(df, 'Purchase'))
print(df.head())
