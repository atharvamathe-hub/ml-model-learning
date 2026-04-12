import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score

def detect_problem_type(y):
    if y.dtype == 'object':
        return "classification"
    if y.nunique() <= 10:
        return "classification"
    return "regression"

def predictive_engine_v2(df, target_column, problem_type=None):

    df = df.copy()

    # -------- Split FIRST --------
    X = df.drop(columns=[target_column])
    y = df[target_column]

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

    # -------- Models --------
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42)
    }

    results = {}

    # -------- Train + Evaluate --------
    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        r2 = r2_score(y_test, predictions)
        results[name] = r2

    # -------- Select best --------
    best_model = max(results, key=results.get)

    return best_model, results

df = pd.DataFrame({
    'Feature1': np.random.randn(100),
    'Feature2': np.random.randn(100),
    'Category': np.random.choice(['A', 'B', 'C'], size=100),
    'Salary': np.random.randn(100)
})

print(predictive_engine_v2(df, 'Salary'))
print(df.head())