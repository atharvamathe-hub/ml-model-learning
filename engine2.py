import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def predictive_engine_v2(df, target_column):

    df = df.copy()

    # -------- Split FIRST --------
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------- Identify column types --------
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # -------- Numeric pipeline --------
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # -------- Categorical pipeline --------
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown='ignore'))
    ])

    # -------- Combine pipelines --------
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    # -------- Final pipeline --------
    model_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", LinearRegression())
    ])

    # -------- Train --------
    model_pipeline.fit(X_train, y_train)

    # -------- Predict --------
    predictions = model_pipeline.predict(X_test)

    # -------- Evaluate --------
    r2 = r2_score(y_test, predictions)

    return r2

df = pd.DataFrame({
    'Feature1': np.random.randn(100),
    'Feature2': np.random.randn(100),
    'Category': np.random.choice(['A', 'B', 'C'], size=100),
    'Salary': np.random.randn(100)
})

print(df.head())
print(predictive_engine_v2(df, 'Salary'))