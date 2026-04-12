import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.impute import SimpleImputer

# Regression & Classification models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC

from sklearn.metrics import r2_score, accuracy_score


def detect_problem_type(y):
    if y.dtype == 'object':
        return "classification"
    if y.nunique() <= 10:
        return "classification"
    return "regression"


def predictive_engine_v4(df, target_column, problem_type=None):

    df = df.copy()

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

    # -------- Column detection --------
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # -------- Pipelines --------
    numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
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
            "Random Forest": RandomForestRegressor(random_state=42),
            "KNN": KNeighborsRegressor(),
            "SVM": SVR()
        }

        for name, model in models.items():

            pipeline = Pipeline([
                ("preprocessing", preprocessor),
                ("model", model)
            ])

            scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
            results[name] = scores.mean()

        best_model = max(results, key=results.get)

    # -------- Classification --------
    else:

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC()
        }

        for name, model in models.items():

            pipeline = Pipeline([
                ("preprocessing", preprocessor),
                ("model", model)
            ])

            scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
            results[name] = scores.mean()

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

print(predictive_engine_v4(df, 'Purchase'))
print(df.head())
