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

from sklearn.metrics import (r2_score, accuracy_score, mean_absolute_error, mean_squared_error, f1_score)

# -------- Detect Problem Type --------
def detect_problem_type(y):
    if y.dtype == 'object':
        return "classification"
    if y.nunique() <= 10:
        return "classification"
    return "regression"


# -------- Drop ID-like Columns --------
def drop_id_like_columns(df, threshold=0.9, exclude_cols=[]):
    cols_to_drop = []

    # ONLY check object (categorical/text) columns
    obj_cols = df.select_dtypes(include=['object']).columns

    for col in obj_cols:
        if col in exclude_cols:
            continue
        
        unique_ratio = df[col].nunique() / len(df)

        if unique_ratio > threshold:
            cols_to_drop.append(col)

    return df.drop(columns=cols_to_drop), cols_to_drop

# -------- Main Engine --------
def predictive_engine_v8(df, target_column, problem_type=None):

    df = df.copy()

    # -------- Attempt to convert columns to numeric --------
    for col in df.columns:
        if col != target_column:  
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass

    # -------- Drop ID-like columns --------
    df, dropped_cols = drop_id_like_columns(df, exclude_cols=[target_column])

    if dropped_cols:
        print("Dropped ID-like columns:", dropped_cols)

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
    pipelines = {}

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
         
            pipeline.fit(X, y)
            preds = pipeline.predict(X)

            mae = mean_absolute_error(y, preds)
            rmse = np.sqrt(mean_squared_error(y, preds))

            results[name] = {
                "R2": scores.mean(),
                "MAE": mae,
                "RMSE": rmse
            }

            pipelines[name] = pipeline

        best_model = max(results, key=lambda x: results[x]["R2"])

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
            pipeline.fit(X, y)
            preds = pipeline.predict(X)

            f1 = f1_score(y, preds)

            results[name] = {
                "Accuracy": scores.mean(),
                "F1": f1
            }

            pipelines[name] = pipeline

        best_model = max(results,  key=lambda x: results[x]["Accuracy"])

    best_pipeline = pipelines[best_model]

    return {
        "Problem Type": final_type,
        "Best Model": best_model,
        "Scores": results,
        "Model": best_pipeline
    }

df = pd.DataFrame({
    'User_ID': [f"ID_{i}" for i in range(200)],  # ID-like
    'Feature1': np.random.randn(200),
    'Feature2': np.random.randn(200),
    'Category': np.random.choice(['A', 'B', 'C'], size=200)
})

df['Salary'] = 5000 * df['Feature1'] + 2000 * df['Feature2'] + np.random.randn(200)*500

result = predictive_engine_v8(df, "Salary")

print("Best Model:", result["Best Model"])
print("Scores:", result["Scores"])

model = result["Model"]

new_data = pd.DataFrame({
    'Feature1': [0.5],
    'Feature2': [1.2],
    'Category': ['A'],
    
})

prediction = model.predict(new_data)

print("Prediction:", prediction)

model = result["Model"]
