import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC

# Metrics
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    f1_score
)

import warnings
warnings.filterwarnings("ignore")
import joblib


# -------- Detect Problem Type --------
def detect_problem_type(y):
    if y.dtype == 'object':
        return "classification"
    if y.nunique() <= 10:
        return "classification"
    return "regression"


# -------- Drop ID-like Columns --------
def drop_id_like_columns(df, target_column):
    cols_to_drop = []

    # Check both object AND numeric columns
    for col in df.columns:
        if col == target_column:
            continue
        if df[col].nunique() / len(df) > 0.5:
            cols_to_drop.append(col)

    return df.drop(columns=cols_to_drop), cols_to_drop


# -------- Smart Numeric Conversion --------
def smart_convert_numeric(df, target_column):
    for col in df.columns:
        if col == target_column:
            continue

        if df[col].dtype == 'object':
            try:
                converted = pd.to_numeric(df[col], errors='coerce')

                if converted.notna().sum() > 0.8 * len(df):
                    df[col] = converted
            except:
                pass

    return df


# -------- Main Engine --------
def predictive_engine_v11(df, target_column, problem_type=None):

    df = df.copy()

    # -------- Smart dtype fix --------
    df = smart_convert_numeric(df, target_column)

    # -------- Drop ID-like columns --------
    df, dropped_cols = drop_id_like_columns(df, target_column=target_column)

    if dropped_cols:
        print("Dropped ID-like columns:", dropped_cols)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # -------- Detect problem type --------
    detected_type = detect_problem_type(y)
    final_type = problem_type if problem_type else detected_type

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
        ("encoder", OneHotEncoder(handle_unknown='ignore', max_categories=50))
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
            Xpreds = pipeline.predict(X)

            mae = mean_absolute_error(y, Xpreds)
            rmse = np.sqrt(mean_squared_error(y, Xpreds))

            results[name] = {
                "R2": round(scores.mean(), 4),
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4)
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
            Xpreds = pipeline.predict(X)

            f1 = f1_score(y, Xpreds)

            results[name] = {
                "Accuracy": round(scores.mean(), 4),
                "F1": round(f1, 4)
            }

            pipelines[name] = pipeline

        best_model = max(results, key=lambda x: results[x]["Accuracy"])

    best_pipeline = pipelines[best_model]

    # -------- Feature Importance --------
    model = best_pipeline.named_steps["model"]
    feature_names = best_pipeline.named_steps["preprocessing"].get_feature_names_out()

    importance = None

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = model.coef_
        if len(importance.shape) > 1:
            importance = importance[0]

    # -------- Feature Selection + Retraining --------
    if importance is not None:

        feature_importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        print("\nTop Feature Importance:")
        print(feature_importance.head(10))

        # Select important features
        important_features = feature_importance[
            feature_importance["Importance"].abs() > 0.01
        ]["Feature"].tolist()

        # Extract original feature names
        important_original = set()

        for f in important_features:
        # Feature names look like "num__ColumnName" or "cat__ColumnName_value"
            prefix, rest = f.split("__", 1)  # Split only on first "__"

            if prefix == "num":
            # Numeric: rest is exactly the column name
             if rest in X.columns:
                important_original.add(rest)
             elif prefix == "cat":
                # Try longest match first to avoid partial matches
                matched = sorted(
                    [col for col in X.columns if rest.startswith(col.replace(" ", "_").replace("-", "_"))],
                    key=len, reverse=True
                     )
                if matched:
                    important_original.add(matched[0])

        important_original = list(important_original)
        print("\nSelected Features:", important_original)

        # Reduce dataset
        X_r = X[important_original]

        r_numeric = [c for c in important_original if X[c].dtype in ['int64', 'float64']]
        r_categorical = [c for c in important_original if X[c].dtype == 'object']

        r_preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, r_numeric),
            ("cat", categorical_pipeline, r_categorical)
            ])

        final_pipeline = Pipeline([
            ("preprocessing", r_preprocessor),
            ("model", model.__class__(**model.get_params()))  # Fresh model instance
        ])

        final_pipeline.fit(X_r, y)
        best_pipeline = final_pipeline

    return {
        "Problem Type": final_type,
        "Best Model": best_model,
        "Scores": results,
        "Model": best_pipeline
    }


df = pd.read_csv('/Users/atharvashyammathe/Data Analysis/ML/Sample - Superstore.csv', encoding='latin1')
print("Columns Used as in Dataset:", df.columns.tolist())
# 1. Extract useful date features before dropping date columns
df['Order_Month'] = pd.to_datetime(df['Order Date']).dt.month
df['Order_Year'] = pd.to_datetime(df['Order Date']).dt.year

# 2. Then drop the raw date columns
df = df.drop(columns=[
    'Row ID', 'Order ID', 'Customer ID', 'Product ID',
    'Order Date', 'Ship Date', 'Customer Name', 'Profit'
])

# 3. Keep Product Name — it carries pricing signal
# (already dropped above, consider keeping it)

result = predictive_engine_v11(df, target_column="Sales")

print(result)

sample = pd.DataFrame([{
    "Ship Mode":    "Second Class",
    "Segment":      "Consumer",
    "Country":      "United States",
    "City":         "Los Angeles",
    "State":        "California",
    "Postal Code":  90036,
    "Region":       "West",
    "Category":     "Furniture",
    "Sub-Category": "Chairs",
    "Quantity":     3,
    "Discount":     0.2
}])

predicted_sales = result["Model"].predict(sample)
print(f"Predicted Sales: ${predicted_sales[0]:.2f}")