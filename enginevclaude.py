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
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, accuracy_score

import warnings
warnings.filterwarnings("ignore")


# ================================================================
#  STEP 1 — Auto-detect problem type
# ================================================================
def detect_problem_type(y):
    if y.dtype == 'object' or y.nunique() <= 10:
        return "classification"
    return "regression"


# ================================================================
#  STEP 2 — Clean the dataframe automatically
# ================================================================
def auto_clean(df, target_column):
    df = df.copy()

    print("\n--- AUTO CLEANING ---")

    # Drop columns where every value is unique (IDs)
    cols_to_drop = []
    for col in df.columns:
        if col == target_column:
            continue
        if df[col].nunique() / len(df) > 0.5:
            cols_to_drop.append(col)

    if cols_to_drop:
        print(f"Dropped ID-like columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # Drop columns with too many missing values (>50%)
    missing_ratio = df.isnull().mean()
    high_missing = missing_ratio[missing_ratio > 0.5].index.tolist()
    high_missing = [c for c in high_missing if c != target_column]
    if high_missing:
        print(f"Dropped high-missing columns: {high_missing}")
        df = df.drop(columns=high_missing)

    # Try to convert object columns to numeric where possible
    for col in df.columns:
        if col == target_column:
            continue
        if df[col].dtype == 'object':
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().sum() > 0.8 * len(df):
                df[col] = converted
                print(f"Converted '{col}' to numeric")

    # Drop datetime columns (can't use raw dates easily)
    date_cols = []
    for col in df.columns:
        if col == target_column:
            continue
        if df[col].dtype == 'object':
            try:
                parsed = pd.to_datetime(df[col], errors='coerce')
                if parsed.notna().sum() > 0.8 * len(df):
                    date_cols.append(col)
            except:
                pass
    if date_cols:
        print(f"Dropped date columns (raw dates aren't useful): {date_cols}")
        df = df.drop(columns=date_cols)

    print(f"Final columns used for training: {[c for c in df.columns if c != target_column]}")
    return df


# ================================================================
#  STEP 3 — Build preprocessor dynamically from the dataframe
# ================================================================
def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown='ignore', max_categories=50, sparse_output=False))
        ])
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers)


# ================================================================
#  STEP 4 — Train and evaluate all models
# ================================================================
def train_models(X, y, problem_type, preprocessor):

    if problem_type == "regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree":     DecisionTreeRegressor(random_state=42),
            "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
            "KNN":               KNeighborsRegressor(),
            "SVM":               SVR()
        }
        scoring = "r2"
        primary_metric = "R2"

    else:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree":       DecisionTreeClassifier(random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
            "KNN":                 KNeighborsClassifier(),
            "SVM":                 SVC()
        }
        scoring = "accuracy"
        primary_metric = "Accuracy"

    results = {}
    trained_pipelines = {}

    print("\n--- MODEL TRAINING ---")

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model)
        ])

        # Cross-validation score (honest estimate)
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring=scoring)

        # Fit on full data for final pipeline
        pipeline.fit(X, y)
        preds = pipeline.predict(X)

        if problem_type == "regression":
            results[name] = {
                "R2 (CV)": round(cv_scores.mean(), 4),
                "MAE":     round(mean_absolute_error(y, preds), 2),
                "RMSE":    round(np.sqrt(mean_squared_error(y, preds)), 2)
            }
        else:
            n_classes = len(np.unique(y))
            avg = "binary" if n_classes == 2 else "weighted"
            results[name] = {
                "Accuracy (CV)": round(cv_scores.mean(), 4),
                "F1":            round(f1_score(y, preds, average=avg), 4)
            }

        trained_pipelines[name] = pipeline
        print(f"  {name}: {primary_metric} = {cv_scores.mean():.4f}")

    best_model_name = max(results, key=lambda x: list(results[x].values())[0])
    return results, trained_pipelines, best_model_name


# ================================================================
#  STEP 5 — Show feature importance
# ================================================================
def show_feature_importance(pipeline):
    model = pipeline.named_steps["model"]
    feature_names = pipeline.named_steps["preprocessing"].get_feature_names_out()

    importance = None
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_)
        if len(importance.shape) > 1:
            importance = importance[0]

    if importance is not None:
        fi = pd.DataFrame({
            "Feature": feature_names,
            "Importance": np.abs(importance)
        }).sort_values("Importance", ascending=False)

        print("\n--- TOP 10 FEATURE IMPORTANCE ---")
        print(fi.head(10).to_string(index=False))


# ================================================================
#  MAIN ENGINE
# ================================================================
def automl(df, target_column, problem_type=None):

    print("=" * 50)
    print(f"TARGET: {target_column}")

    # Step 1 — Clean
    df = auto_clean(df, target_column)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Step 2 — Detect problem type
    final_type = problem_type if problem_type else detect_problem_type(y)
    print(f"\nProblem Type: {final_type.upper()}")

    # Step 3 — Build preprocessor
    preprocessor = build_preprocessor(X)

    # Step 4 — Train all models
    results, pipelines, best_name = train_models(X, y, final_type, preprocessor)

    best_pipeline = pipelines[best_name]

    print(f"\n✅ Best Model: {best_name}")
    print("\n--- ALL SCORES ---")
    scores_df = pd.DataFrame(results).T
    print(scores_df.to_string())

    # Step 5 — Feature importance
    show_feature_importance(best_pipeline)

    print("=" * 50)

    return {
        "Problem Type": final_type,
        "Best Model":   best_name,
        "Scores":       results,
        "Model":        best_pipeline
    }


# ================================================================
#  PREDICT FUNCTION — clean helper for single row prediction
# ================================================================
def predict(result, input_dict):

    sample = pd.DataFrame([input_dict])
    prediction = result["Model"].predict(sample)
    return prediction[0]


# ================================================================
#  EXAMPLE USAGE
# ================================================================
if __name__ == "__main__":

    df = pd.read_csv('/Users/atharvashyammathe/Data Analysis/ML/Sample - Superstore.csv', encoding='latin1')

    # Drop columns YOU know are leaky or irrelevant
    df = df.drop(columns=['Profit', 'Customer Name', 'Product Name', 'Order ID', 'Customer ID', 'Product ID'])

    # Train
    result = automl(df, target_column="Sales")

    # Predict — just pass the columns the model was trained on
    sample_input = {
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
    }

    predicted = predict(result, sample_input)
    print(f"\n🎯 Predicted Sales: ${predicted:.2f}")








import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
 
# ================================================================
#  VISUALIZATION SUITE for automl() results
# ================================================================
 
def visualize(result, df, target_column):
    """
    Full visualization suite for AutoML results.
 
    Parameters:
        result        : dict returned by automl()
        df            : the SAME cleaned dataframe passed to automl()
        target_column : string — the column that was predicted
    """
 
    sns.set_theme(style="darkgrid", palette="muted")
 
    problem_type  = result["Problem Type"]
    best_name     = result["Best Model"]
    scores        = result["Scores"]
    pipeline      = result["Model"]
 
    X = df.drop(columns=[target_column])
    y = df[target_column]
    y_pred = pipeline.predict(X)
 
    # ----------------------------------------------------------------
    #  Figure layout — 2x2 grid
    # ----------------------------------------------------------------
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"AutoML Results  |  Target: {target_column}  |  Best Model: {best_name}",
                 fontsize=16, fontweight='bold', y=0.98)
 
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
 
    ax1 = fig.add_subplot(gs[0, 0])  # Model comparison
    ax2 = fig.add_subplot(gs[0, 1])  # Actual vs Predicted
    ax3 = fig.add_subplot(gs[1, 0])  # Feature importance
    ax4 = fig.add_subplot(gs[1, 1])  # Residuals
 
    # ================================================================
    #  PLOT 1 — Model Score Comparison
    # ================================================================
    model_names = list(scores.keys())
    primary_key = "R2 (CV)" if problem_type == "regression" else "Accuracy (CV)"
    metric_vals = [scores[m][primary_key] for m in model_names]
 
    colors = ["#2ecc71" if m == best_name else "#95a5a6" for m in model_names]
    bars = ax1.barh(model_names, metric_vals, color=colors, edgecolor='white', height=0.6)
 
    # Add value labels
    for bar, val in zip(bars, metric_vals):
        ax1.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va='center', fontsize=10
        )
 
    ax1.set_title(f"Model Comparison — {primary_key}", fontsize=13, fontweight='bold')
    ax1.set_xlabel(primary_key)
    ax1.axvline(0, color='black', linewidth=0.8)
    ax1.set_xlim(min(min(metric_vals) - 0.1, -0.05), max(metric_vals) + 0.15)
 
    # ================================================================
    #  PLOT 2 — Actual vs Predicted
    # ================================================================
    # Sample up to 500 points for clarity
    idx = np.random.choice(len(y), size=min(500, len(y)), replace=False)
    y_sample = np.array(y)[idx]
    p_sample = y_pred[idx]
 
    ax2.scatter(y_sample, p_sample, alpha=0.4, s=25, color="#3498db", edgecolors='none')
 
    # Perfect prediction line
    min_val = min(y_sample.min(), p_sample.min())
    max_val = max(y_sample.max(), p_sample.max())
    ax2.plot([min_val, max_val], [min_val, max_val],
             color="#e74c3c", linewidth=2, linestyle='--', label="Perfect Fit")
 
    ax2.set_title("Actual vs Predicted", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    ax2.legend()
 
    # R² annotation
    ss_res = np.sum((y_sample - p_sample) ** 2)
    ss_tot = np.sum((y_sample - y_sample.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    ax2.annotate(f"R² = {r2:.4f}", xy=(0.05, 0.92), xycoords='axes fraction',
                 fontsize=11, color="#2c3e50",
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
 
    # ================================================================
    #  PLOT 3 — Feature Importance
    # ================================================================
    model = pipeline.named_steps["model"]
 
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_)
        if len(importance.shape) > 1:
            importance = importance[0]
 
    if importance is not None:
        feature_names = pipeline.named_steps["preprocessing"].get_feature_names_out()
        fi = pd.DataFrame({
            "Feature": feature_names,
            "Importance": np.abs(importance)
        }).sort_values("Importance", ascending=False).head(15)
 
        # Clean up feature names for display
        fi["Feature"] = fi["Feature"].str.replace("num__", "").str.replace("cat__", "")
 
        sns.barplot(data=fi, x="Importance", y="Feature", ax=ax3,
                    palette="Blues_r", edgecolor='white')
        ax3.set_title("Top 15 Feature Importances", fontsize=13, fontweight='bold')
        ax3.set_xlabel("Importance")
        ax3.set_ylabel("")
    else:
        ax3.text(0.5, 0.5, "Feature importance\nnot available for this model",
                 ha='center', va='center', fontsize=12, transform=ax3.transAxes)
        ax3.set_title("Feature Importance", fontsize=13, fontweight='bold')
 
    # ================================================================
    #  PLOT 4 — Residuals Plot
    # ================================================================
    residuals = np.array(y)[idx] - p_sample
 
    ax4.scatter(p_sample, residuals, alpha=0.4, s=25, color="#8431a5", edgecolors='none')
    ax4.axhline(0, color="#e74c3c", linewidth=2, linestyle='--', label="Zero Error")
 
    # ±1 std band
    std = residuals.std()
    ax4.axhline(std,  color="#f39c12", linewidth=1, linestyle=':', label=f"+1 STD ({std:.1f})")
    ax4.axhline(-std, color="#f39c12", linewidth=1, linestyle=':', label=f"-1 STD ({-std:.1f})")
 
    ax4.set_title("Residuals (Actual − Predicted)", fontsize=13, fontweight='bold')
    ax4.set_xlabel("Predicted Value")
    ax4.set_ylabel("Residual")
    ax4.legend(fontsize=9)
 
    # Residual stats annotation
    ax4.annotate(
        f"Mean: {residuals.mean():.2f}\nSTD: {std:.2f}",
        xy=(0.05, 0.92), xycoords='axes fraction', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
    )
 
    plt.savefig("automl_report.png", dpi=150, bbox_inches='tight')
    print("\n📊 Plot saved as automl_report.png")
    plt.show()
 
 
