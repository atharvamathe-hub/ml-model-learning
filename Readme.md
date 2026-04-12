# 🧠 ML Model Learning — Automated Model Selection Engine

## 🚀 Overview

This project is a **progressively built Machine Learning engine** that evolves from basic model training to a semi-automated **AutoML-like system**.

The goal of this project is to:

* Understand ML concepts step-by-step
* Build a system that can automatically:

  * Clean data
  * Detect problem type
  * Train multiple models
  * Evaluate performance
  * Select the best model
  * Explain model behavior

---

## 🏗️ Project Structure

```
ml-model-learning/
│
├── engine2.py → Initial basic model training
├── engine3.py → Added preprocessing
├── engine4.py → Multiple model support
├── engine5.py → Cross-validation
├── engine6.py → Classification + Regression support
├── engine7.py → ID column detection
├── engine8.py → Metrics improvement
├── engine9.py → Feature importance
├── engine10.py → Smart dtype handling
├── engine11.py → Feature selection + retraining
├── enginevclaude.py → Claude-based AutoML comparison
│
├── Sample - Superstore.csv → Dataset used
└── README.md
```

---

## ⚙️ How It Works

The engine follows a pipeline-based ML workflow:

```
Raw Data
   ↓
Data Cleaning
   ↓
Preprocessing (Imputation + Encoding + Scaling)
   ↓
Model Training (Multiple Models)
   ↓
Cross Validation
   ↓
Model Selection
   ↓
Feature Importance
   ↓
Feature Selection (v11)
   ↓
Final Model
```

---

## 🧠 Key Features

### ✅ 1. Automatic Problem Detection

* Detects:

  * Regression
  * Classification

---

### ✅ 2. Data Preprocessing Pipeline

* Missing value handling (`SimpleImputer`)
* Feature scaling (`StandardScaler`)
* Categorical encoding (`OneHotEncoder`)
* Fully integrated using `Pipeline` + `ColumnTransformer`

---

### ✅ 3. Multi-Model Training

Supports:

**Regression Models:**

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* KNN Regressor
* SVM Regressor

**Classification Models:**

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* KNN Classifier
* SVM Classifier

---

### ✅ 4. Model Evaluation

**Regression Metrics:**

* R² Score
* MAE
* RMSE

**Classification Metrics:**

* Accuracy
* F1 Score

---

### ✅ 5. Cross Validation

* Uses `cross_val_score`
* Ensures more reliable model selection

---

### ✅ 6. Automatic Model Selection

* Chooses best model based on:

  * R² (regression)
  * Accuracy (classification)

---

### ✅ 7. Feature Importance (Explainability)

* Extracts feature importance from:

  * Linear models (`coef_`)
  * Tree models (`feature_importances_`)
* Helps understand model decisions

---

### ✅ 8. Smart Data Handling

* Detects and removes:

  * ID-like columns
* Converts numeric-like strings safely
* Prevents incorrect encoding

---

### ✅ 9. Feature Selection (v11)

* Identifies low-importance features
* Reduces feature space
* Retrains model on selected features

---

## 🧪 Example Usage

```python
result = predictive_engine_v11(df, target_column="Sales")

print("Best Model:", result["Best Model"])
print("Scores:", result["Scores"])

prediction = result["Model"].predict(new_data)
```

---

## 🔍 Key Insights from the Project

* Data preprocessing is more important than model choice
* High cardinality categorical features can break performance
* Pipelines are critical for consistency
* Feature importance helps in model interpretability
* Overfitting can mislead evaluation metrics

---

## ⚠️ Limitations / Drawbacks

### ❌ 1. No Train-Test Split

* Current evaluation uses training data
* Leads to optimistic (biased) results

---

### ❌ 2. Feature Selection Instability

* Mapping from encoded features → original features is imperfect
* Can cause pipeline mismatches

---

### ❌ 3. High Cardinality Problem

* Columns like `Product Name` create thousands of features
* Slows down training significantly

---

### ❌ 4. No Hyperparameter Tuning

* Models use default parameters
* Performance is not fully optimized

---

### ❌ 5. No Model Persistence Workflow

* Model saving is manual (not integrated)

---

### ❌ 6. No Handling of Imbalanced Data

* Classification may fail on skewed datasets

---

### ❌ 7. Limited Error Handling

* Some edge cases still cause crashes

---

## 🔧 Improvements Needed

### 🚀 Immediate Improvements

* Add `train_test_split`
* Evaluate on unseen data
* Fix feature selection pipeline

---

### 🚀 Intermediate Improvements

* Add `GridSearchCV` / `RandomizedSearchCV`
* Handle high-cardinality features (target encoding)
* Improve feature selection logic

---

### 🚀 Advanced Improvements

* Add model explainability (SHAP / LIME)
* Build Streamlit UI
* Add model versioning
* Deploy as API

---

## 🧠 Learning Outcomes

This project helped in understanding:

* End-to-end ML pipeline design
* Data preprocessing challenges
* Model selection strategies
* Feature engineering importance
* Real-world ML system issues

---

## 📌 Final Summary

This is not just a model — it is a:

> **Self-evolving ML pipeline built step-by-step from scratch**

It demonstrates the transition from:

* Basic ML → Structured ML → Automated ML thinking

---

## 👤 Author

Atharva Mathe
GitHub: [atharvamathe-hub](https://github.com/atharvamathe-hub)

---

## ⭐ Future Vision

Convert this into:

* Full AutoML system
* Web-based ML tool
* Production-ready ML pipeline

---
