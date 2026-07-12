import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import os

# 1. Load dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up one level (to CapstoneProject), then into Part1\Output
output_dir_path = os.path.join(script_dir, "..", "Part1", "Output")
# Normalize the path
output_dir_path = os.path.normpath(output_dir_path)
output_file_path = os.path.join(output_dir_path, "cleaned_data.csv")
# Check folder and file existence
if not os.path.exists(output_dir_path):
    raise FileNotFoundError(f"Required folder not found: {output_dir_path}")
if not os.path.isfile(output_file_path):
    raise FileNotFoundError(f"Required file not found: {output_file_path}")
df = pd.read_csv(output_file_path)
#Define target
# Feature Matrix (X)
X = df.drop(columns=["dieSize", "lotName", "waferMap", "waferMap_array"])
# Regression Label (y_reg)
y_reg = df['dieSize']
# Classification label: binarize at median
y_clf = (y_reg > y_reg.median()).astype(int)
# Alternative natural binary label:
# y_clf = (df['failureType'] != "[['none']]").astype(int)
# Inspect
print("Feature matrix shape:", X.shape)
print("Regression label shape:", y_reg.shape)
print("Classification label distribution:\n", y_clf.value_counts())

# 2. Encode categorical columns:
# engineered ordinal feature based on defect_ratio
def categorize_severity(ratio):
    if ratio < 0.1:
        return "Low"
    elif ratio < 0.3:
        return "Medium"
    else:
        return "High"

df["defect_severity"] = df["defect_ratio"].apply(categorize_severity)
# Apply label encoding to preserve order Low < Medium < High
severity_mapping = {"Low": 0, "Medium": 1, "High": 2}
df["defect_severity_encoded"] = df["defect_severity"].map(severity_mapping)
print(df[["defect_ratio", "defect_severity", "defect_severity_encoded"]].head())
# trianTestLabel, failureType has no natural order → one-hot encoding
X = pd.get_dummies(X, columns=["trianTestLabel", "failureType"], drop_first=True)
print(X.head())
print("Shape after encoding:", X.shape)

# 3. Leak-free train-test split and scaling
#Train-test split
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42
)
# Scaling (fit only on training set to avoid leakage)
scaler = StandardScaler()
# Batch scaling
batch_size = 2000
# Fit scaler incrementally
for start in range(0, len(X_train), batch_size):
    end = start + batch_size
    scaler.partial_fit(X_train.iloc[start:end])

# Transform in batches
def transform_in_batches(X_data, scaler, batch_size=2000):
    scaled_batches = []
    for start in range(0, len(X_data), batch_size):
        end = start + batch_size
        scaled_batches.append(scaler.transform(X_data.iloc[start:end]))
    return np.vstack(scaled_batches)

X_train_scaled = transform_in_batches(X_train, scaler, batch_size)
X_test_scaled = transform_in_batches(X_test, scaler, batch_size)
print("Train shape:", X_train_scaled.shape)
print("Test shape:", X_test_scaled.shape)

# 4. Regression model
#Linear Regression:
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_reg_train)
y_pred_reg = lin_reg.predict(X_test_scaled)

mse_lin = mean_squared_error(y_reg_test, y_pred_reg)
r2_lin = r2_score(y_reg_test, y_pred_reg)
# Coefficients
coef_lin = pd.Series(lin_reg.coef_, index=X.columns)
top3_features = coef_lin.abs().sort_values(ascending=False).head(3)
print("Linear Regression MSE:", mse_lin)
print("Linear Regression R²:", r2_lin)
print("Top 3 features by absolute coefficient:\n", top3_features)

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_reg_train)
y_pred_ridge = ridge.predict(X_test_scaled)
mse_ridge = mean_squared_error(y_reg_test, y_pred_ridge)
r2_ridge = r2_score(y_reg_test, y_pred_ridge)
print("Ridge Regression MSE:", mse_ridge)
print("Ridge Regression R²:", r2_ridge)

# 5. Classification model
#Logistic Regression
# Handle imbalance
print(y_clf_train.value_counts(normalize=True))
if y_clf_train.value_counts(normalize=True).min() < 0.35:
    smote = SMOTE(random_state=42)
    X_train_scaled, y_clf_train = smote.fit_resample(X_train_scaled, y_clf_train)

# Logistic Regression baseline
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg.fit(X_train_scaled, y_clf_train)
y_pred_clf = log_reg.predict(X_test_scaled)
y_proba_clf = log_reg.predict_proba(X_test_scaled)[:, 1]
print("Confusion Matrix:\n", confusion_matrix(y_clf_test, y_pred_clf))
print("Classification Report:\n", classification_report(y_clf_test, y_pred_clf))
# ROC curve
# Get the folder where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "plots")
os.makedirs(output_dir, exist_ok=True)
fpr, tpr, _ = roc_curve(y_clf_test, y_proba_clf)
auc_val = roc_auc_score(y_clf_test, y_proba_clf)
plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
# Save the plot as PNG inside plots folder
plt.savefig(os.path.join(output_dir, "roc_curve_logistic.png"), dpi=300, bbox_inches="tight")
plt.close()

# 5b. Threshold sensitivity
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
results = []
for th in thresholds:
    preds = (y_proba_clf >= th).astype(int)
    results.append([
        th,
        precision_score(y_clf_test, preds),
        recall_score(y_clf_test, preds),
        f1_score(y_clf_test, preds)
    ])
threshold_table = pd.DataFrame(results, columns=["Threshold", "Precision", "Recall", "F1"])
print(threshold_table)


# 6. Regularization experiment on Logistic Regression:
# Baseline Logistic Regression (C=1.0)
log_reg_base = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
log_reg_base.fit(X_train_scaled, y_clf_train)
y_pred_base = log_reg_base.predict(X_test_scaled)
y_prob_base = log_reg_base.predict_proba(X_test_scaled)[:, 1]

precision_base = precision_score(y_clf_test, y_pred_base)
recall_base = recall_score(y_clf_test, y_pred_base)
auc_base = roc_auc_score(y_clf_test, y_prob_base)

# Strongly regularized Logistic Regression (C=0.01)
log_reg_reg = LogisticRegression(class_weight="balanced", max_iter=1000, C=0.01)
log_reg_reg.fit(X_train_scaled, y_clf_train)
y_pred_reg = log_reg_reg.predict(X_test_scaled)
y_prob_reg = log_reg_reg.predict_proba(X_test_scaled)[:, 1]

precision_reg = precision_score(y_clf_test, y_pred_reg)
recall_reg = recall_score(y_clf_test, y_pred_reg)
auc_reg = roc_auc_score(y_clf_test, y_prob_reg)

# Comparison table
print("| Model | Precision | Recall | AUC |")
print("|-------|-----------|--------|-----|")
print(f"| C=1.0 | {precision_base:.3f} | {recall_base:.3f} | {auc_base:.3f} |")
print(f"| C=0.01 | {precision_reg:.3f} | {recall_reg:.3f} | {auc_reg:.3f} |")

# 7. Bootstrap CI
n_boot = 500
diffs = []
for _ in range(n_boot):
    idx = np.random.choice(len(y_clf_test), size=len(y_clf_test), replace=True)
    y_true_sample = y_clf_test.iloc[idx]
    y_proba_base_sample = y_proba_clf[idx]
    y_proba_strong_sample = y_prob_reg[idx]
    auc_base = roc_auc_score(y_true_sample, y_proba_base_sample)
    auc_strong_sample = roc_auc_score(y_true_sample, y_proba_strong_sample)
    diffs.append(auc_base - auc_strong_sample)

mean_diff = np.mean(diffs)
ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

print("\nBootstrap mean AUC diff:", mean_diff)
print("95% CI:", (ci_low, ci_high))