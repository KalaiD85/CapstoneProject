from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
import os
import joblib
import pandas as pd

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
#reduce the dataset to faster processing
#df = df.sample(n=10000, random_state=42)
#Define target
# Feature Matrix (X)
X = df.drop(columns=["dieSize", "lotName", "waferMap", "waferMap_array"])
# Regression Label (y_reg)
y_reg = df['dieSize']
# Classification label: binarize at median
y_clf = (y_reg > y_reg.median()).astype(int)

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
# trianTestLabel, failureType has no natural order → one-hot encoding
X = pd.get_dummies(X, columns=["trianTestLabel", "failureType"], drop_first=True)

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

#1. Decision Tree baseline:
# Baseline Decision Tree (no constraints)
dt_base = DecisionTreeClassifier(random_state=42)  # default max_depth=None
dt_base.fit(X_train_scaled, y_clf_train)

train_acc = accuracy_score(y_clf_train, dt_base.predict(X_train_scaled))
test_acc = accuracy_score(y_clf_test, dt_base.predict(X_test_scaled))

print("Baseline Decision Tree Training Accuracy:", train_acc)
print("Baseline Decision Tree Test Accuracy:", test_acc)


#2, Controlled Decision Tree 
dt_controlled = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
dt_controlled.fit(X_train_scaled, y_clf_train)
train_acc_ctrl = accuracy_score(y_clf_train, dt_controlled.predict(X_train_scaled))
test_acc_ctrl = accuracy_score(y_clf_test, dt_controlled.predict(X_test_scaled))
print("Controlled Decision Tree Training Accuracy:", train_acc_ctrl)
print("Controlled Decision Tree Test Accuracy:", test_acc_ctrl)


#3. Gini VS Entropy
# Gini-based Decision Tree
dt_gini = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=42)
dt_gini.fit(X_train_scaled, y_clf_train)
test_acc_gini = accuracy_score(y_clf_test, dt_gini.predict(X_test_scaled))

# Entropy-based Decision Tree
dt_entropy = DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=42)
dt_entropy.fit(X_train_scaled, y_clf_train)
test_acc_entropy = accuracy_score(y_clf_test, dt_entropy.predict(X_test_scaled))

print("Decision Tree (Gini) Test Accuracy:", test_acc_gini)
print("Decision Tree (Entropy) Test Accuracy:", test_acc_entropy)

#4. Random forest
# Capture feature names before scaling
feature_names = X.columns.tolist()
# Random Forest model
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train_scaled, y_clf_train)
# Performance metrics
train_acc_rf = accuracy_score(y_clf_train, rf.predict(X_train_scaled))
test_acc_rf = accuracy_score(y_clf_test, rf.predict(X_test_scaled))
roc_auc_rf = roc_auc_score(y_clf_test, rf.predict_proba(X_test_scaled)[:, 1])

print("Random Forest Training Accuracy:", train_acc_rf)
print("Random Forest Test Accuracy:", test_acc_rf)
print("Random Forest ROC-AUC:", roc_auc_rf)

# Feature importance
importances = rf.feature_importances_
indices = importances.argsort()[::-1][:5]  # top 5 features
top_features = [(feature_names[i], importances[i]) for i in indices]

print("Top 5 Features by Importance:")
for feat, score in top_features:
    print(f"{feat}: {score:.4f}")

#4a. Gradient Boosting experiment:
# Gradient Boosting model
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train_scaled, y_clf_train)

# Performance metrics
train_acc_gb = accuracy_score(y_clf_train, gb.predict(X_train_scaled))
test_acc_gb = accuracy_score(y_clf_test, gb.predict(X_test_scaled))
roc_auc_gb = roc_auc_score(y_clf_test, gb.predict_proba(X_test_scaled)[:, 1])

print("Gradient Boosting Training Accuracy:", train_acc_gb)
print("Gradient Boosting Test Accuracy:", test_acc_gb)
print("Gradient Boosting ROC-AUC:", roc_auc_gb)


#4b. Feature Ablation Study
# Assume rf is your trained RandomForestClassifier from Task 4
importances = rf.feature_importances_
indices_low = importances.argsort()[:5]  # lowest 5 features

low_features = [feature_names[i] for i in indices_low]
print("Lowest 5 Features:", low_features)

# Full model AUC (already computed in Task 4)
roc_auc_full = roc_auc_score(y_clf_test, rf.predict_proba(X_test_scaled)[:, 1])

# Remove lowest-importance features
X_train_reduced = np.delete(X_train_scaled, indices_low, axis=1)
X_test_reduced = np.delete(X_test_scaled, indices_low, axis=1)

rf_reduced = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_reduced.fit(X_train_reduced, y_clf_train)

roc_auc_reduced = roc_auc_score(y_clf_test, rf_reduced.predict_proba(X_test_reduced)[:, 1])

print("Full Model ROC-AUC:", roc_auc_full)
print("Reduced Model ROC-AUC:", roc_auc_reduced)

#5. Cross-validated comparison

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression (from Part 2)
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
scores_lr = cross_val_score(log_reg, X_train_scaled, y_clf_train, cv=cv, scoring="roc_auc")

# Controlled Decision Tree
dt_ctrl = DecisionTreeClassifier(max_depth=5, random_state=42)
scores_dt = cross_val_score(dt_ctrl, X_train_scaled, y_clf_train, cv=cv, scoring="roc_auc")

# Random Forest (Task 4)
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
scores_rf = cross_val_score(rf, X_train_scaled, y_clf_train, cv=cv, scoring="roc_auc")

# Gradient Boosting (Task 4a)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
scores_gb = cross_val_score(gb, X_train_scaled, y_clf_train, cv=cv, scoring="roc_auc")

# Report mean and std
print("Logistic Regression AUC: %.3f ± %.3f" % (np.mean(scores_lr), np.std(scores_lr)))
print("Decision Tree (controlled) AUC: %.3f ± %.3f" % (np.mean(scores_dt), np.std(scores_dt)))
print("Random Forest AUC: %.3f ± %.3f" % (np.mean(scores_rf), np.std(scores_rf)))
print("Gradient Boosting AUC: %.3f ± %.3f" % (np.mean(scores_gb), np.std(scores_gb)))


#6. Hyperparameter tuning with GridSearchCV:
# Pipeline with preprocessing + Random Forest
pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    RandomForestClassifier(random_state=42)
)

# Parameter grid
param_grid = {
    "randomforestclassifier__n_estimators": [50, 100, 200],
    "randomforestclassifier__max_depth": [5, 10, None],
    "randomforestclassifier__min_samples_leaf": [1, 5]
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(X_train, y_clf_train)  # unscaled, pipeline handles scaling

print("Best Parameters:", grid_search.best_params_)
print("Best CV AUC Score:", grid_search.best_score_)


#7 Manual learning curve. 
fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
results = []

# After running GridSearchCV
best_pipeline = grid_search.best_estimator_

# Now best_pipeline is the tuned pipeline with best_params_
print("Best pipeline:", best_pipeline)

for f in fractions:
    n = int(f * len(X_train))
    X_subset = X_train.iloc[:n]   # keep DataFrame with feature names
    y_subset = y_clf_train.iloc[:n]
    
    # Fit best pipeline from GridSearchCV
    best_pipeline.fit(X_subset, y_subset)
    
    # Training AUC on the same subset
    train_auc = roc_auc_score(y_subset, best_pipeline.predict_proba(X_subset)[:, 1])
    
    # Test AUC on full test set
    test_auc = roc_auc_score(y_clf_test, best_pipeline.predict_proba(X_test)[:, 1])  # use raw X_test
    results.append((f, train_auc, test_auc))

# Print results as table
print("Training fraction | Training AUC | Test AUC")
for f, train_auc, test_auc in results:
    print(f"{f:.1f} | {train_auc:.3f} | {test_auc:.3f}")


#8. Serialize the best model:

# Save the best pipeline to disk
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "Output")
os.makedirs(output_dir, exist_ok=True)
joblib.dump(best_pipeline, os.path.join(output_dir, "best_model.pkl"))

# Reload the saved model
loaded_model = joblib.load(os.path.join(output_dir, "best_model.pkl"))

# Two handcrafted test rows with the same number of features and column names as X_train
test_rows = pd.DataFrame(
    np.random.rand(2, X_train.shape[1]),   # 2 rows, correct dimensionality
    columns=X_train.columns                # preserve feature names
)

# Predict using the reloaded model
preds = loaded_model.predict(test_rows)
print("Predictions for handcrafted rows:", preds)

test_rows = pd.DataFrame([
    {
        "waferIndex": 0.5,
        "defect_count": 100,
        "normal_count": 500000,
        "defect_ratio": 0.90,
        "defect_severity_encoded": 2,
        "trianTestLabel_B": 1,
        "trianTestLabel_C": 0,
        "failureType_A": 0,
        "failureType_B": 1,
        "failureType_C": 0,
        "failureType_D": 0,
        "failureType_E": 0,
        "failureType_F": 0
    },
    {
        "waferIndex": 1.2,
        "defect_count": 5,
        "normal_count": 95,
        "defect_ratio": 0.05,
        "defect_severity_encoded": 0,
        "trianTestLabel_B": 0,
        "trianTestLabel_C": 1,
        "failureType_A": 1,
        "failureType_B": 0,
        "failureType_C": 0,
        "failureType_D": 0,
        "failureType_E": 0,
        "failureType_F": 0
    }
], columns=X_train.columns)

preds = loaded_model.predict(test_rows)
print("Predictions for handcrafted rows:", preds)

