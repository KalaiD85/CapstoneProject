# ------WM811K Wafer Map Dataset: From Data Cleaning to Explainable AI------ #

Choosen the WM‑811K semiconductor wafer map dataset for the Applied AI & ML Essentials — Capstone Project, because it is one of the largest publicly available, real‑world datasets for defect pattern recognition in semiconductor manufacturing.

# PreRequiste 
Make sure the cleaned_data.csv is availble from Part1 inside the Part1/Output (or) Run Part1 python script and continue with Part2
# Installation 
Install the required library to run the python script in Part2 folder
```bash 
    pip install pandas numpy matplotlib scikit-learn imbalanced-learn os
```
# OS
Windows OS
# IDE
Visual Studio Code
# Run the python script
Supervised_ML.py file for Part 2

# Part 2 — Supervised Machine Learning Model — Build, Train, and Evaluate
# Supervised Machine Learning on WM811K cleaned Dataset
# Scenario
With a clean dataset(cleaned_data.csv) from Part 1 in hand, now produce two predictive models: one for regression (predicting a continuous output) and one for binary classification. You must preprocess the data correctly to avoid data leakage, handle class imbalance where applicable, and evaluate both models rigorously.

Two predictive models:
Regression model → predict a continuous output (dieSize).
Classification model → predict a binary label (above vs. below median die size, or defect vs. none).

# TASK
# 1. Load cleaned_data.csv
# Walkthrough
1. Feature matrix (X): All columns except the target column (dieSize).
2. Regression label (y_reg): Continuous numeric column dieSize.
3. Classification label (y_clf): Binary column derived by binarizing dieSize at its median:
                    y_clf = 1 if dieSize > median(dieSize)
                    y_clf = 0 otherwise
4. Alternative Definition in comments
    A natural binary label already in the dataset, failureType:
```python
y_clf = (df['failureType'] != "[['none']]").astype(int)
```
# Interpretation:
y_clf = 1 → wafer has a defect pattern
y_clf = 0 → wafer has no defect (none)
# Inference
Regression task: predict dieSize (continuous).
Classification task: either (a) predict wafers above/below median die size, or (b) predict wafers with defects vs. none.

# 2. Encode categorical columns
# Walkthrough
1. Ordered categories (Label Encoding):
Feature: defect_severity engineered from defect_ratio.
Categories: Low < Medium < High.
Encoding: Low=0, Medium=1, High=2.
# Justify the ordering
This preserves the natural order of severity levels. Models can interpret “High” as greater severity than “Medium” or “Low,” which is meaningful in predicting wafer outcomes
2. Unordered categories (One-Hot Encoding):
Features: trianTestLabel, failureType.
Encoding: Converted into binary dummy variables using pd.get_dummies(..., drop_first=True).
# why one-hot encoding avoids the false-ordinal-relationship problem that label encoding 
Justification: These categories have no inherent order (e.g., “Donut” vs. “Scratch” defect types).
Label encoding would incorrectly impose a numeric order (e.g., Donut=1, Scratch=2), suggesting Scratch > Donut, which is meaningless.
One-hot encoding avoids this false ordinal relationship by creating independent binary indicators.
Dropping the first dummy column prevents multicollinearity (perfect correlation among dummy variables).

# Inference:
Feature Type	Example Column	              Encoding Method	Why?
Ordinal	        defect_severity	             Label Encoding (Low=0, Medium=1, High=2). Preserves natural order  
Nominal	        failureType, trianTestLabel	 One-Hot Encoding (drop_first=True) .Avoids multicollinearity

# 3. Leak-free train-test split and scaling:
# Walkthrough
1. Train-Test Split 
Split results:
Training set: 59,364 rows × 13 features
Test set: 14,842rows × 13 features
2 Scaling:
# Note: 
Memory spike when scaling full dataset.
Hemce, Batch scaling with partial_fit for StandardScaler.
Impact: Full dataset processed safely, then fit with exact OLS LinearRegression
```bash
#Batch scaling with partial_fit
scaler = StandardScaler()
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
```
Applied StandardScaler fit only on training features.
3. Transformed both training and test sets using this fitted scaler.
# Inference: 
The dataset was successfully divided into 59,364 training samples and 14,842 test samples, each with 13 engineered and encoded features.
This ensures that the models will be trained on a sufficiently large portion of the data while still leaving a meaningful test set for evaluation.

# fitting the scaler on the full dataset would constitute data leakage and explain why (the scaler would encode test-set statistics into the training process).
If the scaler were fit on the entire dataset, it would encode test-set statistics (mean and variance) into the training process. This is called data leakage, and it artificially boosts performance because the model indirectly “sees” information from the test set during training
By fitting only on training data, evaluation on the test set reflects true generalization ability — how well the model performs on unseen data

# 4. Regression Model - Linear Regression
# Walkthrough
1. Initialize Linear Regression:
Creates a LinearRegression() model.
Fits it on the scaled training features (X_train_scaled) and regression labels (y_reg_train).
2. Predict with Linear Regression:
Uses the trained model to predict values on the test set (X_test_scaled).
Stores predictions in y_pred_reg.
3. Evaluate Linear Regression:
Computes Mean Squared Error (MSE) between predictions and true test labels.
Computes R² score to measure variance explained by the model.
4. Inspect coefficients:
Extracts model coefficients and maps them to feature names.
dentifies the top 3 features by absolute coefficient value, showing which features most strongly influence predictions
# Inference
Training: Fit on scaled training features (X_train_scaled) and regression label (y_reg_train).
Prediction: Generated y_pred_reg on X_test_scaled.
Metrics:
MSE: mean_squared_error(4.88e‑26)
R²: r2_score(1.0)
Coefficients: Printed alongside feature names.
Top 3 features by absolute coefficient: Identified using coef_lin.abs().sort_values().head(3).
normal_count → 82.28
defect_count → 39.05
defect_ratio → ~0 (negligible effect)
# what a large positive coefficient means (as one unit increase in the scaled feature is associated with how many units increase in the predicted value) and what a large negative coefficient means.
A large positive coefficient means:(e.g., normal_count) for every one‑unit increase in the scaled feature, the predicted dieSize increases by that coefficient value.
A large negative coefficient means: for every one‑unit increase in the scaled feature, the predicted dieSize decreases by that coefficient value.
Here, normal_count and defect_count are the strongest drivers of wafer die size, while defect_ratio has negligible influence

# Ridge Regression
# Wakthrough 
1. Initialize Ridge Regression:
Creates a Ridge(alpha=1.0) model (adds L2 regularization).
Fits it on the same scaled training data.
2. Predict with Ridge Regression:
Uses the Ridge model to predict values on the test set.
Stores predictions in y_pred_ridge.
3. Evaluate Ridge Regression:
Computes MSE and R² for Ridge predictions.
Prints results for comparison against plain Linear Regression.
# Inference
Training: Fit with Ridge(alpha=1.0) on the same scaled training data.
Prediction: Generated y_pred_ridge on X_test_scaled.
Metrics:
MSE: mean_squared_error(5.99e‑06)
R²: r2_score(0.999999999)

|Model	            | MSE	       | R²         |
|-------------------|--------------|------------|
|Linear Regression	| 4.88e‑26     | 1.00000000 |
|Ridge Regression	| 5.99e‑06	   | 0.999999999 |

# why Ridge might produce a different coefficient profile than OLS Linear Regression and what the alpha parameter controls:
Ridge Regression applies L2 regularization, shrinking coefficients toward zero.
This reduces variance and stabilizes estimates when predictors are correlated.
The alpha parameter controls the strength of shrinkage:
Higher α → stronger penalty, more shrinkage, less variance but more bias.
Lower α → weaker penalty, closer to plain OLS.

# Overall Regression Summary
Both models fit the data extremely well, but the OLS Linear Regression appears better, could be due to strong linear relationships in the dataset.
Ridge Regression sacrifices a tiny amount of accuracy to produce a more robust coefficient profile
The dominance of normal_count and defect_count highlights that wafer die size is primarily driven by raw counts rather than ratios.

# 5. Classification model — Logistic Regression:
# Walkthrough
1. Handling Class Imbalance
Checked y_clf_train.value_counts(normalize=True).
This revealed the imbalance: about 64.7% class 0 and 35.3% class 1.
Since one class had fewer than 35% of samples, imbalance was addressed using SMOTE (applied only to the training set) and class_weight='balanced' in the Logistic Regression constructor.
Justification: SMOTE ensures the minority class is adequately represented, while class_weight='balanced' penalizes misclassifications of the minority class more heavily. Together, they prevent bias toward the majority class.
2. Model Training & Evaluation
Trained LogisticRegression(max_iter=1000, class_weight='balanced') on the resampled training set.
3. Predicted class labels (y_pred_clf) and probabilities (y_proba_clf) on X_test_scaled.
4. Verify the Confusion Matrix, Classification Report:(Accuracy, Precision, Recall, F1 score), ROC Curve plotted with False Positive Rate vs True Positive Rate and AUC value annotated on the plot and saved to plots folder roc_curve_logistic.png
# Interpretation: 
ROC curve was sharp and the AUC value was very close to 1.0, indicating near‑perfect classification performance.
The Logistic Regression model, with imbalance handling, achieved excellent performance — confusion matrix showed very few misclassifications, precision and recall were both high, and AUC was nearly 1.0.
# (a) write out the formulas for Precision and Recall using TP, FP, FN notation; 
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
# (b) explain which metric is more important for this specific classification task and why (e.g., if false negatives are more costly than false positives, recall matters more); 
In wafer defect detection, recall is more important than precision.
Reason: False negatives (missed defects) are more costly than false positives (flagging a defect that isn’t there).
High recall ensures defective wafers are not overlooked, even if it means tolerating some false alarms.
# (c) state what the AUC value means for this model's ability to separate the two classes.
AUC (Area Under the ROC Curve):
Measures the model’s ability to separate the two classes.
AUC = 1.0 → perfect separation.
AUC = 0.5 → no better than random guessing.

# 5b. Decision‑Threshold Sensitivity
# Walkthrough
1. Define thresholds:
A list of thresholds [0.3, 0.4, 0.5, 0.6, 0.7] is created.
These represent different cut‑off values for deciding whether a sample is classified as class 1.
2. Initialize results:
An empty list results is prepared to store precision, recall, and F1 scores for each threshold.
Loop through thresholds:
For each threshold th:
Convert predicted probabilities (y_proba_clf) into class predictions using (y_proba_clf >= th).astype(int).
This means: if the probability is greater than or equal to the threshold, classify as 1; otherwise classify as 0.
3. Compute metrics:
For each set of predictions, calculate:
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 score = harmonic mean of precision and recall.
Append these values along with the threshold to the results list.
4. Create DataFrame:
Convert the results list into a pandas DataFrame with columns: Threshold, Precision, Recall, F1.
This organizes the metrics into a neat table.
Print results
Finally, the DataFrame is printed, showing how precision, recall, and F1 change as the threshold varies.

| Threshold | Precision | Recall | F1 |
| --- | --- | --- | --- |
| 0.3 | 0.9996 | 0.9998 | 0.9997 |
| 0.4 | 0.9996 | 0.9998 | 0.9997 |
| 0.5 | 1.0000 | 0.9998 | 0.9999 |
| 0.6 | 1.0000 | 0.9994 | 0.9997 |
| 0.7 | 1.0000 | 0.9973 | 0.9987 |
# Inference
Precision: Already extremely high across all thresholds, reaching 1.0 from 0.5 onward.
Recall: Highest (≈ 0.9998) at thresholds 0.3–0.5, but begins to decline as the threshold increases (0.9994 at 0.6, 0.9973 at 0.7).
F1 Score: Peaks at threshold 0.5 (≈ 0.9999), showing the best balance between precision and recall.

# (a) write out the formulas for Precision (TP / (TP + FP)) and Recall (TP / (TP + FN)); 
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
# (b) identify the threshold that maximises F1-score on this dataset; 
The highest F1‑score (≈ 0.999) occurs at threshold = 0.50.
This threshold balances precision and recall most effectively for the dataset.
# (c) state which metric — precision or recall — is more important for this specific classification task and why; 
For wafer defect detection, recall is more important than precision.
Reason: False negatives (missed defects) are more costly than false positives (flagging a good wafer).
Ensuring defective wafers are not overlooked is critical in manufacturing. 
# (d) state whether you would raise or lower the threshold to optimise for that metric and what the cost is of doing so.
To optimize for recall, we would lower the threshold (e.g., from 0.5 to 0.3-0.4). Raise the threshold to 0.5 or higher if the goal is perfect precision.
Cost: Lowering the threshold increases recall but may reduce precision (more false positives). Rising the threshold will have the recall drops slightly, meaning a few true positives are missed.
The trade‑off is to lower the threshold, which is acceptable in semiconductor defect detection, since missing a defective wafer is riskier than mistakenly flagging a good one.

# 6. Regularization Experiment — Logistic Regression
# Walkthrough
1. Baseline model setup:
Creates a Logistic Regression model with C=1.0 (default regularization strength), class_weight="balanced", and max_iter=1000.
Fits the model on the scaled training data (X_train_scaled, y_clf_train).
2. Baseline predictions:
Predicts class labels on the test set (y_pred_base).
Predicts class probabilities (y_prob_base) and extracts the probability of class 1.
3. Baseline metrics:
Computes precision, recall, and AUC for the baseline model using the test set predictions.
4. Regularized model setup:
Creates a second Logistic Regression model with C=0.01 (stronger L2 penalty).
Fits this model on the same training data.
5.Regularized predictions:
Predicts class labels (y_pred_reg) and probabilities (y_prob_reg) on the test set.
6. Regularized metrics:
Computes precision, recall, and AUC for the strongly regularized model.
7. Comparison table:
Prints a Markdown‑formatted table comparing the baseline (C=1.0) and strongly regularized (C=0.01) models side by side.
Columns: Model | Precision | Recall | AUC.
# Inference
The baseline model (C=1.0) achieved near‑perfect performance with precision, recall, and AUC all essentially equal to 1.0.
The strongly regularized model (C=0.01) showed lower precision, recall, and AUC, indicating that stronger regularization led to underfitting.
In this case, reducing C worsened performance, because the coefficients were shrunk too aggressively, limiting the model’s ability to capture the strong signal in the data.
# Compare its precision, recall, and AUC against the baseline C=1.0 model in a table
| Model | Precision | Recall | AUC |
| --- | --- | --- | --- |
| **C = 1.0 (Baseline)** | 1.000 | 1.000 | 1.000 |
| **C = 0.01 (Strong Regularization)** | 0.994 | 0.986 | 0.998 |
# what C controls in logistic regression and whether reducing C improved or worsened performance on this dataset.
What C controls:
In Logistic Regression, C is the inverse of regularization strength.
Larger C → weaker regularization 
Smaller C → stronger regularization 
Regularization helps stabilize models when predictors are correlated or when overfitting is a risk.

# 7.  Bootstrap confidence interval for AUC difference
# Walkthrough
1. Set bootstrap parameters:
Defines n_boot = 500, meaning 500 bootstrap resamples will be drawn from the test set.
Initializes an empty list diffs to store AUC differences.
2. Loop through bootstrap samples:
For each of the 500 iterations:
Randomly sample indices from the test set with replacement using np.random.choice.
Create a bootstrap sample of true labels (y_true_sample) and predicted probabilities from both models (y_proba_clf for C=1.0 and y_prob_reg for C=0.01).
3. Compute AUCs:
For each bootstrap sample, compute the ROC AUC for the baseline model (auc_base) and the strongly regularized model (auc_strong_sample).
Append the difference (auc_base - auc_strong_sample) to the list diffs.
4. Aggregate results:
After all iterations, convert diffs into a NumPy array.
Compute the mean difference across all bootstrap samples.
Compute the 95% confidence interval using the 2.5th and 97.5th percentiles of the differences.
5. Print results:
Bootstrap mean AUC diff: 0.0024334177286474425
95% CI: (np.float64(0.0015674850757049995), np.float64(0.0033860620537814992))
# Inference
Statistical perspective: The C=1.0 model consistently outperforms the baseline.
# Displays the mean AUC difference and the confidence interval.
Mean AUC difference: 0.0024
Lower bound of 95% CI (2.5th percentile): 0.00154
Upper bound of 95% CI (97.5th percentile): 0.00341
# whether the 95% confidence interval for the AUC difference excludes zero — if it does, the C=1.0 model's advantage is likely consistent across different data samples; if it includes zero, the difference may not be reliable.
Since the confidence interval excludes zero, the baseline Logistic Regression (C=1.0) consistently outperforms the strongly regularized model (C=0.01). This advantage is statistically reliable across different resampled test sets
This confirms that the baseline Logistic Regression is the more robust choice for this dataset.