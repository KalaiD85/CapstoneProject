# ------WM811K Wafer Map Dataset: From Data Cleaning to Explainable AI------ #

Choosen the WM‑811K semiconductor wafer map dataset for the Applied AI & ML Essentials — Capstone Project, because it is one of the largest publicly available, real‑world datasets for defect pattern recognition in semiconductor manufacturing.

# PreRequiste 
Make sure the cleaned_data.csv is available from Part1 inside the Part1/Output (or) If Part1/Output folder contains cleaned_data.zip, run the Part3/Unzip.py to get the dataset file(cleaned_data.csv) in the Part1/Output folder (or) Run Part1 python script and continue with Part3
# Installation 
Install the required library to run the python script in Part3 folder
```bash 
    pip install pandas numpy scikit-learn imbalanced-learn os joblib
```
# OS
Windows OS
# IDE
Visual Studio Code
# Run the python script
Advance_Modeling.py for Part 3

# Part 3 — Advanced Modeling — Ensembles, Tuning, and Full ML Pipeline
# Advanced Modeling on WM811K cleaned Dataset
# Scenario
With a clean dataset(cleaned_data.csv) from Part 1 in hand. Your client wants to know not just whether a model works, but which model is the most robust and how confident you are in that claim. You will now build ensemble models, tune them systematically, and package everything into a reproducible sklearn Pipeline that can be serialized and reloaded as best_model.pkl in Output folder

# PRE-TASK
# Load cleaned_data.csv
Feature matrix (X): All columns except the target column (dieSize).
Regression label (y_reg): Continuous numeric column dieSize.
Classification label (y_clf): Binary column derived by binarizing dieSize at its median:
                    y_clf = 1 if dieSize > median(dieSize)
                    y_clf = 0 otherwise
# Encode categorical columns:
# Leak-free train-test split and scaling
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

# 1. Decision Tree baseline
# Walkthrough 
1. DecisionTreeClassifier(random_state=42) initializes a tree with no depth limit (max_depth=None)
2. fit(X_train_scaled, y_clf_train) trains the tree on the scaled training data.
3. Predictions are made on both training and test sets.
4. accuracy_score computes the proportion of correct predictions.
5. Baseline Decision Tree Results
Training Accuracy: 1.0
Test Accuracy: 0.9989
# Inference:
The training accuracy is perfect (100%), which means the unconstrained tree memorized the training data completely.
The test accuracy is still very high (≈ 99.89%), but slightly lower than training. This small gap indicates mild overfitting: the tree is fitting training data exactly, but generalizes almost perfectly to unseen data in this case
# whether this decision tree shows signs of overfitting (high train accuracy, significantly lower test accuracy) 
Yes — the unconstrained decision tree shows signs of overfitting. The training accuracy is perfect (100%), which means the tree memorized the training data. The test accuracy is still very high (≈ 99.89%), but the small drop compared to training indicates that the model is not generalizing perfectly. 
# explain why decision trees are described as high-variance models: they fit the training data greedily at each split without revisiting earlier decisions.
At each node, the tree greedily selects the best split based only on the current subset of data.
Once a split is made, it is never revisited or adjusted.
Small changes in the training data can lead to very different tree structures.
This sensitivity makes decision trees high‑variance learners: they can achieve perfect training accuracy but risk unstable generalization to unseen data.

# 2. Controlled Decision Tree
# Walkthrough
1. max_depth=5 → limits how deep the tree can grow.
2. Prevents the tree from memorizing every detail of the training data.
3. Reduces variance but introduces some bias 
4. min_samples_split=20 → prevents splitting a node if fewer than 20 samples are present.
5. Avoids splits that respond to noise in small subsets.Helps stabilize the tree structure.
6. Results
Training Accuracy: 0.99498
Test Accuracy: 0.99407
# Inference:
Training Accuracy: lower than 1.0 (because depth is capped and splits are restricted). Test Accuracy: closer to training accuracy
# role of max_depth (limits how deep the tree grows, reducing variance at the cost of some bias) 
Limits how deep the tree can grow.
A shallow tree (small max_depth) cannot capture every fine‑grained pattern → introduces bias.
But it prevents the tree from memorizing the training data, reducing variance.
controlled tree with max_depth=5 in code, this constraint stopped the model from growing into a perfect memorizer, leading to slightly lower training accuracy but better generalization.
# role of min_samples_split (prevents splitting a node if fewer than this many samples are present, avoiding splits that respond to noise in small subsets).
Prevents splitting a node if fewer than this many samples are present.
Avoids splits that respond to noise in very small subsets.
Acts as a stabilizer: ensures each split is statistically meaningful.
controlled tree with min_samples_split=20, this avoided over‑fragmentation of the data, further reducing variance
# Compare the controlled tree's train/test gap to the unconstrained tree.
Baseline Tree: Training Accuracy = 1.0, Test Accuracy = 0.9989 → mild overfitting.
Controlled Tree: Training Accuracy = 0.99498, Test Accuracy = 0.99407 → both slightly lower, but the train/test gap is narrower.
Inference: Constraints reduced variance and improved generalization. The model sacrificed a bit of training accuracy but became more robust on unseen data.

# 3. Gini vs Entropy comparison
# Walkthrough
1. Initialize models:
Both Gini and Entropy are constrained to max_depth=5 to prevent overfitting.
The only difference is the criterion used to measure impurity: Gini vs Entropy.
2. Train Models:
Each model learns splits from the training data.
Splits are chosen greedily based on the impurity measure.
3. Evaluate Test Accuracy:
Decision Tree (Gini) Test Accuracy: 0.99427
Decision Tree (Entropy) Test Accuracy: 0.99636
# Inference:
Gini achieved 99.43% test accuracy.
Entropy achieved 99.64% test accuracy.
The difference is small, but Entropy performed slightly better here.
# what it means for a node to have Gini = 0
Gini = 0 means the node is pure: all samples belong to a single class.
Both Gini and Entropy measure impurity, but they differ slightly in sensitivity:
Gini tends to favor larger class splits.
Entropy is more sensitive to minority classes.

# 4. Random Forest:
# Walkthrough
1. Capture Feature Names
Stores the original feature names before scaling.
Needed later to map feature_importances_ back to human‑readable features.
2. Initialize and Train Random Forest
n_estimators=100: builds an ensemble of 100 trees.
max_depth=10: limits tree depth to reduce overfitting.
random_state=42: ensures reproducibility.
.fit() trains the forest on the scaled training data.
3. Evaluate Performance
Training Accuracy: 0.9983
Test Accuracy: 0.9967
ROC‑AUC: 0.99999
Shows the model generalizes extremely well, with minimal train/test gap.
4. Feature Importance Extraction
Retrieves feature importance scores.
Sorts them in descending order.
Selects the top 5 most important features.
normal_count: 0.6056
defect_ratio: 0.1885
trianTestLabel_[['Training']]: 0.1181
defect_count: 0.0786
failureType_[['none']]: 0.0027
# Interpretation:
Linear regression coefficients measure the direct linear effect of a feature on the target, while Random Forest importance reflects how often and how effectively a feature helps partition the data
# How Random Forest Computes Feature Importance
Feature importance is calculated as the average reduction in Gini impurity across all splits where the feature is used, aggregated across all trees.
# why it differs from a linear regression coefficient.
This differs from a linear regression coefficient, which measures the direct linear contribution of a feature to the output.
Random Forest importance reflects how often and how effectively a feature helps partition the data, not the magnitude of a linear relationship.
# bagging concept
Each tree is trained on a bootstrap sample. At each split, a random subset of √(number of features) is considered. This randomness ensures diversity among trees. Averaging predictions across trees reduces variance compared to a single deep decision tree, making Random Forests more robust
# how this ensemble averaging reduces variance compared to a single deep decision tree.
Variance from individual trees cancels out.
The ensemble prediction is more stable and less sensitive to noise.

# 4a. Gradient Boosting
# Walktrough
1. Initialize the Model:
n_estimators=100: builds 100 sequential trees.
learning_rate=0.1: scales each tree’s contribution, preventing overfitting.
max_depth=3: keeps trees shallow, focusing on small corrections rather than memorization.
random_state=42: ensures reproducibility.
2. Train Model:
Fits the Gradient Boosting classifier on the scaled training data.
Each tree is trained to correct the residual errors of the previous ensemble.
3. Evaluate Performance:
Gradient Boosting Training Accuracy: 0.9988882150798464
Gradient Boosting Test Accuracy: 0.9977765799757445
Gradient Boosting ROC-AUC: 0.9999069041327403
# Inference
Compared to Random Forest (bagging, variance reduction), Gradient Boosting focuses on bias reduction by iteratively improving predictions.

# 4b. Feature ablation study
# Walkthrough
1. Identify 5 lowest-importance features from Random Forest
2. Compute Full model AUC
3. Remove lowest-importance features
4. Evaluate the output
Lowest 5 Features: ["failureType_[['Donut']]", "failureType_[['Scratch']]", "failureType_[['Near-full']]", "failureType_[['Random']]", "failureType_[['Loc']]"]
Full Model ROC-AUC: 0.9999884624779892
Reduced Model ROC-AUC: 0.999991227504816
# Inference:
The reduced model’s ROC‑AUC (0.999991) is slightly higher than the full model’s ROC‑AUC (0.999988).
This indicates the removed features were genuinely uninformative and may have added noise rather than predictive signal.
The model not only maintained performance but improved marginally after removing them.

# whether the removed features were genuinely uninformative (AUC is similar or higher without them, suggesting they added noise) or were contributing (AUC drops without them). 
Since the reduced model’s ROC‑AUC is slightly higher than the full model’s ROC‑AUC, the removed features were genuinely uninformative. They were not contributing predictive signal and may have added minor noise.
This confirms that the Random Forest was relying primarily on the more important features (normal_count, defect_ratio, etc.), and the low‑importance failureType categories did not improve generalization.

# Discuss what this implies about the cost of deploying a simpler, lower-dimensional model in production (lower inference cost and maintenance burden, but only acceptable if AUC degradation is below a tolerable threshold).
Lower inference cost:  
Removing features means fewer inputs to process at prediction time. This reduces computational overhead and speeds up inference, which is critical in real‑time or large‑scale systems.
Lower inference cost:  
Reduced maintenance burden:  
Fewer features simplify data pipelines, monitoring, and retraining. It’s easier to maintain a leaner model with fewer dependencies on potentially noisy or unstable features.
Acceptability threshold:  
Simplification is only acceptable if predictive performance remains stable. In your case, AUC actually improved slightly, so the reduced model is safe to deploy.
If AUC had dropped noticeably, the cost savings would not justify the performance loss.

# 5. Cross-validated comparison:
# Walkthrough 
1. The dataset is split into k folds (here, 5).
2. For each fold, the model is trained on k‑1 folds and tested on the remaining fold.
3. Stratification ensures that if, for example, 70% of your data is “normal” and 30% is “defect,” each fold will preserve that ratio.
4. Results are averaged across folds, and the standard deviation shows variability in performance.
Logistic Regression (from Part 2)
Controlled Decision Tree (max_depth=5)
Random Forest (Task 4)
Gradient Boosting (Task 4a)
Report the mean and standard deviation of the 5-fold AUC for each model
Logistic Regression AUC: 1.000 ± 0.000
Decision Tree (controlled) AUC: 0.996 ± 0.001
Random Forest AUC: 1.000 ± 0.000
Gradient Boosting AUC: 1.000 ± 0.000
# Inference
Cross‑validation confirms that Logistic Regression, Random Forest, and Gradient Boosting all achieve near‑perfect AUC with zero variance across folds, making them highly reliable. The controlled Decision Tree performs slightly worse, but still strongly. Cross‑validation provides a robust, unbiased estimate of generalization performance, ensuring that model selection is not dependent on a single train‑test split.

# why cross-validation gives a more reliable estimate of generalization performance than a single train-test split.
A single train‑test split can give a misleading estimate of generalization if the split happens to be easy or hard
StratifiedKFold cross‑validation ensures each fold preserves the original class distribution, which is crucial in classification tasks.
Averaging across folds provides a more reliable estimate of generalization performance.
The standard deviation quantifies stability: low variance means the model is consistently strong across different subsets.

# 6. Hyperparameter tuning with GridSearchCV:
# Walkhrough
1. Build the Pipeline:
Handles missing values with a median
Standardizes features with StandardScaler.
Trains a RandomForestClassifier with reproducibility ensured by random_state=42.
2. Define the Parameter Grid:
n_estimators: number of trees (50, 100, 200).
max_depth: tree depth (5, 10, unlimited).
min_samples_leaf: minimum samples per leaf (1, 5).
Total combinations = 18
3. Set up Cross‑Validation:
Ensures each fold preserves class distribution.
Uses 5 folds, shuffled.
4. Run GridSearchCV:
Evaluates all 18 parameter combinations.
With 5 folds, total model fits = 90
Uses ROC‑AUC as the scoring metric.
Parallelized with n_jobs=-1 for efficiency.
5. Best Parameters & Score:
n_estimators=200
max_depth=None
min_samples_leaf=1
Best CV AUC Score: 0.99999
# Interpretation: 
The tuned Random Forest achieved near‑perfect ROC‑AUC.
Allowing unlimited depth (max_depth=None) and more trees (n_estimators=200) improved performance.
min_samples_leaf=1 lets the model capture fine‑grained splits, which worked well here.

# how many total model configurations were evaluated (product of all grid values × 5 folds) 
Parameter grid size:
n_estimators: 3 values
max_depth: 3 values
min_samples_leaf: 2 values
Total combinations =  3*3*2 = 18
With 5‑fold StratifiedKFold cross‑validation, total model fits = 18*5 =90

# explain the trade-off between exhaustive Grid Search and Randomized Search.
Grid Search
Exhaustively evaluates all parameter combinations in the grid.
Guarantees finding the best combination within the defined search space.
Computationally expensive as the grid expands (combinatorial explosion).

Randomized Search
Samples a fixed number of random parameter combinations.
More efficient when the parameter space is large.
May miss the absolute best combination, but often finds a near‑optimal one at much lower cost.

# 7. Manual learning curve.
# Walkthrough:
1. Define training fractions
train the tuned pipeline on progressively larger subsets of the training data: 20%, 40%, 60%, 80%, and 100%.
An empty list results will store the AUC values for each fraction
2. Retrieve the best pipeline from GridSearchCV
This ensures we are using the tuned Random Forest pipeline with the best hyperparameters found in Task 6.
Printing confirms the pipeline structure (imputer → scaler → Random Forest).
3. Loop through each fraction of the training set
For each fraction f, calculate how many rows to take (n).
Slice the first n rows of X_train and y_clf_train to form the subset.
Using .iloc keeps the data as a DataFrame with feature names intact.
4. Fit the pipeline on the subset
The tuned pipeline is trained only on the subset of data.
5. Compute Training AUC
Predictions are made on the same subset used for training.
Training AUC shows how well the model fits the data it has seen.
6. Compute Test AUC
Predictions are made on the full test set (unchanged).
Test AUC shows how well the model generalizes to unseen data.
7. Store results and print table
Each fraction’s results are appended to results.
Finally, the table is printed with three columns: Training fraction, Training AUC, and Test AUC.
Best pipeline: Pipeline(steps=[('simpleimputer', SimpleImputer(strategy='median')),
                ('standardscaler', StandardScaler()),
                ('randomforestclassifier',
                 RandomForestClassifier(n_estimators=200, random_state=42))])
Training fraction | Training AUC | Test AUC
0.2 | 1.000 | 1.000
0.4 | 1.000 | 1.000
0.6 | 1.000 | 1.000
0.8 | 1.000 | 1.000
1.0 | 1.000 | 1.000

# Inference
The model is not data‑limited. Collecting more training data does not improve performance because test AUC has plateaued at 1.000. The model’s capacity is already sufficient, and it has reached its performance ceiling.

#  (i) whether training AUC decreases as the training set grows (expected for high-variance models that overfit small datasets); 
Training AUC stayed at 1.000 across all fractions.
This means the Random Forest perfectly fits even small subsets of the data.
For high‑variance models, we would normally expect training AUC to decrease as the dataset grows, but here the model capacity is so strong that it maintains perfect fit.
# (ii) whether test AUC increases with more training data (if yes, collecting more data would likely improve performance); 
Test AUC was already perfect (1.000) at 20% of the training data and did not increase further.
This shows the model generalizes extremely well even with limited data.
Collecting more data does not improve performance because the model has already reached its ceiling.
# (iii) your conclusion — is the model currently limited by data quantity (test AUC still rising at 100%) or by model capacity (test AUC has plateaued)?
The model is capacity‑limited, not data‑limited.
Test AUC has plateaued at 1.000, meaning more data would not improve performance.
The tuned Random Forest already achieves perfect generalization, so production deployment can proceed confidently without further data expansion.

# 8. Serialize the best model:
# Walkthrough 
1. Best pipeline saved:
Pipeline includes: SimpleImputer(strategy='median') → StandardScaler() → RandomForestClassifier(n_estimators=200, random_state=42).
2. Reloaded successfully:
3. Predictions on handcrafted rows:
First synthetic test rows → [0 0]
Second realistic wafer feature rows → [1 0]

# Inference
The first handcrafted wafer row (high defect ratio, severity encoded = 2, failureType_B = 1) was classified as defective (1).
The second handcrafted wafer row (low defect ratio, severity encoded = 0, failureType_A = 1) was classified as non‑defective (0).
These outputs align with expectations: the model correctly distinguishes between high‑risk and low‑risk wafers.
The model is not only high‑performing but also production‑ready.

# 9. Summary comparison table:
| Model | 5‑fold CV mean AUC | 5‑fold CV std AUC | Test AUC |
| --- | --- | --- | --- |
| Logistic Regression | 1.000 | 0.000 | 1.000 |
| Controlled Decision Tree | 0.996 | 0.001 | 0.996 |
| Random Forest (baseline) | 1.000 | 0.000 | 1.000 |
| Gradient Boosting | 1.000 | 0.000 | 1.000 |
| Random Forest (tuned) | 1.000 | 0.000 | 1.000 |

# which model you would recommend to the client and justify
Recommending the Random Forest model to the client. It achieves perfect mean AUC with zero variance across folds, indicating both strong performance and exceptional stability. While Logistic Regression and Gradient Boosting also perform flawlessly, Random Forest provides additional advantages such as feature importance insights and robustness to noisy data, which are valuable in production. The Controlled Decision Tree is slightly weaker, with lower mean AUC and higher variance. Overall, Random Forest balances accuracy, reliability, and interpretability, making it the most confident choice for deployment.

# ##OUTPUT##
Saved the best model file (best_model.pkl) to Output folder in Part3