# ------WM811K Wafer Map Dataset: From Data Cleaning to Explainable AI------ #
Choosen the WM‑811K semiconductor wafer map dataset for the Applied AI & ML Essentials — Capstone Project, because it is one of the largest publicly available, real‑world datasets for defect pattern recognition in semiconductor manufacturing.

# Installation
Install the required library to run the python script in Part1 folder
```bash 
    pip install pandas numpy matplotlib seaborn kaggle os re pickle zipfile
```
# OS
Windows OS
# IDE
Visual Studio Code
# PreRequisties
Make sure the dataset file(WM811k.csv) is available in the Part1\wm811k_data folder (or) If Part1\wm811k_data folder contains WM811k.zip, run the Unzip.py to get the dataset file(WM811k.csv) in the Part1\wm811k_data folder (or) follow the Instruction on how to download WM811K Dataset from Kaggle, as mentioned at Notes section in the end of the README.md. 
# Run the python script
DA_C_EA.py file for Part 1

# Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis
# WM811K Wafer Map Dataset Cleaning & Exploratory Data Analysis
# Scenario
A data file(WM811k.csv) has just arrived from the client in raw form. Before any modeling can begin, the data must be inspected, cleaned, and understood. the job is to produce a clean dataset and a clear picture of what the data contains, where it has gaps, and which variables are likely to matter and produce a clean dataset (cleaned_data.csv) 

# TASK
# Data Cleaning Steps
# 1. Load the dataset
# Walkthrough
1. Loaded using pd.read_csv() 
2. Printed first five rows, .dtypes, and shape.
3. Initial inspection revealed mixed types and string‑encoded wafer maps.
# Interpretation
waferMap → stored as object strings with embedded arrays. Requires parsing into NumPy arrays for defect analysis.
dieSize → numeric (float64), suitable for regression.
lotName → object, should be converted to category
waferIndex → numeric (float64).
trianTestLabel and failureType → repetitive string values stored as object, should be converted to category.
Dataset size → 811,457 rows × 6 columns, large enough to require careful memory optimization.

# 2. Null value analysis
# Observation: 
    In the raw dataset, trianTestLabel and failureType contained empty square brackets [] instead of valid labels.
    Hence, Replaced all [] values with NaN 
    [] is not a valid category, but a placeholder for missing information.
    Converting to NaN ensures Pandas treats them as nulls, allowing consistent handling in imputation and null percentage analysis.
# Walkthrough
1. Computed null counts and percentages with df.isnull().sum() and (df.isnull().sum()/df.shape[0])*100.
2. Columns exceeding 20% null rate flagged.
3. For numeric columns below 20% nulls, imputed with median using fillna(df[col].median()).
    # Justification: 
    Median is robust to skewness.
    Median better represents central tendency
# Inference:
Two categorical columns (trianTestLabel, failureType) had ~78.7% missing values.
Retaining rows with missing categorical labels would introduce noise and bias.
Dropping them preserves dataset integrity for downstream modeling.
Although the dataset size decreased, 172k rows remain — still sufficient for robust analysis.

# 3. Duplicate detection and removal:
# Walkthrough
df.duplicated().sum() used to count duplicates
Removed with df.drop_duplicates().
Reported rows removed (0) and confirmed null percentages unchanged.
# Inference: 
Null percentages identical before and after removal.

# 4. Data Type Correction
Numeric conversion: waferIndex and dieSize were enforced as numeric (float64) 
Categorical conversion: lotName, trianTestLabel, and failureType were converted to category dtype

Additionaly, transformed the raw wafer map strings into structured, analyzable data
# Why, Wafer Map Parsing Was Done
The original waferMap column contained string representations of arrays 
They couldn’t be used for numeric analysis or defect detection.
By parsing them into NumPy arrays, each wafer map becomes a structured 2D grid of integers, where values represent die states (e.g., 0 = empty, 1 = normal, 2 = defective).
This conversion enables computations such as defect counts and visualization.
# Why Derived Features Were Created
defect_count → quantifies how many dies are defective (2).
normal_count → quantifies how many dies are normal (1).
defect_ratio → expresses the proportion of defective dies relative to total die size.
These provide numeric summaries of each wafer map, making them suitable for Statistical analysis and 
Predictive modeling 
# Why Row Filtering Was Done
Some wafer maps contained no defects and no normal dies (all zeros or invalid entries).
Such rows add no analytical value and can distort statistics.
Removing them reduced the dataset to 74,206 rows × 10 columns, ensuring only meaningful wafer maps remain.

# Walkthrough
1. Memory usage before is calculated
2. Enforce numberic conversion on dieSize and waferIndex
3. Convert categorical columns identified
4. transformed the raw wafer map strings into structured, analyzable data
5. Memory ysage after is calculated
Memory usage compared before vs after with df.memory_usage(deep=True).sum()
# Inference:
Before conversion: ~140MB.
After conversion: ~562 MB.
Reason: Parsing wafer maps into full arrays expanded storage requirements.
Benefit: Although memory usage increased, the dataset is now structured and analyzable, enabling defect detection computation — which raw strings could not support.

# Descriptive Statistics
# 5. Descriptive Statistics & Skewness
# Walkthrough
Ran df.describe() on numeric columns.
Computed skewness with df[col].skew().
Most skewed column: defect_count (+3.64, positive skew).

# Interpretation: 
Positive skew → mean > median. Median chosen for imputation.

# what positive vs negative skew means for that column's distribution 
Positive skew (right‑skewed)  
The distribution has a long tail on the right. Most values are clustered at the lower end, but a few very large values pull the mean upward.
Example: defect_count (skewness ≈ 3.64) — most wafers have few defects, but a small number have extremely high counts.
Negative skew (left‑skewed)  
The distribution has a long tail on the left. Most values are clustered at the higher end, but a few very small values pull the mean downward.
Example: waferIndex (skewness ≈ −0.0027, essentially symmetric, but if it were more negative, the mean would be dragged left).
# what consequence it has for imputing missing values with the mean.
Positive skew:  
The mean is larger than the median. If impute missing values with the mean, will insert values that are higher than typical, which can distort the dataset and exaggerate the tail.
Negative skew:  
The mean is smaller than the median. Imputing with the mean will insert values that are lower than typical, again biasing the dataset toward the tail.
Symmetric distributions:  
Mean ≈ Median. Imputing with the mean is safe because it represents the “center” of the data well.

# 6. Applied IQR method to dieSize and defect_ratio.
# Walkthrough
1. Documented counts of outliers with
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 − Q1
Lower bound = Q1 − 1.5 × IQR
Upper bound = Q3 + 1.5 × IQR
Rows outside these bounds are flagged as outliers.
# Decision: 
Retain outliers for now.
# Reasoning:
Outliers also represent real defect patterns.
# Inference
Will consider capping/transforming in Part 2.
dieSize → Cap extreme values at IQR bounds to reduce distortion while preserving range.
waferIndex → Retain (no outliers detected).
defect_count → Retain extreme values, as they represent meaningful defect events.
normal_count → Cap extreme values to stabilize variance.
defect_ratio → Retain extreme values, since they highlight rare but critical defect ratios.

# Visualizations
# 7.  Plot
# Walkthrough
Line Plot: Numeric variable over index → wafer size trend.
Bar Chart: Mean dieSize across failureType categories → defect impact visible.
Histogram: dieSize distribution → right‑skewed, long tail.
Scatter Plot: dieSize vs defect_ratio → moderate positive correlation.
Box Plot: dieSize split by failureType → differences in medians and spreads documented
All five plots are now saved as .png files inside the plots folder:
lineplot_dieSize.png
bar_mean_dieSize_failureType.png
hist_dieSize.png
scatter_dieSize_defectCount.png
box_dieSize_failureType.png

# Inference
Line Plot
Shows how dieSize varies across row index. Useful for spotting anomalies across wafers.
Bar Chart
Compares average dieSize across categories of failureType. Highlights categorical influence on wafer characteristics.
Histogram
Distribution of dieSize is positively skewed. Most wafers cluster around 500–560, with a few much larger dies.
Scatter Plot
Shows a positive relationship between dieSize and defect_count. Larger wafers tend to have more defects, though correlation is moderate.
Box Plot
Median dieSize differs across failure types. Spread varies, with some categories showing wider variability. Outliers are visible, reinforcing skewness and outlier analysis.

# Correlation Analysis
# 8. Correlation heat map
# Walkthrough
1. Correlation matrix: Computed for all numeric columns (dieSize, waferIndex, defect_count, normal_count, defect_ratio).
2. Heatmap: Saved to plots/heatmap_correlations.png. Shows strength and direction of relationships.
3. Strongest correlation: Between defect_count and defect_ratio (≈ 0.96).

# Inference:
Strong positive correlation is expected because defect_ratio is mathematically derived from defect_count and dieSize

# whether this correlation might indicate a causal relationship or whether a third variable could explain it,
Strong positive correlation is expected because defect_ratio is mathematically derived from defect_count and dieSize.
It does not imply causation — the relationship is formulaic.
A plausible third variable (dieSize) explains the dependency: larger dies can produce more defects, which increases both defect count and defect ratio.
# Alternative explanation: 
The correlation reflects structural dependence rather than a causal mechanism. In modeling, including both variables may introduce redundancy, so dimensionality reduction or feature selection should be considered.

# 9a. Imputation Strategy Comparison
# Walkthough
1. Columns chosen:
dieSize (skewness ≈ +1.77)
defect_count (skewness ≈ +3.64)
Both columns are positively skewed.
2. The mean is greater than the median in each case, pulled upward by extreme high values
Applied median imputation with fillna().
Column: dieSize | Mean: 559.4155 | Median: 533.0000
Column: defect_count | Mean: 81.0155 | Median: 79.0000
3. Confirmed no nulls remain
# Inference:
Median imputation applied to skewed columns (dieSize, defect_count). This ensures that missing values are filled with a value representative of the majority of wafers, not distorted by rare extreme cases.

# 9b. Pearson and Spearman Correlation
# Walkthrough
1. Pearson correlation matrix
Compute with df.corr(method='pearson')
2. Spearman Rank Correlation
Computed with df.corr(method='spearman').
3. Compared against Pearson.
4. Identified three pairs with largest |Spearman − Pearson| differences.
Top 3 pairs with largest |Spearman - Pearson|:
dieSize  defect_count  0.316513
dieSize  defect_ratio  0.280311
defect_count  normal_count  0.206174
# Interpretation:
|Spearman| > |Pearson| → monotonic but non‑linear.
|Pearson| ≥ |Spearman| → approximately linear.
Decision: Use Spearman for feature selection when monotonic but non‑linear relationships are detected.
#  (a) whether the relationship appears to be monotonic but non-linear (if |Spearman| > |Pearson|, indicating the variables move together consistently but not proportionally) or approximately linear (if |Pearson| ≥ |Spearman|);
Pair 1: dieSize vs defect_count
Pearson ≈ 0.01 (almost no linear relationship).
Spearman ≈ -0.31 (moderate monotonic negative).
Conclusion: Relationship is monotonic but non-linear. Larger dies consistently relate to fewer defects, but not proportionally.
Feature selection: Rely on Spearman.
Pair 2: dieSize vs defect_ratio
Pearson ≈ -0.24 vs Spearman ≈ -0.52.
Conclusion: Monotonic but non-linear. As die size increases, defect ratio consistently decreases.
Feature selection: Use Spearman.
Pair 3: defect_count vs normal_count
Pearson ≈ -0.46 vs Spearman ≈ -0.67.
Conclusion: Stronger monotonic negative than linear. As defect count rises, normal count falls consistently, but not in a straight-line proportion.
# (b) which correlation measure you will rely on for feature-selection
Feature selection: Prefer Spearman.

# 9c: Grouped Aggregation
# Walkthough
1. group by failureType (categorical) and aggregate dieSize (numeric)
2. Identify groups with highest mean and highest std
                        mean        std  count
failureType                                  
[['Center']]     537.115554  62.069705   2726
[['Donut']]      633.652174  96.052357     23
[['Edge-Loc']]   584.115385  80.095816   1638
[['Edge-Ring']]  617.304609  85.079005    499
[['Loc']]        578.201681  81.822845   1071
[['Near-full']]  595.260870  98.981946     92
[['Random']]     573.371429  80.689173    245
[['Scratch']]    590.592715  79.382352    302
[['none']]       558.727526  72.167119  67610
3. Compute ratio of highest mean to lowest mean
Group with highest mean: [['Donut']] (≈ 634).
Group with highest standard deviation: [['Near-full']] (≈ 99).
# Inference
The Donut group’s mean is about 18% higher than the lowest group mean. This difference is noticeable but not dramatic. It suggests that the categorical feature (failureType) carries some predictive signal, though it’s not strong enough to be used in isolation.
# (a) name the group with the highest mean and the group with the highest standard deviation
Group with highest mean: [['Donut']]
Group with highest std: [['Near-full']]
# (b) state whether high within-group standard deviation is a concern for a predictive model using this categorical feature (high within-group variance means the feature alone is insufficient to predict the target reliably for members of that group)
The high within-group standard deviation for [['Near-full']] means that failureType alone is not a reliable predictor of dieSize for wafers in this group. High variance reduces predictive reliability.
# (c) compute the ratio of the highest group mean to the lowest group mean and state whether this ratio is large enough to suggest the categorical feature carries predictive signal.
Ratio of highest to lowest mean: ≈ 1.18.
This ratio is relatively small, suggesting that while failureType has some influence on dieSize, the predictive signal is weak.

# 10. Output
# Walkthough
1. Get the folder where the script is located
2. Clean dataset saved as cleaned_data.csv in Output folder
3. Additionaly, add the code to zip the cleaned_data.csv file to avoid the GitHub rejecting push because of have large files (>100 MB) 



# NOTES
# Instruction on how to download WM811K Dataset from Kaggle
1. Ensure Kaggle CLI is installed and configured along with the libraries listed
```bash
    pip install kaggle subprocess pandas zipfile os
```
2. Install python 3.8 for Windows x86-64 from https://www.python.org/downloads/release/python-380/
Add to PATH
3. Create and activate the virtual environment 
```bash
py -3.8 -m venv wm811k_env 
wm811k_env\Scripts\activate
py -3.8 -m pip install --user pandas==0.25.3 
pip install pandas==0.25.3 numpy==1.19.5
```
4. Run the Download_Extract.py script
5. Switch back to Python 3.14
```bash
deactivate
```
6. dataset(WM811K.csv) is saved to the wm811k_data in Part1