import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# 1. Load and Inspect
script_dir = os.path.dirname(os.path.abspath(__file__))
extract_dir = os.path.join(script_dir, "wm811k_data")
output_file_path = os.path.join(extract_dir, "WM811K.csv")
# Check folder and file existence
if not os.path.exists(extract_dir):
    raise FileNotFoundError(f"Required folder not found: {extract_dir}")
if not os.path.isfile(output_file_path):
    raise FileNotFoundError(f"Required file not found: {output_file_path}")
df = pd.read_csv(output_file_path)
# Print the first five rows
print("First five rows:")
print(df.head())
# Print the column data types
print("\nColumn data types:")
print(df.dtypes)
# Print the DataFrame shape
print("\nDataFrame shape:")
print(df.shape)

# 2. Null Value Analysis
# Replace [] with NaN
df["trianTestLabel"] = df["trianTestLabel"].replace("[]", np.nan)
df["failureType"]    = df["failureType"].replace("[]", np.nan)
# Now check null counts and percentages
null_counts = df.isnull().sum()
null_percent = (null_counts / df.shape[0]) * 100
null_table = pd.DataFrame({"Null Count": null_counts, "Null %": null_percent})
print("\nNull Value Table:\n", null_table)
high_nulls = null_table[null_table["Null %"] > 20]
print("\nColumns >20% nulls:", high_nulls.index.tolist())
for col in df.select_dtypes(include=[np.number]).columns:
    if null_percent[col] <= 20:
        # Numeric column → fill with median
        df[col] = df[col].fillna(df[col].median())
    else:
        # Categorical column → fill with mode
        df[col] = df[col].fillna(df[col].mode()[0])
print("\nShape:", df.shape)
# Drop rows with any NaN values
df = df.dropna()
# Check shape after dropping
print("\nShape after dropping nulls:", df.shape)

# 3. Duplicate Detection
# Count duplicates
duplicate_count = df.duplicated().sum()
print("Number of duplicate rows:", duplicate_count)
# Rows before removal
rows_before = df.shape[0]
# Remove duplicates
df_clean = df.drop_duplicates()
# Rows after removal
rows_after = df_clean.shape[0]
# How many rows were removed
removed_rows = rows_before - rows_after
print("Rows removed:", removed_rows)

# Compare null percentages before vs after
null_percent_before = df.isnull().mean() * 100
null_percent_after = df_clean.isnull().mean() * 100

# Combine into a comparison table
null_comparison = pd.DataFrame({
    "Before %": null_percent_before,
    "After %": null_percent_after
})

print(null_comparison)

# 4. Data Type Correction
mem_before = df.memory_usage(deep=True).sum()
print("Memory usage before:", mem_before)
# waferIndex → convert to numeric
df["waferIndex"] = pd.to_numeric(df["waferIndex"], errors="coerce")
# convert dieSize to numeric
df["dieSize"] = pd.to_numeric(df["dieSize"], errors="coerce")

# Convert categorical columns
for col in ["lotName", "trianTestLabel", "failureType"]:
    df[col] = df[col].astype("category")

#structured wafer maps
def to_array(wm_str):
     # Step 1: Remove ellipsis placeholders
    wm_str = wm_str.replace("...", "")
    # Step 2: Collapse multiple spaces
    wm_str = re.sub(r"\s+", " ", wm_str.strip())
    # Step 3: Split into rows
    rows = wm_str.strip("[]").split("] [")
    # Step 4: Convert each row into list of ints
    matrix = []
    for row in rows:
        row_clean = row.strip("[]").strip()
        if row_clean:  # skip empty rows
            matrix.append(list(map(int, row_clean.split())))

    return np.array(matrix, dtype=int)

# Apply conversion
df["waferMap_array"] = df["waferMap"].apply(to_array)

def count_defects(wm):
    try:
        arr = wm
        return int(np.sum(arr == 2))
    except Exception as e:
        return 0

def count_normal(wm):
    try:
        arr = wm
        return int(np.sum(arr == 1))
    except Exception as e:
        return 0

# Apply to DataFrame
df["defect_count"] = df["waferMap_array"].apply(count_defects)
# Count normal dies (value = 1)
df["normal_count"] = df["waferMap_array"].apply(count_normal)

# Compute defect ratio
df["defect_ratio"] = df["defect_count"] / df["dieSize"]

# Keep only rows where at least one of defect_count or normal_count is non-zero
df = df[~((df["defect_count"] == 0) & (df["normal_count"] == 0))]
print("\nShape after removing rows with both counts = 0:", df.shape)
# Optional: confirm no such rows remain
print(df[(df["defect_count"] == 0) & (df["normal_count"] == 0)])

print(df.dtypes)
# Memory usage after conversion
mem_after = df.memory_usage(deep=True).sum()
print("Memory usage after:", mem_after)

# Report difference
print("Memory saved:", mem_before - mem_after)

# 5. Descriptive Statistics & Skewness
print("\nDescriptive statistics:\n", df.describe())
skews = df.skew(numeric_only=True)
print("\nSkewness:\n", skews)
most_skewed = skews.abs().idxmax()
print("Most skewed column:", most_skewed)

# 6. Outlier Detection (IQR)
for col in ["dieSize", "waferIndex", "defect_count", "normal_count", "defect_ratio"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    count_outliers = outliers.shape[0]
    print(f"\nColumn: {col}")
    print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
    print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
    print(f"Number of outliers: {count_outliers}")

# 7. Visualizations
# Create 'plots' folder at the same level as the script if it doesn't exist
# Get the folder where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "plots")
os.makedirs(output_dir, exist_ok=True)
# Line plot of dieSize by row index
plt.figure(figsize=(10,5))
plt.plot(df.index, df['dieSize'], color='blue')
# Add title and labels
plt.title("Line Plot of Die Size by Row Index")
plt.xlabel("Row Index")
plt.ylabel("Die Size")
plt.savefig(os.path.join(output_dir, "lineplot_dieSize.png"), dpi=300, bbox_inches='tight')
plt.close()

# Group by categorical column and compute mean of numeric column
# Explicitly set observed=True to adopt the future default
mean_values = df.groupby('failureType', observed=True)['dieSize'].mean()
# Plot as bar chart
mean_values.plot.bar(color='skyblue', figsize=(10,5))
# Add title and labels
plt.title("Mean Die Size Across Failure Types")
plt.xlabel("Failure Type")
plt.ylabel("Mean Die Size")
plt.savefig(os.path.join(output_dir, "bar_mean_dieSize_failureType.png"), dpi=300, bbox_inches='tight')
plt.close()

# Histogram of dieSize
plt.figure(figsize=(8,5))
sns.histplot(df['dieSize'], bins=20, color='steelblue', edgecolor='black')
# Add title and labels
plt.title("Histogram of Die Size")
plt.xlabel("Die Size")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, "hist_dieSize.png"), dpi=300, bbox_inches='tight')
plt.close()

# Scatter plot between dieSize and defect_count
plt.figure(figsize=(8,6))
sns.scatterplot(x=df['dieSize'], y=df['defect_count'], alpha=0.5)
# Add title and labels
plt.title("Scatter Plot: Die Size vs Defect Count")
plt.xlabel("Die Size")
plt.ylabel("Defect Count")
plt.savefig(os.path.join(output_dir, "scatter_dieSize_defectCount.png"), dpi=300, bbox_inches='tight')
plt.close()

# Box plot of dieSize split by failureType
plt.figure(figsize=(10,6))
sns.boxplot(x='failureType', y='dieSize', data=df)
# Add title and labels
plt.title("Box Plot of Die Size Across Failure Types")
plt.xlabel("Failure Type")
plt.ylabel("Die Size")
plt.savefig(os.path.join(output_dir, "box_dieSize_failureType.png"), dpi=300, bbox_inches='tight')
plt.close()

# 8. Correlation Heatmap
# Compute correlation matrix
corr_matrix = df.corr(numeric_only=True)
# Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numeric Columns")
plt.savefig(os.path.join(output_dir, "heatmap_correlations.png"), dpi=300, bbox_inches='tight')
plt.close()
# Identify pair with highest absolute correlation
abs_corr = corr_matrix.abs()
# Mask diagonal
for i in range(len(abs_corr)):
    abs_corr.iloc[i,i] = 0
max_pair = abs_corr.unstack().idxmax()
max_value = abs_corr.unstack().max()
print(f"\nHighest absolute correlation: {max_pair} = {max_value}")

# 9a. Imputation Strategy Comparison
# Compute mean and median for the two most skewed columns
for col in ['dieSize', 'defect_count']:
    mean_val = df[col].mean()
    median_val = df[col].median()
    print(f"Column: {col} | Mean: {mean_val:.4f} | Median: {median_val:.4f}")
# Apply median imputation for missing values
df['dieSize'] = df['dieSize'].fillna(df['dieSize'].median())
df['defect_count'] = df['defect_count'].fillna(df['defect_count'].median())
# Confirm no nulls remain
print("\nNull values after imputation:")
print(df[['dieSize','defect_count']].isnull().sum())

# 9b. Spearman vs Pearson Correlation
# Pearson correlation matrix
pearson_corr = df.corr(method='pearson', numeric_only=True)
# Spearman correlation matrix
spearman_corr = df.corr(method='spearman', numeric_only=True)
# Absolute difference between Spearman and Pearson
diff_matrix = (spearman_corr - pearson_corr).abs()
# Identify top 3 pairs with largest differences
# Mask diagonal and upper triangle to avoid duplicates
mask = np.triu(np.ones(diff_matrix.shape), k=0).astype(bool)
diff_matrix_masked = diff_matrix.mask(mask)
top3_pairs = (
    diff_matrix_masked.unstack()
    .dropna()
    .sort_values(ascending=False)
    .head(3)
    .reset_index()
)
top3_pairs.columns = ['Variable_1', 'Variable_2', 'Abs_Diff']

print("\nPearson correlation matrix:\n", pearson_corr)
print("\nSpearman correlation matrix:\n", spearman_corr)
print("\nAbsolute differences:\n", diff_matrix)
print("\nTop 3 pairs with largest |Spearman - Pearson|:")
for _, row in top3_pairs.iterrows():
    print(f"{row['Variable_1']}  {row['Variable_2']}  {row['Abs_Diff']:.6f}")

# 9c. Grouped Aggregation
#group by failureType (categorical) and aggregate dieSize (numeric)
group_stats = df.groupby('failureType', observed=False)['dieSize'].agg(['mean', 'std', 'count'])
print("\n",group_stats)
# Identify groups with highest mean and highest std
highest_mean_group = group_stats['mean'].idxmax()
highest_std_group = group_stats['std'].idxmax()
# Compute ratio of highest mean to lowest mean
mean_ratio = group_stats['mean'].max() / group_stats['mean'].min()

print(f"\nGroup with highest mean: {highest_mean_group}")
print(f"Group with highest std: {highest_std_group}")
print(f"Ratio of highest mean to lowest mean: {mean_ratio}")


# 10. Save Cleaned Dataset
# Get the folder where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "Output")
os.makedirs(output_dir, exist_ok=True)
# Path to your file
file_path = os.path.join(output_dir, "cleaned_data.csv")
df.to_csv(file_path, index=False)
print("\nCleaned dataset saved as cleaned_data.csv")

import zipfile
# Check file size in MB
size_bytes = os.path.getsize(file_path)
size_mb = size_bytes / (1024 * 1024)
print(f"File size: {size_mb:.2f} MB")
# Zip the file for git push
zip_path = os.path.join(output_dir, "cleaned_data.zip")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(file_path, os.path.basename(file_path))
print(f"File compressed to: {zip_path}")
## Unzip the file for other Parts in the Capstone project
#os.makedirs(output_dir, exist_ok=True)
#with zipfile.ZipFile(zip_path, 'r') as zipf:
#    zipf.extractall(output_dir)
#print(f"File extracted to: {output_dir}")

