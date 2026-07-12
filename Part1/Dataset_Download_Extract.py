
#Step 1: Ensure Kaggle CLI is installed and configured
import subprocess
import pandas as pd
import zipfile
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
extract_dir = os.path.join(script_dir, "wm811k_data")
# Create the directory if it doesn't exist
os.makedirs(extract_dir, exist_ok=True)
# Step 2: Download the WM-811K dataset
subprocess.run(["kaggle", "datasets", "download", "-d", "qingyi/wm811k-wafer-map","-p", extract_dir], check=True)
# Step 3: Unzip the dataset
zip_path = os.path.join(extract_dir, "wm811k-wafer-map.zip")
os.makedirs(extract_dir, exist_ok=True)
# Extract all files
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_dir)
# Setup 4: Convert to csv format
pkl_path = os.path.join(extract_dir, "LSWMD.pkl")
df = pd.read_pickle(pkl_path)
df.to_csv(os.path.join(extract_dir, "WM811K.csv"), index=False)
print("✅ Saved as wm811k_data/WM811K.csv")

