# Unzip the dataset file for other Part1 in the Capstone project
import zipfile
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
rawDS_dir = os.path.join(script_dir, "wm811k_data")
zip_path = os.path.join(rawDS_dir, "WM811K.zip")
with zipfile.ZipFile(zip_path, 'r') as zipf:
    zipf.extractall(rawDS_dir)
print(f"File extracted to: {rawDS_dir}")