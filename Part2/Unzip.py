import os
import zipfile

# Get the directory where this script lives (Part2)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Path to the zip file in Part1 (relative to Part2)
zip_path = os.path.join(script_dir, "..", "Part1/Output", "cleaned_data.zip")
# Destination folder inside Part1
extract_dir = os.path.join(script_dir, "..", "Part1/Output")
os.makedirs(extract_dir, exist_ok=True)

# Unzip the file
with zipfile.ZipFile(zip_path, 'r') as zipf:
    zipf.extractall(extract_dir)

print(f"✅ Extracted {zip_path} into {extract_dir}")
