import os
import re
import shutil
from datetime import datetime, date

# Source directory containing files like:
# AAPL_2025-11-01_messages.parquet
source_dir = "META"

# Destination folders
train_dir = "train" #os.path.join(source_dir, "train")
val_dir = "val" #os.path.join(source_dir, "val")
test_dir = "test" #os.path.join(source_dir, "test")

# Create destination folders if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Match files like AAPL_2025-10-01_messages.parquet
pattern = re.compile(r"^[A-Z]+_(\d{4}-\d{2}-\d{2})_(messages|snapshots|meta)\.parquet$")

# Date ranges
train_start = date(2025, 11, 1)
train_end   = date(2025, 11, 10)

val_start = date(2025, 11, 11)
val_end   = date(2025, 11, 15)

test_start = date(2025, 11, 16)
test_end   = date(2025, 11, 19)

for filename in os.listdir(source_dir):
    match = pattern.match(filename)
    if not match:
        continue

    file_date_str = match.group(1)
    file_date = datetime.strptime(file_date_str, "%Y-%m-%d").date()

    src_path = os.path.join(source_dir, filename)

    if train_start <= file_date <= train_end:
        dst_path = os.path.join(train_dir, filename)
        split = "train"
    elif val_start <= file_date <= val_end:
        dst_path = os.path.join(val_dir, filename)
        split = "val"
    elif test_start <= file_date <= test_end:
        dst_path = os.path.join(test_dir, filename)
        split = "test"
    else:
        continue


    # Avoid overwriting existing files
    if not os.path.exists(dst_path):
        print(f"Copying {filename} -> {split}/")
        shutil.copy2(src_path, dst_path)
    else:
        print(f"Skipping (already exists): {filename}")