import os
import pyarrow.parquet as pq
import pandas as pd

"""
folder_path = "/scratch/project_2012747/mars_data/order_model/train/final"

log_file = "corrupted_parquet_files.txt"

with open(log_file, "w") as log:
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".parquet"):
                filepath = os.path.join(root, file)
                try:
                    pq.ParquetFile(filepath)  # metadata check only
                    #print(f"OK: {filepath}")
                except Exception as e:
                    print(f"CORRUPTED: {filepath}")
                    log.write(filepath + "\n")
                    os.remove(filepath)

"""


print(pd.read_parquet("/scratch/project_2012747/mars_data/order_model/val/final/AAPL_2025-11-28_features.parquet"))