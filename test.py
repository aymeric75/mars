from pathlib import Path
import pandas as pd
from datasets import load_dataset
import numpy as np

print(pd.read_parquet("AAPL_2025-11-03_features.parquet"))


# Path to your parquet file
parquet_path = "AAPL_2025-11-03_features.parquet"

def add_f4(batch):
    t = np.asarray(batch["Time"], dtype=np.int64)
    t_sec = (t // 1_000_000_000).astype(np.int64)
    batch["f4"] = np.clip(t_sec - 34200, 0, 23399).astype(np.int64)
    return batch

# Load parquet as HuggingFace dataset
ds = load_dataset("parquet", data_files=parquet_path, split="train")

# Recompute f4
ds = ds.map(add_f4, batched=True, batch_size=200_000, num_proc=1)

# Overwrite the same parquet file
ds.to_parquet(parquet_path)

print("Done. f4 column corrected.")


print(pd.read_parquet("AAPL_2025-11-03_features.parquet"))
