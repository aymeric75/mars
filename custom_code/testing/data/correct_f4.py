import glob
import numpy as np
from datasets import load_dataset

# /scratch/project_2012747/mars_data/order_model/val/final
parquet_dir = "."
files = sorted(glob.glob(f"{parquet_dir}/*_features.parquet"))


print(files)
exit()

def add_f4(batch):
    t = np.asarray(batch["Time"], dtype=np.int64)
    t_sec = (t // 1_000_000_000).astype(np.int64)
    batch["f4"] = np.clip(t_sec - 34200, 0, 23399).astype(np.int64)
    return batch

for path in files:
    print(f"Processing {path}...")
    ds = load_dataset("parquet", data_files=path, split="train")
    ds = ds.map(add_f4, batched=True, batch_size=200_000, num_proc=1)
    ds.to_parquet(path)

print("All files updated.")
