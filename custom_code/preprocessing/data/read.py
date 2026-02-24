import pandas as pd


messages = pd.read_parquet("LOBSTER_META_2025-10-01_snapshots_10.parquet")


print(messages)
