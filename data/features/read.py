import pandas as pd


ft= pd.read_parquet("AAPL_2025-11-28_messages.parquet")


print(ft.iloc[40:150])


print("ft")
