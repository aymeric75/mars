import pandas as pd
import matplotlib.pyplot as plt

features = pd.read_parquet("NFLX_2025-12-09_features.parquet")
features = features.rename(columns={"i": "Step"})
messages = pd.read_parquet("NFLX_2025-12-09_messages.parquet")


#

print(features)


sub_messages = messages[(messages["Time"] >= 34200000226319) & (messages["Time"] <= 57599998528372)]

print(sub_messages)










##################### ddze s
