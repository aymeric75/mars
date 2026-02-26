import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("LOBSTER_META_2025-10-01_messages_10.parquet")

print(df)
# print(df["Message_Type"].unique())
# print((df["Message_Type"] == 12).sum())
# # 64065
# print(df[df["Message_Type"] == 12])
exit()




df["Mid_Price"] = (
    (df["Ask_Price_1"] + df["Bid_Price_1"]) / 2
).where(
    df["Ask_Price_1"].notna() & df["Bid_Price_1"].notna()
)


valid_indices = df.index[df["Mid_Price"].notna()].tolist()

#print(valid_indices)
print(len(valid_indices)) # 1696627


print(df)


# Get the starting timestamp (ns)
start_time = df["Time"].iloc[0]

# Compute minute index relative to start
df["minute"] = ((df["Time"] - start_time) // 60_000_000_000).astype(int)

print(df)

# --- Detect mid price changes (ignore NaNs automatically) ---
df["Mid_Change"] = df["Mid_Price"].diff().ne(0)

# Remove rows where mid price is NaN
valid = df["Mid_Price"].notna()

# Count changes per minute
changes_per_minute = (
    df[valid & df["Mid_Change"]]
    .groupby("minute")
    .size()
)

# --- Plot ---
plt.figure()
plt.bar(changes_per_minute.index, changes_per_minute.values)
plt.xlabel("Minute")
plt.ylabel("Number of Mid Price Changes")
plt.title("Mid Price Changes per Minute")
plt.savefig("mid_price_changes_per_minute.png", dpi=300, bbox_inches="tight")
plt.close()
