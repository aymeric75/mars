import pandas as pd
import matplotlib.pyplot as plt

df_ = pd.read_parquet("NFLX_2025-12-09_snapshots.parquet")
print(df_.columns)
#print(len(pd.read_parquet("NFLX_2025-12-09_messages.parquet")))

# pd.read_parquet(parquet_path, columns=[]).shape[0]

# start_time = 34200000226319
# end_time = 57599998528372

# df = df_[(df_["Time"] >= start_time) & (df_["Time"] <= end_time)]


# # en fct du Message_type puis de la direction, tu affecte soit 0(S), 1(B), 2(C)
# df["Mars_type"] = 0

# # TYPE 2: cancel / delete
# df.loc[df["Message_Type"].isin([2, 3]), "Mars_type"] = 2

# # TYPE 1 : buy passive limit order
# df.loc[
#     ((df["Message_Type"] == 1) & (df["Direction"] == -1)),
#     "Mars_type"
# ] = 1

# # TYPE 3 : sell agressive order
# df.loc[
#     ((df["Message_Type"] == 4) & (df["Direction"] == -1)),
#     "Mars_type"
# ] = 3


# # TYPE 4 buy aggressive

# df.loc[
#     ((df["Message_Type"] == 4) & (df["Direction"] == 1)),
#     "Mars_type"
# ] = 4


# print(df)

# print(df["Mars_type"].value_counts().sort_index())



# # print(df["Mars_type"].unique())
# import matplotlib.pyplot as plt
# plt.figure()

# plt.hist(df["Mars_type"], bins=[-0.5,0.5,1.5,2.5,3.5,4.5])

# plt.xticks([0,1,2,3,4])
# plt.xlabel("Mars_type")
# plt.ylabel("Count")
# plt.title("Histogram of Mars_type")

# plt.savefig("mars_type_histogram.png", dpi=300, bbox_inches="tight")
# plt.close()
