from pathlib import Path
import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # or: from tqdm import tqdm
ONE_MIN_NS = 60_000_000_000




def past_16min_indices_from_df(df: pd.DataFrame) -> pd.DataFrame:

    """ 
        Return:
            out of shape (16, len(df)) s.t. out[0] indicates for each order the past
            index in the df that corresponds to grossly 1 minute
    """
    
    d = df[["Time"]].sort_values("Time").reset_index(drop=True)
    t = d["Time"].to_numpy()

    out = {}
    n = len(t)

    for k in range(1, 17):
        pos = np.searchsorted(t, t - k * ONE_MIN_NS, side="right") - 1
        col = np.full(n, np.nan)
        valid = pos >= 0
        col[valid] = pos[valid]
        out[k - 1] = col

    res = pd.DataFrame(out)
    return res.dropna().astype("int64")


def next_1min_index_from_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df[["Time"]].sort_values("Time").reset_index(drop=True)
    t = d["Time"].to_numpy()

    pos = np.searchsorted(t, t + ONE_MIN_NS, side="left")

    col = np.full(len(t), np.nan)
    valid = pos < len(t)
    col[valid] = pos[valid]

    res = pd.DataFrame({0: col})
    return res.dropna().astype("int64")


def create_min16_plus1_indices_from_feature_parquets(input_dir: str, output_dir: str) -> None:

    """
    go over all feature parquets in input_dir (each has index starting at 0)
    for each df, for each row find last 16 indices (<=> 16m distance) and the next 1 minute index
    returns df for last16 indices, next1 index, and the updated df. 
    All 3 have same Index column (not starting at 0).
    """
    
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    #pat = re.compile(r"features_(?P<stock>[^_]+)_(?P<date>\d{4}-\d{2}-\d{2})_messages_.*\.parquet$")
    pat = re.compile(r"(?P<stock>[^_]+)_(?P<date>\d{4}-\d{2}-\d{2})_features.*\.parquet$")

    for p in tqdm(sorted(in_dir.glob("*.parquet")), desc="Processing parquets"):
        m = pat.match(p.name)
        if not m:
            continue

        stock = m.group("stock")
        date = m.group("date")

        df = pd.read_parquet(p).reset_index(drop=True)

        
        past = past_16min_indices_from_df(df)
        nxt = next_1min_index_from_df(df)

        common_idx = past.index.intersection(nxt.index)
        if len(common_idx) == 0:
            continue

        past_cut = past.loc[common_idx]
        nxt_cut = nxt.loc[common_idx]
        df_cut = df.loc[common_idx]


        df_cut.to_parquet(out_dir / f"features_{stock}_{date}_cut.parquet")
        past_cut.to_parquet(out_dir / f"past16_{stock}_{date}_cut.parquet")
        nxt_cut.to_parquet(out_dir / f"next1_{stock}_{date}_cut.parquet")




    
input_dir = Path("/scratch/project_2012747/mars_data/order_model/train/final") # dir with all "features".parquet files
output_dir = Path("/scratch/project_2012747/mars_data/order_batch_model/train/intermediate") 



#pd_ = pd.read_parquet("../../data/features/features_AMZN_2025-12-11_messages_10.parquet")
#print(pd_)

#result = past_16min_indices("../../data/features/features_AAPL_2025-12-17_messages_10.parquet")
#print(result)

#next_mins = next_1min_index_from_df(pd_)
#print(next_mins)

create_min16_plus1_indices_from_feature_parquets(input_dir, output_dir)
