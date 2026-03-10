import torch
import json
import pandas as pd
import numpy as np

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor



from tqdm import tqdm
from pathlib import Path

from custom_code.testing.utils import load_order_model, load_ensemble_model, load_order_batch_model

from custom_code.preprocessing.order_model.messages_to_features_no_engine import from_messages_to_features


# a function that takes a feature file as input
# , put f0 in a list
# , call the Order Model, on the feature vector (in BATCHES)
#    take the output (do the argmax and so forth see the stats.py file)

def compute_value(message_file, snapshot_file):



    feature_df = from_messages_to_features(message_file, snapshot_file)
    feature_df = feature_df[(feature_df["Time"] >= 34200000226319) & (feature_df["Time"] <= 57599998528372)]

    stock = message_file.stem.split("_")[0]
    day = message_file.stem.split("_")[1]

    print(f"processing: {stock} and {day}")



    # RENAME COLS TO F0 ... F14
    cols = feature_df.columns.tolist()
    start = cols.index("f0")
    for i in range(start + 1, len(cols)):
        cols[i] = f"f{i - start}"
    feature_df.columns = cols

    # CAST col from f4 to f14 as integer
    feature_df[[f"f{i}" for i in range(4, 15)]] = feature_df[[f"f{i}" for i in range(4, 15)]].astype(int)


    predicted_list = []
    gt_list = feature_df["f0"].tolist()



    gt_dico = {}
    for i, val in enumerate(gt_list):
        gt_dico[i] = val

    device="cuda"
    # Load the Order Model
    order_model = load_order_model(
        ckpt_path="step=step=3360-val=val_loss=3.7445.ckpt",
        device=device
    )

    order_model = order_model.to(device).eval()

    sub_df = feature_df.loc[:, 'f0':'f14']

    N = len(sub_df)
    seq_len = 1024
    batch_size = 15

    import time
    prev = time.perf_counter()

    #for start in range(0, N - seq_len + 1, batch_size):
    for start in tqdm(range(0, N - seq_len + 1, batch_size)):

        # now = time.perf_counter()
        # elapsed = now - prev
        # print("elapsed:", elapsed)
        # prev = now

        #print(start)
        # if start % 15000 == 0:
        #     print(start)

        batch = []
        for b in range(batch_size):
            i = start + b
            if i + seq_len > N:
                break
            batch.append(sub_df[i:i + seq_len])

        batch = np.stack(batch)   # shape (B, 1024, 15)
        X = torch.from_numpy(batch).to(device=device, dtype=torch.long)
        base_logits = order_model(X)
        # print(base_logits)
        # print(base_logits.shape)

        logits_next = base_logits[:, -1, :]          # (49152,)
        probs_next  = torch.softmax(logits_next, 1) # (49152,)
        pred_id = torch.argmax(probs_next, dim=1) #.item()

        predicted_list.extend(pred_id.tolist())

    print("len(predicted_list)")
    print(len(predicted_list))
    print(len(gt_list))
    
    predicted_dico = {}
    for i, ele in enumerate(predicted_list):
        predicted_dico[i] = ele
    gt_dico = {}
    for i, ele in enumerate(gt_list):
        gt_dico[i] = ele
    json.dump(gt_dico, open(f"jsons/{stock}_{day}_order-indices-gt.json", "w"), default=lambda x: x.item())
    json.dump(predicted_dico, open(f"jsons/{stock}_{day}_order-indices-pred.json", "w"), default=lambda x: x.item())

    
    return










if __name__ == "__main__":
    
    #data_dir = Path("/scratch/project_2012747/mars_data/order_model/test/final")

    #files = list(data_dir.glob("*_features.parquet"))

    # '/scratch/project_2012747/mars_data/order_model/test/raw/AMZN_2025-12-09_messages.parquet
    



    """
    # Filtering the list
    tickers = {"NFLX", "NVDA", "TSLA"}
    start = datetime.fromisoformat("2025-12-09")
    end = datetime.fromisoformat("2025-12-11")

    filtered = []
    for f in files:
        ticker, date_str, *_ = f.stem.split("_")
        date = datetime.fromisoformat(date_str)

        if ticker in tickers and start <= date <= end:
            filtered.append(f)
    """

    #with ProcessPoolExecutor(1) as ex:
    #    list(ex.map(compute_value, files))

    compute_value(Path("NFLX_2025-12-09_messages.parquet"), Path("NFLX_2025-12-09_snapshots.parquet"))

