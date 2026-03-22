import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from custom_code.testing.utils import load_order_model
from custom_code.preprocessing.order_model.messages_to_features_no_engine import from_messages_to_features

DEVICE = "cuda"
SEQ_LEN = 1024
BATCH_SIZE = 15
ORDER_MODEL_CKPT = (
    "../../../mars_runs/order_model/tensorboard/bs=8_lr=1e-4/"
    "step=step=13920-val=val_loss=3.2903.ckpt"
)


@lru_cache(maxsize=1)
def get_order_model():
    return load_order_model(ckpt_path=ORDER_MODEL_CKPT, device=DEVICE).eval()


# a function that takes a feature file as input
# , put f0 in a list
# , call the Order Model, on the feature vector (in BATCHES)
#    take the output (do the argmax and so forth see the stats.py file)

def compute_value(tuple_of_paths):
    message_file, snapshot_file = tuple_of_paths

    feature_df = from_messages_to_features(message_file, snapshot_file)

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


    # print(feature_df)
    # exit()

    sub_df = feature_df.loc[:, "f0":"f14"]
    values = np.ascontiguousarray(sub_df.to_numpy(dtype=np.int64, copy=True))
    n_rows = len(values)

    if n_rows < SEQ_LEN:
        print(f"skipping {stock} {day}: only {n_rows} rows")
        return

    window_count = n_rows - SEQ_LEN + 1
    windows = np.lib.stride_tricks.sliding_window_view(values, SEQ_LEN, axis=0)
    windows = np.transpose(windows, (0, 2, 1))

    order_model = get_order_model()
    predicted_list = []

    with torch.inference_mode():
        for start in tqdm(range(0, window_count, BATCH_SIZE)):
            batch = np.ascontiguousarray(windows[start:start + BATCH_SIZE])
            x = torch.from_numpy(batch).to(device=DEVICE, dtype=torch.long, non_blocking=True)
            logits_next = order_model(x)[:, -1, :]
            pred_id = torch.argmax(logits_next, dim=1)
            predicted_list.extend(pred_id.cpu().tolist())


    predicted_dico = {}
    for i, ele in enumerate(predicted_list):
        predicted_dico[i] = ele

    gt_list = feature_df["f0"].tolist()[SEQ_LEN - 1:]
    gt_dico = {}
    for i, ele in enumerate(gt_list):
        gt_dico[i] = ele

    assert len(gt_dico) == len(predicted_dico)

    # predicted_dico = gt_dico

    json.dump(gt_dico, open(f"jsons/{stock}_{day}_order-indices-gt.json", "w"), default=lambda x: x.item())
    json.dump(predicted_dico, open(f"jsons/{stock}_{day}_order-indices-pred.json", "w"), default=lambda x: x.item())

    return










if __name__ == "__main__":

    #data_dir = Path("/scratch/project_2012747/mars_data/order_model/test/raw")
    data_dir = Path("../../../data/test")

    list_of_pairs = []

    for message_file in data_dir.glob("*_messages.parquet"):
        snap_path = Path(str(message_file).replace("_messages", "_snapshots"))
        if snap_path.exists():
            list_of_pairs.append((message_file, snap_path))


    list_of_pairs = [next(t for t in list_of_pairs if "AVGO" in str(t[0]))]


    #list_of_pairs = list_of_pairs[:2]

    with ProcessPoolExecutor(1) as ex:
       list(ex.map(compute_value, list_of_pairs))

    #compute_value(Path("NFLX_2025-12-09_messages.parquet"), Path("NFLX_2025-12-09_snapshots.parquet"))
