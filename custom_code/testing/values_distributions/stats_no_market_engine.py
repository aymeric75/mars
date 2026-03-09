import torch
import pandas as pd
import numpy as np
from custom_code.testing.utils import load_order_model, load_ensemble_model, load_order_batch_model

# a function that takes a feature file as input
# , put f0 in a list
# , call the Order Model, on the feature vector (in BATCHES)
#    take the output (do the argmax and so forth see the stats.py file)

def compute_value(feature_file):

    df = pd.read_parquet(feature_file)

    predicted_list = []
    gt_list = df["f0"].tolist()


    device="cuda"
    # Load the Order Model
    order_model = load_order_model(
        ckpt_path="../step=step=3360-val=val_loss=3.7445.ckpt",
        device=device
    )

    order_model = order_model.to(device).eval()


    sub_df = df.loc[:, 'f0':'f14']

    N = len(sub_df)
    seq_len = 1024
    batch_size = 15

    import time
    prev = time.perf_counter()

    for start in range(0, N - seq_len + 1, batch_size):


        now = time.perf_counter()
        elapsed = now - prev
        print("elapsed:", elapsed)
        prev = now

        print(start)
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


    print(len(predicted_list))
    print(len(gt_list))
    return

compute_value("../data/NFLX_2025-12-09_features.parquet")
