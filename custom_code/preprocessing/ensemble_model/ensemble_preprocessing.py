""" """
# ensemble_preprocessing_parquet_topk_plus_target.py
import os, sys, glob
import numpy as np
import pandas as pd
import torch
import zarr
import re
import torch.nn.functional as F


from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from pathlib import Path
from tqdm import tqdm
from numcodecs import Blosc


print("ICI000")

base = Path(__file__).resolve().parents[2]
sys.path.append(str(base))

from custom_code.training.order_model.train_Order_Model_hypersearch import OrderLightningModule
print("ICI1111")
CKPT_PATH = "step=step=3360-val=val_loss=3.7445.ckpt"
PARQ_FEATURES_DIR  = "/scratch/project_2012747/mars_data/order_batch_model/val/intermediate"
NEXT1_DIR = "/scratch/project_2012747/mars_data/order_batch_model/val/final"
OUT_DIR   = "/scratch/project_2012747/mars_data/ensemble_model/val/final"


SEQ_LEN = 1024
BATCH   = 64
TOPK    = 64
NUM_WORKERS = 1

FEATURE_COLS = [f"f{i}" for i in range(15)]  # f0..f14
TARGET_COL   = "f0"                          # next-order ground truth

os.makedirs(OUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = OrderLightningModule.load_from_checkpoint(CKPT_PATH).to(device).eval()
print("ICI2222")

class ParquetWindowWithTarget(Dataset):
    def __init__(self, parquet_path: str, seq_len: int, next1_dir: str):
        df = pd.read_parquet(parquet_path, columns=FEATURE_COLS)
        arr = df.to_numpy(dtype=np.int32, copy=False)   # (N, 15)

        self.x = torch.from_numpy(arr)
        self.f0 = arr[:, 0]
        self.T = seq_len
        self.n = arr.shape[0] - seq_len

        stem = Path(parquet_path).stem                     # e.g. AMD_2025-11-03_features-cut.parquet
        
        stock = stem.split("_")[0]
        date = stem.split("_")[1]


        next1_path = Path(next1_dir) / f"{stock}_{date}_next1-tokens.zarr"
        self.next1 = zarr.open(next1_path, mode="r")
        
        # check lengths
        if arr.shape[0] != self.next1.shape[0]:
            raise ValueError(
                f"Length mismatch:\n"
                f"{parquet_path} -> {arr.shape[0]}\n"
                f"{next1_path} -> {self.next1.shape[0]}"
            )


    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return (
            self.x[i:i+self.T],                           # (T, 15)
            int(self.f0[i+self.T]),                      # target int
            torch.from_numpy(self.next1[i+self.T]),      # (64,)
        )
        
        
parq_files = sorted(glob.glob(os.path.join(PARQ_FEATURES_DIR, "*_features*.parquet")))


for iii, pq_path in enumerate(parq_files):
    
    
    
    print(pq_path)
    
    #if "features_AMZN_2025-12-09_cut_tenth.parquet" not in pq_path:
    #    continue
    
    ds = ParquetWindowWithTarget(pq_path, SEQ_LEN, NEXT1_DIR)
    

    
    n = len(ds)
    n_sample = max(1, int(0.01 * n))
    indices = np.random.choice(n, n_sample, replace=False)
    n_out = len(indices)
    
    sampler = SubsetRandomSampler(indices)
    
    dl = DataLoader(ds, batch_size=BATCH, sampler=sampler,
                    num_workers=NUM_WORKERS, pin_memory=True)
    
    val_loss = 0.0
    n_tot = 0
    
    
    #### Used to check the Order Model val loss on the current file
    """
    with torch.inference_mode():
        
        pbar = tqdm(dl, desc="Validation", total=len(dl))
        for x, y in pbar:
            print("on iteration")
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
             
            logits = model(x)[:, -1, :]          # (B, V)
            loss = F.cross_entropy(logits, y, reduction="sum")
            
            val_loss += loss.item()
            n_tot += y.numel()
            pbar.set_postfix({"avg_loss": val_loss / max(n_tot,1)})
            
    print(f"[OrderModel] val_loss = {val_loss / n_tot:.6f}")
    """
    


    
    V = model.hparams.vocab_size if hasattr(model, "hparams") and hasattr(model.hparams, "vocab_size") else 49152
    
    # Compressor: zstd + bitshuffle is usually great for float16 tensors
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    
    stem = Path(pq_path).stem
    
    
    stock_date = stem.split('_')[0]+"_"+stem.split('_')[1]
    
    out_logits  = os.path.join(OUT_DIR, f"{stock_date}_dense-f16.zarr")
    out_targets = os.path.join(OUT_DIR, f"{stock_date}_targets.zarr")
    out_next1 = os.path.join(OUT_DIR, f"{stock_date}_tokens.zarr")
    
    # directory stores (not .zip) = faster + simpler
    r1 = zarr.open_group(out_logits, mode="w")
    r2 = zarr.open_group(out_targets, mode="w")
    r3 = zarr.open_group(out_next1, mode="w")
    
    
    
    log_arr = r1.create_dataset(
        "logits", shape=(n_out, V),
        chunks=(min(BATCH, n_out), V),
        dtype="f2",
        compressor=compressor,
    )
    
    tgt_arr = r2.create_dataset(
        "target_f0", shape=(n_out,),
        chunks=(max(min(BATCH, n_out), 1),),
        dtype="i4",
        compressor=compressor,
    )
    
    next1_arr = r3.create_dataset(
        "next1_tokens", shape=(n_out, 64),
        chunks=(min(max(BATCH, 1), n_out), 64),
        dtype="i2",
        compressor=compressor,
    )


    max_batches = int(0.1 * len(dl))
    
    offset = 0
    with torch.inference_mode():
        for i, (x, y, z) in enumerate(tqdm(dl, desc=Path(pq_path).name)):
            x = x.to(device, non_blocking=True)
            logits = model(x)[:, -1, :]
            logits = logits.to(torch.float16)
    
            n = logits.shape[0]
            log_arr[offset:offset+n] = logits.cpu().numpy()
            tgt_arr[offset:offset+n] = y.numpy().astype(np.int32)
            next1_arr[offset:offset+n] = z.numpy().astype(np.int16)
            offset += n

         
    print("saved:", out_logits)
    print("saved:", out_targets)
    print("saved:", out_next1)
    
    """
    stem = Path(pq_path).stem
    
    logits_stem  = stem.replace("feature", "logits")
    targets_stem = stem.replace("feature", "order_idx")
    
    out_logits  = os.path.join(OUT_DIR, f"{logits_stem}_top{TOPK}.zarr.zip")
    out_targets = os.path.join(OUT_DIR, f"{targets_stem}_targets.zarr.zip")


    # zarr outputs
    s1 = zarr.ZipStore(out_logits, mode="w");  r1 = zarr.group(store=s1)
    s2 = zarr.ZipStore(out_targets, mode="w"); r2 = zarr.group(store=s2)

    idx_arr = r1.create_dataset("topk_idx",   shape=(len(ds), TOPK), chunks=(BATCH, TOPK), dtype="i4")
    val_arr = r1.create_dataset("topk_logit", shape=(len(ds), TOPK), chunks=(BATCH, TOPK), dtype="f2")
    tgt_arr = r2.create_dataset("target_f0",  shape=(len(ds),),      chunks=(max(BATCH, 1024),), dtype="i4")

    offset = 0
    with torch.inference_mode():
        for x, y in tqdm(dl, desc=Path(pq_path).name):
            x = x.to(device, non_blocking=True)                 # (B,T,15)
            logits = model(x)[:, -1, :]                         # (B,V)
            topv, topi = torch.topk(logits, TOPK, dim=-1)

            n = topi.shape[0]
            idx_arr[offset:offset+n] = topi.to(torch.int32).cpu().numpy()
            val_arr[offset:offset+n] = topv.to(torch.float16).cpu().numpy()
            tgt_arr[offset:offset+n] = y.numpy().astype(np.int32)
            offset += n

    s1.close(); s2.close()
    print("saved:", out_logits)
    print("saved:", out_targets)
    """
