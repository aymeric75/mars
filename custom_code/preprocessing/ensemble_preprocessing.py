# ensemble_preprocessing_parquet_topk_plus_target.py
import os, sys, glob
import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F

base = Path(__file__).resolve().parents[2]
sys.path.append(str(base))

from custom_code.training.train_Order_Model_lightning import OrderLightningModule

CKPT_PATH = "../training/checkpoints_order_model/step=step=1920-val=val_loss=2.7747.ckpt"
PARQ_DIR  = "../../data/ensemble_model/train/features"
OUT_DIR   = "/scratch/project_2012747/mars_data/output_order_model_for_ensemble_topk"

SEQ_LEN = 1024
BATCH   = 64
TOPK    = 64
NUM_WORKERS = 4

FEATURE_COLS = [f"f{i}" for i in range(15)]  # f0..f14
TARGET_COL   = "f0"                          # next-order ground truth

os.makedirs(OUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = OrderLightningModule.load_from_checkpoint(CKPT_PATH).to(device).eval()





class ParquetWindowWithTarget(Dataset):
    def __init__(self, parquet_path: str, seq_len: int):
        df = pd.read_parquet(parquet_path, columns=FEATURE_COLS)
        arr = df.to_numpy(dtype=np.int32, copy=False)       # (N, 15)
        self.x = torch.from_numpy(arr)                      # CPU
        self.f0 = arr[:, 0]                                 # numpy view
        self.T = seq_len
        self.n = arr.shape[0] - seq_len                     # targets need i+T
        if self.n <= 0:
            raise ValueError(f"File too short for seq_len={seq_len}: {parquet_path}")

    def __len__(self): return self.n
    def __getitem__(self, i):
        return self.x[i:i+self.T], int(self.f0[i+self.T])   # (T,15), target

parq_files = sorted(glob.glob(os.path.join(PARQ_DIR, "*.parquet")))


for pq_path in parq_files:
    
    
    if "features_AMZN_2025-12-09_cut_tenth.parquet" not in pq_path:
        continue
    
    ds = ParquetWindowWithTarget(pq_path, SEQ_LEN)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)
    
    val_loss = 0.0
    n_tot = 0
    
    with torch.inference_mode():
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
             
            logits = model(x)[:, -1, :]          # (B, V)
            loss = F.cross_entropy(logits, y, reduction="sum")
            
            val_loss += loss.item()
            n_tot += y.numel()
    
    print(f"[OrderModel] val_loss = {val_loss / n_tot:.6f}")
    
    
    
    continue
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    import zarr
    from numcodecs import Blosc
    
    V = model.hparams.vocab_size if hasattr(model, "hparams") and hasattr(model.hparams, "vocab_size") else 49152
    
    # Compressor: zstd + bitshuffle is usually great for float16 tensors
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    
    stem = Path(pq_path).stem
    
    logits_stem  = stem.replace("feature", "logits")
    targets_stem = stem.replace("feature", "order_idx")
    
    
    out_logits  = os.path.join(OUT_DIR, f"{logits_stem}_dense_f16.zarr")
    out_targets = os.path.join(OUT_DIR, f"{targets_stem}_targets.zarr")
    
    # directory stores (not .zip) = faster + simpler
    r1 = zarr.open_group(out_logits, mode="w")
    r2 = zarr.open_group(out_targets, mode="w")
    
    log_arr = r1.create_dataset(
        "logits", shape=(len(ds), V),
        chunks=(BATCH, V),
        dtype="f2",  # float16
        compressor=compressor,
    )
    
    tgt_arr = r2.create_dataset(
        "target_f0", shape=(len(ds),),
        chunks=(max(BATCH, 1024),),
        dtype="i4",
        compressor=compressor,
    )
    
    offset = 0
    with torch.inference_mode():
        for x, y in tqdm(dl, desc=Path(pq_path).name):
            x = x.to(device, non_blocking=True)           # (B,T,15)
            logits = model(x)[:, -1, :]                   # (B,V) float32/bf16 on GPU
            logits = logits.to(torch.float16)             # store as float16 to save space
    
            n = logits.shape[0]
            log_arr[offset:offset+n] = logits.cpu().numpy()
            tgt_arr[offset:offset+n] = y.numpy().astype(np.int32)
            offset += n
    
    print("saved:", out_logits)
    print("saved:", out_targets)







    
    
    
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
