from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np


from utils import *


NS_PER_MIN = 60 * 1_000_000_000
H, W, C = 32, 32, 3
V_MAX = 100




class OrderImageDataset(Dataset):
    def __init__(self, ds, time_col="Time", f0_col="f0", include_current=False):
        self.include_current = include_current


        ds = ds.with_format("numpy", columns=[time_col, f0_col])

        self.t  = np.asarray(ds[time_col], dtype=np.int64)
        self.f0 = np.asarray(ds[f0_col],   dtype=np.int64)

        # seg0[i] = start of the (stock/day) segment containing i
        s = np.empty(len(self.t), np.int64)
        s[0] = 0
        for i in range(1, len(self.t)):
            s[i] = i if self.t[i] <= self.t[i-1] else s[i-1]
        self.seg0 = s

        # precompute for each i: j0[i] = first index in [seg0[i], i) with t >= t[i]-60s
        self.j0 = np.empty(len(self.t), np.int64)
        self.valid = []
        for i in range(len(self.t)):
            a = int(self.seg0[i])
            if self.t[i] - self.t[a] < NS_PER_MIN:
                self.j0[i] = a # HARMLESS PLACEHOLDER (this value will never be used)
                continue
            self.j0[i] = a + np.searchsorted(self.t[a:i], self.t[i] - NS_PER_MIN, side="left")
            self.valid.append(i)
        self.valid = np.asarray(self.valid, dtype=np.int64)

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, k):
        i = int(self.valid[k])
        j0 = int(self.j0[i])
        j1 = i + 1 if self.include_current else i

        arr3 = decode_order_index_vec(self.f0[j0:j1])
        img  = build_order_image(arr3)
        x    = img.transpose(2, 0, 1).astype(np.float32)
        x    = (x / 100.0) * 2.0 - 1.0
        return torch.from_numpy(x)




# class OrderImageDataset(Dataset):
#     """
#     PyTorch Dataset wrapper around a HuggingFace datasets.Dataset.

#     Requirements:
#       - `ds` is a HF Dataset (has .with_format and column_names)
#       - columns: time_col (int64 ns), f0_col (int64 order index)
#       - data is globally sorted by Time, but can contain "segments" where Time restarts/decreases
#     """
#     def __init__(self, ds, time_col="Time", f0_col="f0", include_current=False):
#         # enforce HF dataset only
#         if not (hasattr(ds, "with_format") and hasattr(ds, "column_names")):
#             raise TypeError("OrderImageDataset expects a HuggingFace datasets.Dataset")

#         self.include_current = include_current

#         print("here0")

#         # Make column access fast and consistent
#         ds = ds.with_format("numpy", columns=[time_col, f0_col])

#         print("here000")

#         # Pull into contiguous numpy arrays for fast slicing/searchsorted
#         self.t = np.asarray(ds[time_col], dtype=np.int64)
#         self.f = np.asarray(ds[f0_col],   dtype=np.int64)

#         print("here1")

#         # segment start per row: new segment when Time doesn't strictly increase
#         s = np.empty(len(self.t), np.int64)
#         s[0] = 0
#         for i in range(1, len(self.t)):
#             s[i] = i if self.t[i] <= self.t[i - 1] else s[i - 1]
#         self.seg0 = s

#         print("here2")

#         # valid rows: need >= 60s history within each segment
#         seg_starts = np.r_[0, 1 + np.flatnonzero(self.t[1:] <= self.t[:-1])]
#         seg_ends   = np.r_[seg_starts[1:], len(self.t)]
#         vids = []
#         for a, b in zip(seg_starts, seg_ends):
#             i0 = a + np.searchsorted(self.t[a:b], self.t[a] + NS_PER_MIN, side="left")
#             if i0 < b:
#                 vids.append(np.arange(i0, b, dtype=np.int64))


#         print("here3")
#         self.valid = np.concatenate(vids) if vids else np.empty((0,), np.int64)

#     def __len__(self):
#         return int(self.valid.size)

#     def __getitem__(self, k):
#         i  = int(self.valid[k])
#         a  = int(self.seg0[i])
#         ti = self.t[i]

#         j0 = a + np.searchsorted(self.t[a:i], ti - NS_PER_MIN, side="left")
#         j1 = i + 1 if self.include_current else i

#         arr3 = decode_order_index_vec(self.f[j0:j1])
#         img  = build_order_image(arr3)                       # HWC uint8
#         x    = img.transpose(2, 0, 1).astype(np.float32)     # CHW
#         x    = (x / 100.0) * 2.0 - 1.0                       # -> [-1,1] for VQ
#         return torch.from_numpy(x)






def make_vq_collate(vq_for_orders, device="cuda", return_x=False):
    """
    Returns a collate_fn that:
      - stacks images
      - runs vq_for_orders.m.encode to get codebook indices
    """
    vq_for_orders.eval().to(device)

    @torch.no_grad()
    def collate(batch):
        x = torch.stack(batch, dim=0).to(device, non_blocking=True)  # (B,3,32,32)

        # encode: q, _, info  (info usually contains indices)
        q, _, info = vq_for_orders.m.encode(x)  # vq_for_orders.m is the actual VQ model :contentReference[oaicite:1]{index=1}

        # try to extract indices robustly
        if isinstance(info, (tuple, list)) and len(info) >= 1:
            indices = info[-1]
        elif isinstance(info, dict) and ("indices" in info):
            indices = info["indices"]
        else:
            raise RuntimeError(f"Can't find indices in encode() info: {type(info)}")

        indices = indices.reshape(indices.shape[0], -1).long()  # (B, 64) typically

        return (indices, x) if return_x else indices

    return collate




def load_parquets_hf(folder: str | Path, pattern: str = "*.parquet"):
    folder = Path(folder)
    paths = sorted(folder.glob(pattern))
    if not paths:
        return None

    # Load as Arrow dataset (memory-mapped when possible, much nicer than big pandas concat)
    ds = load_dataset(
        "parquet",
        data_files=[str(p) for p in paths],
        split="train",
    )

    #ds = ds.sort("Time")

    return ds


print(len(t))
print(len(f0))

# from features files to mmap (only filtering what's useful),


########## nan mais sérieux, bordel de merde, METS TOUT EN putain de



# one_big_ds = load_parquets_hf("../data/features")



# ds = OrderImageDataset(one_big_ds)  # df must be sorted by Time


# collate_fn = make_vq_collate(vq_for_orders=your_VQForOrders_instance, device="cuda", return_x=False)
# dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)

# for codes in dl:
#     # codes: (B, 64) discrete token indices (typically)
#     pass


######### I am going to give you the code to do a training:
#     1) separate into train/eval/test
#     2)
