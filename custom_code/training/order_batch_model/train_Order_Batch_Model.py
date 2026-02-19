from market_simulation.models.order_batch_model import OrderBatchModel

from market_simulation.models.utils_order_model import unzip_zarr_zips
from market_simulation.models.utils_order_batch_model import (
    MultiDirZarrTokenDataset, lm_loss_next_token
)
from torch.utils.data import DataLoader

zarr_dirs = unzip_zarr_zips("../../data/order_batch_model", "*tokens_v2.zarr.zip")
# 1) dataset: each file holds tokens shaped (T,) or (N,T)
ds = MultiDirZarrTokenDataset(zarr_dirs, seq_len=16, array_path="")
print(type(ds))
print(type(ds[0]))
print(ds[0].shape)

dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

# 2) model
model = OrderBatchModel(emb_dim=768, num_layers=12, num_heads=12, vocab_size=8192)

# 3) train step
for input_ids in dl:
    logits = model(input_ids)                 # (B,T,V)
    loss = lm_loss_next_token(logits, input_ids)
    loss.backward()
    break
