import torch
import pandas as pd
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset, DataLoader
from market_simulation.models.order_model import OrderModel


features_df = pd.read_parquet("../data/mymessages.parquet")



t_sec = (features_df["Time"].to_numpy() // 1_000_000_000).astype("int64")  # ns -> seconds
features_df["f4"] = t_sec - 34200  # seconds since 09:30
features_df["f4"] = features_df["f4"].clip(0, 23399)  # 6.5h = 23400s



K = 128

F = features_df[[f"f{i}" for i in range(15)]].to_numpy(dtype=np.float32)  # (T,15)
Y = features_df["f0"].to_numpy(dtype=np.int64)                            # (T,)

class OrderClipDataset(Dataset):
    def __init__(self, F, Y, K):
        self.F, self.Y, self.K = F, Y, K
    def __len__(self):
        return len(self.Y) - self.K
    def __getitem__(self, idx):
        t = idx + self.K
        x = torch.from_numpy(self.F[t-self.K:t]).long()    # (K,15)
        y = torch.tensor(self.Y[t], dtype=torch.long)
        return x, y

ds = OrderClipDataset(F, Y, K)
dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)


emb_dim    = 256
num_layers = 6
num_heads  = 8

print("'LLLAa1111")

model = OrderModel(
    emb_dim = emb_dim,
    num_layers = num_layers,
    num_heads = num_heads,
    num_max_orders = K,
)


print("'LLLAa22222")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

print("'LLLAa33333")

for epoch in range(10):
    print("'LLLAa44444")
    for X_in, X_out in dl:
        print("'LLLAa55555")
        logits = model(X_in.long())               # (B, vocab)
        print("'LLLAa66666666")
        loss = criterion(logits, X_out)    # X_out already long
        print("'LLLAa77777777")
        loss.backward()
        print("'LLLAa888888")
        optimizer.step()
        print("'LLLAa99999999999")
        optimizer.zero_grad()
        print("'LLLAa11110000000000000000")
    print("epoch", epoch, "loss", float(loss))
