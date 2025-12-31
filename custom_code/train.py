import torch
import pandas as pd
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset, DataLoader
from market_simulation.models.order_model import OrderModel


features_df = pd.read_parquet("../data/mymessages.parquet")







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

model = OrderModel(
    emb_dim = emb_dim,
    num_layers = num_layers,
    num_heads = num_heads,
    num_max_orders = K,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for X_in, X_out in dl:
        logits = model(X_in.long())               # (B, vocab)
        loss = criterion(logits, X_out)    # X_out already long
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print("epoch", epoch, "loss", float(loss))
