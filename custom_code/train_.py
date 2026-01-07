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


model = OrderModel(
    emb_dim = emb_dim,
    num_layers = num_layers,
    num_heads = num_heads,
    num_max_orders = K,
)



optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

log_every = 50  # set None to disable step logging

for epoch in range(10):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for step, (X_in, X_out) in enumerate(dl, start=1):
        X_in = X_in.to(device, non_blocking=True)
        X_out = X_out.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(X_in)                 # (B, T, vocab)  OR (B, vocab) if you changed the model
        # print(logits.shape) # torch.Size([256, 128, 49152])
        if logits.dim() == 3:
            logits = logits[:, -1, :]        # (B, vocab)
            # #print(logits_last.shape) # torch.Size([256, 49152])



        breakpoint()
        loss = criterion(logits, X_out)
        loss.backward()
        optimizer.step()

        # stats
        bsz = X_out.size(0)
        total_loss += loss.item() * bsz

        preds = logits.argmax(dim=-1)        # (B,)
        total_correct += (preds == X_out).sum().item()
        total_seen += bsz

        if log_every is not None and step % log_every == 0:
            avg_loss = total_loss / total_seen
            acc = total_correct / total_seen
            print(f"epoch {epoch} step {step}/{len(dl)}  loss {avg_loss:.4f}  acc {acc:.4f}")

    epoch_loss = total_loss / total_seen
    epoch_acc = total_correct / total_seen
    print(f"epoch {epoch}  loss {epoch_loss:.4f}  train_acc {epoch_acc:.4f}")