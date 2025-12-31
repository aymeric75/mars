import torch
import pandas as pd
import torch.nn as nn

from market_simulation.models.order_model import OrderModel


features_df = pd.read_parquet("../data/mymessages.parquet")







K = 200  # must match num_max_orders

X_seq = []
y_seq = []

for t in range(K, len(features_df)):
    X_seq.append(
        features_df[[f"f{i}" for i in range(15)]].iloc[t-K:t].values
    )
    y_seq.append(
        features_df["f0"].iloc[t]
    )

X_in  = torch.tensor(X_seq, dtype=torch.float32)   # (B, K, 15)
X_out = torch.tensor(y_seq, dtype=torch.long)      # (B,)







# print(X_in.shape)
# exit()
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
    logits = model(X_in)
    loss = criterion(logits, X_out.long())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"epoch {epoch}, loss {loss.item():.4f}")
