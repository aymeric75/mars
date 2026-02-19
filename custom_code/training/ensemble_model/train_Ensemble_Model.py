import torch
from market_simulation.models.ensemble_model import EnsembleModel
from market_simulation.models.utils_ensemble_model import (
    make_ensemble_train_loader,
    ensemble_training_step,
)

V = 49152
device = "cuda" if torch.cuda.is_available() else "cpu"

loader = make_ensemble_train_loader(
    next64_path="../../data/ensemble_model/train/64tokens/next1_tokens_AAPL_2025-12-17_first228527.zarr",
    topk_idx_path="../../data/ensemble_model/train/logits/topk_idx",
    topk_logit_path="../../data/ensemble_model/train/logits/topk_logit",
    targets_path="../../data/ensemble_model/train/order_indices_pred_gt/target_f0",
    batch_size=64,
    num_workers=0,
)

model = EnsembleModel(order_vocab_size=V).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

batch = next(iter(loader))
batch = {k: v.to(device) for k, v in batch.items()}

model.train()
opt.zero_grad(set_to_none=True)
loss = ensemble_training_step(model=model, batch=batch, vocab_size=V)
loss.backward()
opt.step()

print("loss:", float(loss))
