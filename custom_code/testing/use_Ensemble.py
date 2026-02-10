import torch
from market_simulation.models.ensemble_model import EnsembleModel

def load_ensemble_model(ckpt_path: str, order_vocab_size: int, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)

    model = EnsembleModel(order_vocab_size=order_vocab_size)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    model.to(device)
    model.eval()
    return model


model = load_ensemble_model(
    ckpt_path="/scratch/project_2012747/mars_runs/ensemble_model/31731449/ckpt_step=0_val=3.307867.pt",
    order_vocab_size=49152,
    device="cuda"
)
