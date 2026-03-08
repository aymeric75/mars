import torch
from market_simulation.models.ensemble_model import EnsembleModel

from custom_code.training.train_Order_Batch_Model_lightning import OrderBatchLightningModule





# Load the Order Model (see how it is called in custom_code/preprocessing/ensemble_preprocessing.py)


# Load the Order Batch Model
def load_order_batch_model(ckpt_path: str, device: str = "cpu"):
    lm = OrderBatchLightningModule.load_from_checkpoint(
        ckpt_path,
        map_location=device,
        strict=True,
    )
    model = lm.model.to(device)
    model.eval()
    return model

ob_model = load_order_batch_model(
    "/scratch/project_2012747/mars_runs/order_batch_model/31737330/val=val_loss=1.6047.ckpt",
    device="cpu"
)




# Load the Ensemble Model

def load_ensemble_model(ckpt_path: str, order_vocab_size: int, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)

    model = EnsembleModel(order_vocab_size=order_vocab_size)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    model.to(device)
    model.eval()
    return model


ensemble_model = load_ensemble_model(
    ckpt_path="/scratch/project_2012747/mars_runs/ensemble_model/31731449/ckpt_step=0_val=3.307867.pt",
    order_vocab_size=49152,
    device="cpu"
)
