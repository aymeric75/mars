import torch

def adamw_lower_bound_bytes(model: torch.nn.Module) -> int:
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bytes_per_fp32 = 4
    copies = 4  # weights + grads + exp_avg + exp_avg_sq
    return n_params * copies * bytes_per_fp32


def activation_estimate_bytes(B, S, D, L, bytes_per_act=2, c=8):
    # c ~ 6..12; larger = more pessimistic
    return int(B * S * D * L * c * bytes_per_act)


def measure_peak_memory(model, batch, loss_fn, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model.train()

    batch = batch.to(device, non_blocking=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    opt.zero_grad(set_to_none=True)

    with torch.amp.autocast(
        "cuda",
        enabled=(device.type == "cuda"),
        dtype=torch.bfloat16,
    ):
        logits = model(batch)
        loss = loss_fn(logits, batch)

    loss.backward()
    opt.step()

    peak = torch.cuda.max_memory_allocated(device)
    return peak




def build_model_from_variant(model_variant: str):
    """
    base ~ your current config (emb=64, layers=2, heads=4)
    small = fewer params (emb=48, layers=1, heads=4) -> significantly smaller
    """
    if model_variant == "base":
        EMB_DIM, NUM_LAYERS, NUM_HEADS = 64, 2, 4
    elif model_variant == "small":
        # smaller than base; keep heads dividing emb_dim nicely
        EMB_DIM, NUM_LAYERS, NUM_HEADS = 48, 1, 4
    else:
        raise ValueError(f"Unknown model_variant={model_variant}")

    model = OrderModel(
        emb_dim=EMB_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_max_orders=K,
    ).to(device)

    return model, {"emb_dim": EMB_DIM, "num_layers": NUM_LAYERS, "num_heads": NUM_HEADS}
