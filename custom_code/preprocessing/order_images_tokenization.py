# --- Example where we create a random "order image" and encode -> tokens -> decode ---
import numpy as np

# Random "order image" in [0, 100] like MarS mentions pixel V in [0,100].
# We'll normalize to [-1, 1] because LDM expects that.
x_np = np.random.randint(0, 101, size=(3, 32, 32), dtype=np.uint8)

# Normalize to [-1, 1]
x = torch.from_numpy(x_np).float() / 100.0          # [0,1]
x = x * 2.0 - 1.0                                   # [-1,1]
x = x.unsqueeze(0).to(device)                       # [B=1, C=3, H=32, W=32]

with torch.no_grad():
    # VQModel.encode returns: quant, emb_loss, info
    # info usually: (perplexity, min_encodings, min_encoding_indices)
    quant, emb_loss, info = model.encode(x)  # THIS IS THE VQGAN model that was trained previously

    # Token indices (shape [B*H'*W'] or [B, H', W'] depending on impl)
    # In CompVis VQ, info[2] is "min_encoding_indices"
    indices = info[2]

    # Decode back (reconstruction from quantized latents)
    x_rec = model.decode(quant)

print("Input shape:", x.shape)
print("Quant shape:", quant.shape)     # expected [1, 3, 8, 8] for f=4 and d=3
print("Token indices shape:", indices.shape)
print("Reconstruction shape:", x_rec.shape)

# For convenience: reshape tokens to 8x8 if needed
# (Often indices are flat; if so, reshape)
if indices.ndim == 2 and indices.shape[0] == 1:
    # sometimes [B, H'*W']
    H_ = W_ = int(indices.shape[1] ** 0.5)
    tokens_8x8 = indices.view(1, H_, W_)
elif indices.ndim == 1:
    H_ = W_ = int(indices.numel() ** 0.5)
    tokens_8x8 = indices.view(1, H_, W_)
else:
    tokens_8x8 = indices

print("Tokens (8x8) min/max:", int(tokens_8x8.min()), int(tokens_8x8.max()))
