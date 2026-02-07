# market_simulation/models/ensemble_model.py
# pyright: strict, reportUnknownMemberType=false, reportMissingTypeStubs=false
from __future__ import annotations

import math

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn


class _CrossAttnBlock(nn.Module):
    """One cross-attention + MLP block, batch-first.

    Query:   (B, 1, D)  (represents the order model prediction state)
    Memory:  (B, M, D)  (represents the order-batch tokens / channels)
    Output:  (B, 1, D)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_m = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q: Tensor, mem: Tensor, mem_key_padding_mask: Tensor | None = None) -> Tensor:
        # q: (B, 1, D), mem: (B, M, D)
        qn = self.norm_q(q)
        mn = self.norm_m(mem)

        attn_out, _ = self.cross_attn(
            query=qn,
            key=mn,
            value=mn,
            key_padding_mask=mem_key_padding_mask,  # True = mask
            need_weights=False,
        )
        q = q + attn_out
        q = q + self.ff(self.norm_ff(q))
        return q


class EnsembleModel(nn.Module, PyTorchModelHubMixin):
    """Ensemble (cross-attention) model.

    Purpose (per MarS paper):
      Refine the Order Model's next-order logits conditioned on the order-batch "channels"
      (represented here as the 64 VQ tokens for the next order image).

    Input:
      - base_logits: Float Tensor (B, V) from the frozen Order Model for the next order token.
      - batch_tokens: Long Tensor (B, 64) containing the conditioning order-image tokens.

    Output:
      - refined_logits: Float Tensor (B, V)
    """

    def __init__(
        self,
        # order side
        order_vocab_size: int,   # V
        # batch side
        batch_vocab_size: int = 8192,
        batch_tokens_len: int = 64,
        # model config
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.order_vocab_size = int(order_vocab_size)
        self.batch_vocab_size = int(batch_vocab_size)
        self.batch_tokens_len = int(batch_tokens_len)

        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)

        # Project order logits -> query embedding (B, V) -> (B, 1, D)
        self.order_logits_proj = nn.Linear(self.order_vocab_size, self.d_model, bias=False)

        # Embed batch tokens -> memory (B, 64, D)
        self.batch_tok_emb = nn.Embedding(self.batch_vocab_size, self.d_model)
        self.batch_pos_emb = nn.Parameter(torch.zeros(1, self.batch_tokens_len, self.d_model))
        nn.init.normal_(self.batch_pos_emb, mean=0.0, std=0.02)

        # Cross-attention blocks
        self.blocks = nn.ModuleList(
            [_CrossAttnBlock(self.d_model, self.num_heads, dropout) for _ in range(self.num_layers)]
        )

        # Map back to logits delta and add residual on base logits
        self.out_norm = nn.LayerNorm(self.d_model)
        self.delta_logits = nn.Linear(self.d_model, self.order_vocab_size, bias=False)

        # small init so we start near "do nothing"
        nn.init.zeros_(self.delta_logits.weight)

    def forward(self, base_logits: Tensor, batch_tokens: Tensor) -> Tensor:
        """
        base_logits:  (B, V) float
        batch_tokens: (B, 64) long
        """
        if base_logits.dim() != 2:
            raise ValueError(f"base_logits must be (B, V), got {tuple(base_logits.shape)}")
        if base_logits.size(-1) != self.order_vocab_size:
            raise ValueError(
                f"base_logits last dim must be V={self.order_vocab_size}, got {base_logits.size(-1)}"
            )
        if batch_tokens.dtype != torch.long:
            raise TypeError(f"batch_tokens must be torch.long, got {batch_tokens.dtype}")
        if batch_tokens.dim() != 2:
            raise ValueError(f"batch_tokens must be (B, 64), got {tuple(batch_tokens.shape)}")
        if batch_tokens.size(1) != self.batch_tokens_len:
            raise ValueError(
                f"batch_tokens must have length {self.batch_tokens_len}, got {batch_tokens.size(1)}"
            )

        # Build query from logits (single position)
        q = self.order_logits_proj(base_logits).unsqueeze(1)  # (B, 1, D)

        # Build memory from batch tokens
        mem = self.batch_tok_emb(batch_tokens) + self.batch_pos_emb  # (B, 64, D)

        # Cross-attend
        for blk in self.blocks:
            q = blk(q, mem)

        # Produce delta logits and refine
        delta = self.delta_logits(self.out_norm(q)).squeeze(1)  # (B, V)
        return base_logits + delta
