from __future__ import annotations

import torch
from torch import Tensor, nn


class _CrossAttnBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_m = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q: Tensor, mem: Tensor) -> Tensor:
        qn = self.norm_q(q)
        mn = self.norm_m(mem)
        out, _ = self.cross_attn(query=qn, key=mn, value=mn, need_weights=False)
        q = q + out
        q = q + self.ff(self.norm_ff(q))
        return q


class EnsembleModel(nn.Module):
    """
    Inputs:
      base_logits: (B, V) float
      batch_tokens: (B, 64) long
    Output:
      refined_logits: (B, V)
    """

    def __init__(
        self,
        order_vocab_size: int,
        *,
        batch_vocab_size: int = 8192,
        batch_tokens_len: int = 64,
        d_model: int = 128,  # 128, #256,192
        num_layers: int = 2,  # 2, #4,3
        num_heads: int = 4,  # 4, #6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.order_vocab_size = int(order_vocab_size)
        self.batch_vocab_size = int(batch_vocab_size)
        self.batch_tokens_len = int(batch_tokens_len)

        self.order_logits_proj = nn.Linear(self.order_vocab_size, d_model, bias=False)

        self.batch_tok_emb = nn.Embedding(self.batch_vocab_size, d_model)
        self.batch_pos_emb = nn.Parameter(torch.zeros(1, self.batch_tokens_len, d_model))
        nn.init.normal_(self.batch_pos_emb, mean=0.0, std=0.02)

        self.blocks = nn.ModuleList([_CrossAttnBlock(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(d_model)
        self.delta_logits = nn.Linear(d_model, self.order_vocab_size, bias=False)

        # Start near identity: refined_logits ~= base_logits
        nn.init.zeros_(self.delta_logits.weight)

    def forward(self, base_logits: Tensor, batch_tokens: Tensor) -> Tensor:
        if base_logits.dim() != 2 or base_logits.size(-1) != self.order_vocab_size:
            raise ValueError(f"base_logits must be (B, {self.order_vocab_size})")
        if batch_tokens.dim() != 2 or batch_tokens.size(1) != self.batch_tokens_len:
            raise ValueError(f"batch_tokens must be (B, {self.batch_tokens_len})")
        if batch_tokens.dtype != torch.long:
            raise TypeError("batch_tokens must be torch.long")

        q = self.order_logits_proj(base_logits).unsqueeze(1)  # (B, 1, D)
        mem = self.batch_tok_emb(batch_tokens) + self.batch_pos_emb  # (B, 64, D)

        for blk in self.blocks:
            q = blk(q, mem)

        delta = self.delta_logits(self.out_norm(q)).squeeze(1)  # (B, V)
        return base_logits + delta
