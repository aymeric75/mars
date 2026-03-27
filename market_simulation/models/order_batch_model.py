# pyright: strict, reportUnknownMemberType=false, reportMissingTypeStubs=false
from __future__ import annotations

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn
from transformers import LlamaConfig, LlamaForCausalLM  # type: ignore


class OrderBatchModel(nn.Module, PyTorchModelHubMixin):
    """Autoregressive transformer over *order-batch* tokens.

    In the MarS paper, an order-batch (e.g., 1-minute of orders) is first converted into an
    "order image" (C=3, H=W=32) and then tokenized by a VQ-VAE/VQGAN-style image tokenizer
    into a small grid of discrete codes (e.g., 8x8=64 tokens with vocab size 8192).

    This module implements the Stage-2 model: a causal LM trained to predict the next code
    in the flattened sequence of these discrete order-image tokens.

    Expected usage
    --------------
    - Input: token ids shaped (B, T), where T = num_batches * tokens_per_batch.
      Example: 16 batches * 64 tokens = 1024 tokens.
    - Output: logits shaped (B, T, vocab_size).

    Note: This file intentionally does *not* implement the VQ tokenizer/decoder.
    """

    def __init__(
        self,
        # model config
        emb_dim: int,
        num_layers: int,
        num_heads: int,
        # tokenization config (Stage-1 output)
        vocab_size: int = 8192,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)

        llama_config = LlamaConfig(
            hidden_size=emb_dim,
            num_attention_heads=num_heads,
            intermediate_size=4 * emb_dim,
            num_hidden_layers=num_layers,
            attention_dropout=dropout,
            use_cache=False,
            vocab_size=self.vocab_size,
            tie_word_embeddings=True,
        )
        self.decoder = LlamaForCausalLM(llama_config)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        input_ids:
            Long tensor of shape (B, T) with values in [0, vocab_size).
        attention_mask:
            Optional int/bool mask of shape (B, T) (1/True = keep, 0/False = mask).

        Returns
        -------
        logits:
            Float tensor of shape (B, T, vocab_size).
        """
        if input_ids.dtype != torch.long:
            raise TypeError(f"input_ids must be torch.long, got {input_ids.dtype}")
        out = self.decoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)  # type: ignore
        return out.logits

    def forward_hidden_states(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Return the last hidden states for the input sequence."""
        if input_ids.dtype != torch.long:
            raise TypeError(f"input_ids must be torch.long, got {input_ids.dtype}")
        out = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )  # type: ignore
        hidden_states = out.hidden_states
        if hidden_states is None:
            raise RuntimeError("Decoder did not return hidden states")
        return hidden_states[-1]

    def encode(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Return one feature vector per sample, using the last token representation."""
        return self.forward_hidden_states(input_ids=input_ids, attention_mask=attention_mask)[:, -1, :]

    @torch.no_grad()
    def sample_next(self, input_ids: Tensor, temperature: float = 1.0, top_k: int | None = None) -> Tensor:
        """Sample the *next* token (one step) given a prefix.

        Returns a tensor of shape (B, 1) containing the sampled token ids.
        """
        logits = self(input_ids)[:, -1, :]  # (B, vocab)
        logits = logits / float(max(temperature, 1e-8))

        if top_k is not None and top_k > 0:
            k = min(int(top_k), logits.size(-1))
            v, _ = torch.topk(logits, k, dim=-1)
            cutoff = v[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < cutoff,
                torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype),
                logits,
            )

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, replacement=True)

    @torch.no_grad()
    def top_next(self, input_ids: Tensor) -> Tensor:
        """Greedy *next* token (argmax). Returns shape (B, 1)."""
        logits = self(input_ids)[:, -1, :]
        return torch.argmax(logits, dim=-1, keepdim=True)
