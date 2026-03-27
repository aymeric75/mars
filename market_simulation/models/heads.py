from __future__ import annotations

import torch
from torch import Tensor, nn


def _make_mlp(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class FutureReturnRegressionHead(nn.Module):
    """Predict the future return from one MarS feature vector."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.trunk = _make_mlp(int(input_dim), int(hidden_dim), float(dropout))
        self.out = nn.Linear(int(hidden_dim), 1)

    def forward(self, features: Tensor) -> Tensor:
        if features.dim() != 2:
            raise ValueError("features must be a 2D tensor of shape (B, D)")
        return self.out(self.trunk(features)).squeeze(-1)


class FuturePnLRegressionHead(nn.Module):
    """Predict the future round-trip PnL from one MarS feature vector."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.trunk = _make_mlp(int(input_dim), int(hidden_dim), float(dropout))
        self.out = nn.Linear(int(hidden_dim), 1)

    def forward(self, features: Tensor) -> Tensor:
        if features.dim() != 2:
            raise ValueError("features must be a 2D tensor of shape (B, D)")
        return self.out(self.trunk(features)).squeeze(-1)


class ProfitableTradeProbabilityHead(nn.Module):
    """Predict unprofitable / unclear / profitable trade classes."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.trunk = _make_mlp(int(input_dim), int(hidden_dim), float(dropout))
        self.out = nn.Linear(int(hidden_dim), 3)

    def forward(self, features: Tensor) -> Tensor:
        if features.dim() != 2:
            raise ValueError("features must be a 2D tensor of shape (B, D)")
        return self.out(self.trunk(features))

    def probability(self, features: Tensor) -> Tensor:
        return torch.softmax(self(features), dim=-1)


class ReturnHeads(nn.Module):
    """Shared multitask head for return regression and profitability."""

    def __init__(
        self,
        *,
        order_feature_dim: int | None = None,
        batch_feature_dim: int | None = None,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if order_feature_dim is None and batch_feature_dim is None:
            raise ValueError("At least one feature source must be provided")

        self.order_feature_dim = None if order_feature_dim is None else int(order_feature_dim)
        self.batch_feature_dim = None if batch_feature_dim is None else int(batch_feature_dim)

        self.order_proj = (
            None
            if self.order_feature_dim is None
            else nn.Sequential(
                nn.LayerNorm(self.order_feature_dim),
                nn.Linear(self.order_feature_dim, int(hidden_dim)),
            )
        )
        self.batch_proj = (
            None
            if self.batch_feature_dim is None
            else nn.Sequential(
                nn.LayerNorm(self.batch_feature_dim),
                nn.Linear(self.batch_feature_dim, int(hidden_dim)),
            )
        )

        num_sources = int(self.order_proj is not None) + int(self.batch_proj is not None)
        fusion_dim = int(hidden_dim) * num_sources
        self.shared = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.return_head = nn.Linear(int(hidden_dim), 1)
        self.pnl_head = nn.Linear(int(hidden_dim), 1)
        self.profit_head = nn.Linear(int(hidden_dim), 3)

    def _combine_features(self, order_features: Tensor | None, batch_features: Tensor | None) -> Tensor:
        parts: list[Tensor] = []

        if order_features is not None:
            if self.order_proj is None:
                raise ValueError("order_features were provided but order_feature_dim was not configured")
            if order_features.dim() != 2:
                raise ValueError("order_features must be a 2D tensor of shape (B, D)")
            parts.append(self.order_proj(order_features))

        if batch_features is not None:
            if self.batch_proj is None:
                raise ValueError("batch_features were provided but batch_feature_dim was not configured")
            if batch_features.dim() != 2:
                raise ValueError("batch_features must be a 2D tensor of shape (B, D)")
            parts.append(self.batch_proj(batch_features))

        if not parts:
            raise ValueError("At least one feature tensor must be provided")

        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def forward(
        self,
        *,
        order_features: Tensor | None = None,
        batch_features: Tensor | None = None,
    ) -> dict[str, Tensor]:
        fused = self._combine_features(order_features=order_features, batch_features=batch_features)
        hidden = self.shared(fused)
        return {
            "pred_return": self.return_head(hidden).squeeze(-1),
            "pred_pnl": self.pnl_head(hidden).squeeze(-1),
            "profit_logits": self.profit_head(hidden),
        }

    def profit_probability(
        self,
        *,
        order_features: Tensor | None = None,
        batch_features: Tensor | None = None,
    ) -> Tensor:
        outputs = self(order_features=order_features, batch_features=batch_features)
        return torch.softmax(outputs["profit_logits"], dim=-1)


__all__ = [
    "FuturePnLRegressionHead",
    "FutureReturnRegressionHead",
    "ProfitableTradeProbabilityHead",
    "ReturnHeads",
]
