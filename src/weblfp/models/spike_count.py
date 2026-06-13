import torch
import torch.nn as nn


class SpikeCountPresenceHead(nn.Module):
    """
    同时预测：
    - count log_rate: [B, 2]
    - presence logits: [B, 2]
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.count_head = nn.Linear(hidden_dim, 2)
        self.presence_head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        log_rate = self.count_head(h)
        presence_logits = self.presence_head(h)
        return log_rate, presence_logits

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_rate, presence_logits = self.forward(x)
        count = torch.exp(log_rate)
        presence_prob = torch.sigmoid(presence_logits)
        return count, presence_prob
