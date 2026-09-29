# src/ssl_contrastive.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from src.lstm_encoder import HybridCNNBiLSTMEncoder


class GridDataAugmenter:
    """
    Physically grounded augmentations for multivariate smart grid load curves.
    """
    def __init__(
        self,
        jitter_sigma: float = 0.015,
        scale_range: Tuple[float, float] = (0.90, 1.10),
        mask_block_len: int = 4
    ):
        self.jitter_sigma = jitter_sigma
        self.scale_range = scale_range
        self.mask_block_len = mask_block_len

    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.jitter_sigma
        return torch.clamp(x + noise, min=0.0, max=1.0)

    def scaling(self, x: torch.Tensor) -> torch.Tensor:
        # Sample uniform scale factor per sequence in the batch: (B, 1, 1)
        b, _, _ = x.shape
        low, high = self.scale_range
        factor = (low + (high - low) * torch.rand(b, 1, 1, device=x.device))
        return torch.clamp(x * factor, min=0.0, max=1.0)

    def block_mask(self, x: torch.Tensor) -> torch.Tensor:
        # Simulate an intermittent AMI packet loss over a contiguous time block
        x_aug = x.clone()
        b, seq_len, _ = x_aug.shape
        start_idx = torch.randint(0, seq_len - self.mask_block_len + 1, (b,))
        for i in range(b):
            s = start_idx[i]
            x_aug[i, s : s + self.mask_block_len, :] = 0.0
        return x_aug

    def generate_augmented_view(self, x: torch.Tensor) -> torch.Tensor:
        """Composes augmentations stochastically."""
        x_view = x.clone()
        if torch.rand(1).item() > 0.3:
            x_view = self.jitter(x_view)
        if torch.rand(1).item() > 0.3:
            x_view = self.scaling(x_view)
        if torch.rand(1).item() > 0.4:
            x_view = self.block_mask(x_view)
        return x_view


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent / SimCLR InfoNCE).
    Treats two augmented views of sequence i as positive pair, all other sequences as negatives.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i: Projections from View A, shape (Batch, Proj_Dim)
            z_j: Projections from View B, shape (Batch, Proj_Dim)
        """
        batch_size = z_i.shape[0]

        # 1. L2 Normalize representations along embedding dimension
        z_i_norm = F.normalize(z_i, dim=1)
        z_j_norm = F.normalize(z_j, dim=1)

        # 2. Concatenate: total representations = 2 * batch_size
        representations = torch.cat([z_i_norm, z_j_norm], dim=0)  # (2B, D)

        # 3. Pairwise cosine similarity matrix
        sim_matrix = torch.matmul(representations, representations.T) / self.temperature  # (2B, 2B)

        # 4. Mask out self-similarity (diagonal)
        sim_indices = torch.eye(2 * batch_size, device=z_i.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(sim_indices, -1e9)

        # 5. Positive pairs ground truth targets:
        # View A_i targets View B_i (index + batch_size), View B_i targets View A_i (index - batch_size)
        pos_targets = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z_i.device),
            torch.arange(0, batch_size, device=z_i.device)
        ], dim=0)

        # 6. Standard Cross Entropy computes InfoNCE
        loss = F.cross_entropy(sim_matrix, pos_targets)
        return loss


class ContrastiveSSLModel(nn.Module):
    """
    End-to-End Temporal Contrastive Pretraining Network.
    Shared Backbone: 1D-CNN + BiLSTM (Phases 6-7)
    Projection Head: 2-layer MLP (hidden -> proj_dim)
    """
    def __init__(
        self,
        in_channels: int = 25,
        cnn_hidden_dims: Tuple[int, ...] = (64, 128),
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        proj_dim: int = 64
    ):
        super().__init__()
        # Backbone Feature Extractor
        self.encoder = HybridCNNBiLSTMEncoder(
            in_channels=in_channels,
            cnn_hidden_dims=cnn_hidden_dims,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
            pool_strategy="mean"  # Pools time series into sequence representation vector
        )

        # Non-linear projection head (discarded after pretraining, retains rich features in backbone)
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, self.encoder.out_dim),
            nn.BatchNorm1d(self.encoder.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder.out_dim, proj_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: backbone sequence representation (B, 2 * lstm_hidden_dim)
        h = self.encoder(x, return_sequence=False)
        # z: projected vector for contrastive matching (B, proj_dim)
        z = self.projection_head(h)
        return h, z