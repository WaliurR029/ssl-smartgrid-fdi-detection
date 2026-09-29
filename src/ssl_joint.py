# src/ssl_joint.py

import torch
import torch.nn as nn
from typing import Tuple
from src.lstm_encoder import HybridCNNBiLSTMEncoder
from src.ssl_reconstruction import TemporalReconstructionDecoder
from src.ssl_contrastive import GridDataAugmenter, NTXentLoss


class DualPretextSSLModel(nn.Module):
    """
    Unified Dual Pretext Self-Supervised Learning Framework.
    Jointly optimizes Masked Reconstruction (local imputation) and
    Contrastive Learning (global representation invariance) on a shared backbone.
    """
    def __init__(
        self,
        in_channels: int = 25,
        cnn_hidden_dims: Tuple[int, ...] = (64, 128),
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        proj_dim: int = 64,
        mask_ratio: float = 0.20
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.in_channels = in_channels

        # 1. Main Research Backbone: Shared CNN + BiLSTM
        self.encoder = HybridCNNBiLSTMEncoder(
            in_channels=in_channels,
            cnn_hidden_dims=cnn_hidden_dims,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
            pool_strategy="mean"  # Default global pool
        )
        self.latent_dim = self.encoder.out_dim  # 2 * lstm_hidden_dim = 256

        # 2. Pretext Head A: Temporal Reconstruction Decoder
        self.decoder = TemporalReconstructionDecoder(
            latent_dim=self.latent_dim,
            out_channels=in_channels,
            hidden_dim=lstm_hidden_dim
        )

        # 3. Pretext Head B: Contrastive Projection MLP Head
        self.projection_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim, proj_dim)
        )

    def apply_temporal_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stochastically masks entire timesteps across all channels."""
        if not self.training or self.mask_ratio <= 0.0:
            return x, torch.ones_like(x, dtype=torch.bool)
        
        batch_size, seq_len, num_channels = x.shape
        mask = torch.rand(batch_size, seq_len, device=x.device) > self.mask_ratio
        mask = mask.unsqueeze(-1).expand(-1, -1, num_channels)
        x_masked = x * mask.float()
        return x_masked, mask

    def forward_reconstruction(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs the masked autoencoding branch."""
        x_masked, mask = self.apply_temporal_mask(x)
        # Sequence-level representation across all timesteps (B, L, 256)
        z_seq = self.encoder(x_masked, return_sequence=True)
        x_rec = self.decoder(z_seq)  # (B, L, 25)
        return x_rec, x_masked

    def forward_contrastive(self, view: torch.Tensor) -> torch.Tensor:
        """Runs global sequence representation through projection head."""
        h = self.encoder(view, return_sequence=False)  # (B, 256)
        z = self.projection_head(h)                   # (B, proj_dim)
        return z