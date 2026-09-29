# src/ssl_reconstruction.py

import torch
import torch.nn as nn
from typing import Tuple
from src.lstm_encoder import HybridCNNBiLSTMEncoder


class TemporalReconstructionDecoder(nn.Module):
    """
    Symmetric Decoder: maps sequence latent vectors (B, L, hidden_dim) 
    back to original physical measurement space (B, L, in_channels).
    """
    def __init__(self, latent_dim: int, out_channels: int, hidden_dim: int = 64):
        super().__init__()
        self.deconv_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels),
            nn.Sigmoid()  # Data was scaled to [0, 1] with MinMaxScaler
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, L, latent_dim)
        lstm_out, _ = self.deconv_lstm(z)  # (B, L, hidden_dim)
        reconstruction = self.projection(lstm_out)  # (B, L, out_channels)
        return reconstruction


class MaskedTemporalAutoencoder(nn.Module):
    """
    Masked Autoencoder Pretraining Framework for Smart Grid Time-Series.
    Combines the verified Hybrid 1D-CNN + BiLSTM backbone with a reconstruction decoder.
    """
    def __init__(
        self,
        in_channels: int = 25,
        cnn_hidden_dims: Tuple[int, ...] = (64, 128),
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        mask_ratio: float = 0.20
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.in_channels = in_channels
        
        # Core Feature Extractor Backbone (Phases 6 & 7)
        self.encoder = HybridCNNBiLSTMEncoder(
            in_channels=in_channels,
            cnn_hidden_dims=cnn_hidden_dims,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
            pool_strategy="none"  # Keep full temporal sequence for reconstruction
        )
        
        # Symmetrical Reconstruction Decoder
        self.decoder = TemporalReconstructionDecoder(
            latent_dim=self.encoder.out_dim,
            out_channels=in_channels,
            hidden_dim=lstm_hidden_dim
        )

    def apply_temporal_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly masks a fraction of timesteps across all channels 
        to force the model to learn temporal imputation dynamics.
        """
        if not self.training or self.mask_ratio <= 0.0:
            return x, torch.ones_like(x, dtype=torch.bool)

        batch_size, seq_len, num_channels = x.shape
        # Mask whole timesteps uniformly at random
        mask = torch.rand(batch_size, seq_len, device=x.device) > self.mask_ratio
        mask = mask.unsqueeze(-1).expand(-1, -1, num_channels)  # (B, L, C)
        
        x_masked = x * mask.float()
        return x_masked, mask

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Apply masking
        x_masked, mask = self.apply_temporal_mask(x)
        
        # 2. Extract latent representations via CNN + BiLSTM
        z = self.encoder(x_masked, return_sequence=True)  # (B, L, 2 * lstm_hidden_dim)
        
        # 3. Reconstruct uncorrupted signal
        x_rec = self.decoder(z)  # (B, L, in_channels)
        return x_rec, x_masked