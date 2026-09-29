# src/fdi_classifier.py

import torch
import torch.nn as nn
from typing import Tuple
from src.lstm_encoder import HybridCNNBiLSTMEncoder


class TransferFDIClassifier(nn.Module):
    """
    Downstream FDI Detection Architecture with Transfer Learning Backbone.
    Adapts 12-channel feeder data -> Pretrained CNN+BiLSTM -> Classification MLP.
    """
    def __init__(
        self,
        in_channels: int = 12,
        pretrained_channels: int = 25,
        cnn_hidden_dims: Tuple[int, ...] = (64, 128),
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        mlp_hidden_dim: int = 64,
        dropout: float = 0.3,
        freeze_backbone: bool = True
    ):
        super().__init__()
        
        # 1. 1x1 Pointwise Channel Projector: 12 feeder channels -> 25 pretrained channels
        self.channel_projector = nn.Conv1d(
            in_channels=in_channels,
            out_channels=pretrained_channels,
            kernel_size=1
        )
        
        # 2. Pretrained CNN + BiLSTM Backbone
        self.encoder = HybridCNNBiLSTMEncoder(
            in_channels=pretrained_channels,
            cnn_hidden_dims=cnn_hidden_dims,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
            pool_strategy="mean"
        )
        
        # 3. Downstream Classification MLP Head
        self.mlp_head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, mlp_hidden_dim),  # 256 -> 64
            nn.BatchNorm1d(mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1)  # Binary logit output
        )
        
        # Apply freezing strategy if specified
        if freeze_backbone:
            self.set_backbone_frozen(True)

    def set_backbone_frozen(self, frozen: bool):
        """Freezes or unfreezes backbone parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = not frozen
            
    def load_pretrained_backbone(self, checkpoint_path: str, device: torch.device):
        """Loads Phase 11 pretrained weights into the encoder backbone."""
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.encoder.load_state_dict(state_dict)
        print(f"[SUCCESS] Loaded Pretrained Encoder Backbone from: {checkpoint_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Seq_Len=30, In_Channels=12)
        # Transpose for 1D convolution: (Batch, 12, 30)
        x_transposed = x.transpose(1, 2)
        x_proj = self.channel_projector(x_transposed)
        # Restore shape: (Batch, 30, 25)
        x_adapted = x_proj.transpose(1, 2)
        
        # Feature representation: (Batch, 256)
        representation = self.encoder(x_adapted, return_sequence=False)
        
        # Binary classification logit: (Batch, 1)
        logit = self.mlp_head(representation)
        return logit