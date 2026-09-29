# src/cnn_encoder.py

import torch
import torch.nn as nn
from typing import Optional, Tuple


class Conv1DBlock(nn.Module):
    """
    Standard single-layer 1D-CNN unit: Conv1d -> BatchNorm1d -> Activation -> Dropout.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Residual projection shortcut if dimensions change
        self.shortcut = nn.Identity()
        if use_residual and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        self.use_residual = use_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x) if self.use_residual else 0.0
        out = self.act(self.bn(self.conv(x)))
        out = self.drop(out)
        if self.use_residual:
            out = out + res
        return out


class StandaloneCNNEncoder(nn.Module):
    """
    Modular 1D-CNN Feature Extractor for Smart Grid Time Series:
    Input (Batch, Length, Channels) -> Conv1D Hierarchies -> Temporal Features.
    """
    def __init__(
        self,
        in_channels: int,
        cnn_hidden_dims: Tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 3,
        dropout: float = 0.2,
        pool_type: Optional[str] = "max",  # Options: 'max', 'avg', 'global_avg', None
        pool_size: int = 2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.pool_type = pool_type
        
        # Build multi-stage convolutional layers
        layers = []
        curr_channels = in_channels
        padding = kernel_size // 2  # Preserves temporal length across convolutions
        
        for h_dim in cnn_hidden_dims:
            layers.append(
                Conv1DBlock(
                    in_channels=curr_channels,
                    out_channels=h_dim,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dropout=dropout
                )
            )
            curr_channels = h_dim

        self.conv_stack = nn.Sequential(*layers)
        self.out_dim = cnn_hidden_dims[-1]

        # Configurable pooling module
        if pool_type == "max":
            self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        elif pool_type == "avg":
            self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_size)
        elif pool_type == "global_avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pool_type is None:
            self.pool = nn.Identity()
        else:
            raise ValueError(f"Unsupported pool_type: {pool_type}. Choose 'max', 'avg', 'global_avg', or None.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass:
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Features)
        Returns:
            features: Temporal embeddings of shape:
                      - (Batch, Pooled_Len, Out_Dim) if sequence-preserving or downsampled
                      - (Batch, Out_Dim) if pool_type == 'global_avg'
        """
        # Convert from (Batch, Seq_Len, Features) to (Batch, Features, Seq_Len)
        x_transposed = x.permute(0, 2, 1)
        
        # 1D-CNN temporal feature learning
        feat = self.conv_stack(x_transposed)
        
        # Temporal pooling
        feat_pooled = self.pool(feat)
        
        if self.pool_type == "global_avg":
            # (Batch, Out_Dim, 1) -> (Batch, Out_Dim)
            return feat_pooled.squeeze(-1)
        
        # Permute back to standard format: (Batch, Seq_Len_Pooled, Out_Dim)
        return feat_pooled.permute(0, 2, 1)