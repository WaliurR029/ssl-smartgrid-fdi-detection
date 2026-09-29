# src/lstm_encoder.py

from typing import Optional, Tuple
import torch
import torch.nn as nn
from src.cnn_encoder import StandaloneCNNEncoder


class StandaloneBiLSTMEncoder(nn.Module):
  """Bidirectional LSTM sequence processor with multi-strategy temporal pooling."""

  def __init__(
      self,
      in_dim: int,
      hidden_dim: int = 64,
      num_layers: int = 2,
      dropout: float = 0.2,
      pool_strategy: str = "mean",  # Options: 'mean', 'last', 'attn_pool', 'none'
  ):
    super().__init__()
    self.in_dim = in_dim
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.pool_strategy = pool_strategy

    self.lstm = nn.LSTM(
        input_size=in_dim,
        hidden_size=hidden_dim,
        num_layers=num_layers,
        batch_first=True,
        bidirectional=True,
        dropout=dropout if num_layers > 1 else 0.0,
    )
    # Bidirectional processing concatenates forward and backward passes: hidden_dim * 2
    self.out_dim = hidden_dim * 2

    if pool_strategy == "attn_pool":
      self.attention_weights = nn.Sequential(
          nn.Linear(self.out_dim, 32), nn.Tanh(), nn.Linear(32, 1)
      )

  def forward(
      self, x: torch.Tensor
  ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Forward pass:

    Args:
        x: Input tensor of shape (Batch, Seq_Len, in_dim)
    Returns:
        representation: (Batch, out_dim) if pooled, or (Batch, Seq_Len, out_dim)
        if pool_strategy == 'none'
        (h_n, c_n): Raw final recurrent hidden and cell states
    """
    lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (B, L, 2 * hidden_dim)

    if self.pool_strategy == "mean":
      # Average across all temporal steps
      rep = torch.mean(lstm_out, dim=1)
    elif self.pool_strategy == "last":
      # Concatenate last step of forward LSTM and first step of backward LSTM
      forward_last = lstm_out[:, -1, : self.hidden_dim]
      backward_first = lstm_out[:, 0, self.hidden_dim :]
      rep = torch.cat([forward_last, backward_first], dim=-1)
    elif self.pool_strategy == "attn_pool":
      # Learn dynamic attention weights across sequence positions
      scores = self.attention_weights(lstm_out)  # (B, L, 1)
      weights = torch.softmax(scores, dim=1)  # (B, L, 1)
      rep = torch.sum(lstm_out * weights, dim=1)  # (B, 2 * hidden_dim)
    elif self.pool_strategy == "none":
      rep = lstm_out
    else:
      raise ValueError(f"Unknown pool_strategy: {self.pool_strategy}")

    return rep, (h_n, c_n)


class HybridCNNBiLSTMEncoder(nn.Module):
  """Unified Backbone Encoder:

  Connects: Input -> 1D-CNN (Local Features) -> BiLSTM (Long-range Context) ->
  Representation
  """

  def __init__(
      self,
      in_channels: int,
      cnn_hidden_dims: Tuple[int, ...] = (32, 64),
      cnn_kernel_size: int = 3,
      cnn_dropout: float = 0.2,
      lstm_hidden_dim: int = 64,
      lstm_layers: int = 2,
      lstm_dropout: float = 0.2,
      pool_strategy: str = "mean",
  ):
    super().__init__()
    # Front-end: 1D-CNN (pool_type=None preserves full sequence length for the BiLSTM)
    self.cnn = StandaloneCNNEncoder(
        in_channels=in_channels,
        cnn_hidden_dims=cnn_hidden_dims,
        kernel_size=cnn_kernel_size,
        dropout=cnn_dropout,
        pool_type=None,
    )

    # Back-end: BiLSTM processes CNN feature maps across time
    self.bilstm = StandaloneBiLSTMEncoder(
        in_dim=self.cnn.out_dim,
        hidden_dim=lstm_hidden_dim,
        num_layers=lstm_layers,
        dropout=lstm_dropout,
        pool_strategy=pool_strategy,
    )
    self.out_dim = self.bilstm.out_dim

  def forward(
      self, x: torch.Tensor, return_sequence: bool = False
  ) -> torch.Tensor:
    """Args:

        x: (Batch, Seq_Len, in_channels)
        return_sequence: If True, overrides pooling and returns (Batch, Seq_Len,
        out_dim)
    Returns:
        z: Sequence embedding (Batch, out_dim) or temporal sequence (Batch,
        Seq_Len, out_dim)
    """
    # 1. Extract localized cross-channel spatial-temporal patterns
    cnn_feats = self.cnn(x)  # (B, L, cnn_out_dim)

    # 2. Extract long-range bidirectional dynamics
    if return_sequence:
      prev_pool = self.bilstm.pool_strategy
      self.bilstm.pool_strategy = "none"
      rep, _ = self.bilstm(cnn_feats)
      self.bilstm.pool_strategy = prev_pool
      return rep

    rep, _ = self.bilstm(cnn_feats)  # (B, 2 * lstm_hidden_dim)
    return rep