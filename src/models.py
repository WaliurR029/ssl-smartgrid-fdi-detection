# src/models.py

import torch
import torch.nn as nn

class TemporalCNNEncoder(nn.Module):
    """
    1D-CNN temporal feature extractor.
    Learns spatial coupling across feeder channels and local temporal patterns.
    """
    def __init__(self, in_channels: int, cnn_out_dim: int = 64):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(in_channels=32, out_channels=cnn_out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(cnn_out_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (Batch, Seq_Len, Features) -> permute to (Batch, Features, Seq_Len)
        x = x.permute(0, 2, 1)
        feat = self.conv_net(x)
        # Permute back to (Batch, Seq_Len, cnn_out_dim) for LSTM
        return feat.permute(0, 2, 1)


class BiLSTMTemporalEncoder(nn.Module):
    """
    Bidirectional LSTM capturing dynamic context across the temporal sequence.
    """
    def __init__(self, in_dim: int = 64, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.out_dim = hidden_dim * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Seq_Len, in_dim)
        lstm_out, _ = self.bilstm(x)
        # Temporal average pooling: (Batch, hidden_dim * 2)
        pooled = torch.mean(lstm_out, dim=1)
        return pooled


class FDIDetectorBaseline(nn.Module):
    """
    Scratch Supervised Baseline:
    Input (B, L, F) -> 1D-CNN -> BiLSTM -> Global Pooling -> MLP -> Logit
    """
    def __init__(self, in_features: int, cnn_dim: int = 64, lstm_hidden: int = 64):
        super().__init__()
        self.cnn = TemporalCNNEncoder(in_channels=in_features, cnn_out_dim=cnn_dim)
        self.bilstm = BiLSTMTemporalEncoder(in_dim=cnn_dim, hidden_dim=lstm_hidden)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.bilstm.out_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_cnn = self.cnn(x)
        h_lstm = self.bilstm(h_cnn)
        logits = self.classifier(h_lstm)
        return logits.squeeze(-1)