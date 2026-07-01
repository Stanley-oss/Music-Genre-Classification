from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size, padding) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.15, max_len: int = 512) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class MLPTimeBackbone(nn.Module):
    def __init__(
        self,
        emb_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 160,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        dims = list(hidden_dims or [192, 192, 192, 192, 192])
        self.trunk_channels = int(output_dim)

        layers: List[nn.Module] = [nn.LayerNorm(96)]
        in_dim = 96
        for hidden_dim in dims:
            layers.extend(
                [
                    nn.Linear(in_dim, int(hidden_dim)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, self.trunk_channels))
        self.mlp = nn.Sequential(*layers)

    def trunk(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h.mean(dim=2).transpose(1, 2))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.trunk(h)


class LSTMTimeBackbone(nn.Module):
    def __init__(
        self,
        emb_dim: int = 128,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        output_dim: int = 160,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_size = int(hidden_size)
        self.trunk_channels = int(output_dim)
        self.lstm = nn.LSTM(
            input_size=96,
            hidden_size=self.hidden_size,
            num_layers=int(num_layers),
            batch_first=True,
            bidirectional=True,
            dropout=dropout if int(num_layers) > 1 else 0.0,
        )
        out_dim = self.hidden_size * 2
        self.out_proj = nn.Identity() if out_dim == self.trunk_channels else nn.Linear(out_dim, self.trunk_channels)

    def trunk(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(h.mean(dim=2).transpose(1, 2))
        return self.out_proj(out)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.trunk(h)


class RNNTimeBackbone(nn.Module):
    def __init__(
        self,
        emb_dim: int = 128,
        hidden_size: int = 80,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.trunk_channels = int(hidden_size) * (2 if bidirectional else 1)
        self.rnn = nn.RNN(
            input_size=96,
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            bidirectional=bool(bidirectional),
            dropout=dropout if int(num_layers) > 1 else 0.0,
        )

    def trunk(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(h.mean(dim=2).transpose(1, 2))
        return out

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.trunk(h)


class TransformerTimeBackbone(nn.Module):
    def __init__(
        self,
        emb_dim: int = 128,
        d_model: int = 80,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 320,
        dropout: float = 0.15,
        output_dim: int = 160,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.trunk_channels = int(output_dim)
        self.proj = nn.Linear(96, int(d_model))
        self.pos_encoder = PositionalEncoding(d_model=int(d_model), dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.out_proj = nn.Identity() if int(d_model) == self.trunk_channels else nn.Linear(int(d_model), self.trunk_channels)

    def trunk(self, h: torch.Tensor) -> torch.Tensor:
        seq = self.proj(h.mean(dim=2).transpose(1, 2))
        return self.out_proj(self.encoder(self.pos_encoder(seq)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.trunk(h)


class BasicCNNBackbone(nn.Module):
    def __init__(self, emb_dim: int = 128, width_mult: float = 0.75, depth: int = 1) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        depth = max(1, int(depth))
        channels = [max(16, int(c * width_mult)) for c in (64, 96, 160)]
        self.trunk_channels = channels[-1]

        blocks: List[nn.Module] = [ConvBNAct(96, channels[0], 3, 1)]
        blocks.extend(ConvBNAct(channels[0], channels[0], 3, 1) for _ in range(depth - 1))
        blocks.append(nn.MaxPool2d(2))
        blocks.append(ConvBNAct(channels[0], channels[1], 3, 1))
        blocks.extend(ConvBNAct(channels[1], channels[1], 3, 1) for _ in range(depth - 1))
        blocks.append(nn.MaxPool2d(2))
        blocks.append(ConvBNAct(channels[1], channels[2], 3, 1))
        blocks.extend(ResidualConvBlock(channels[2]) for _ in range(depth))
        self.body = nn.Sequential(*blocks)

    def trunk(self, h: torch.Tensor) -> torch.Tensor:
        return self.body(h)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.trunk(h)


class ResNetBackbone(nn.Module):
    def __init__(self, emb_dim: int = 128, width_mult: float = 1.0, depth: int = 3) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        depth = max(1, int(depth))
        channels = [max(16, int(c * width_mult)) for c in (64, 96, 160)]
        self.trunk_channels = channels[-1]

        blocks: List[nn.Module] = [ConvBNAct(96, channels[0], 3, 1)]
        blocks.extend(ResidualConvBlock(channels[0]) for _ in range(depth))
        blocks.append(nn.MaxPool2d(2))
        blocks.append(ConvBNAct(channels[0], channels[1], 3, 1))
        blocks.extend(ResidualConvBlock(channels[1]) for _ in range(depth))
        blocks.append(nn.MaxPool2d(2))
        blocks.append(ConvBNAct(channels[1], channels[2], 3, 1))
        blocks.extend(ResidualConvBlock(channels[2]) for _ in range(depth))
        self.body = nn.Sequential(*blocks)

    def trunk(self, h: torch.Tensor) -> torch.Tensor:
        return self.body(h)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.trunk(h)
