import torch
from torch import nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe[None, :, :]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PLTModel(nn.Module):
    def __init__(self, config, *args, **kwargs):
        # embedding layer is not necessary: was done by tokenization
        super().__init__(*args, **kwargs)

        raw_feature_dim = config['PLT']['raw_feature_dim']
        embedding_dim = config['PLT']['embedding_dim']
        hidden_size = config['PLT']['hidden_size']
        nheads = config['PLT']['nheads']
        n_layers = config['PLT']['n_layers']
        max_len = config['PLT']['max_len']
        dropout = config['PLT']['dropout']

        self.pre_embedder = nn.Linear(in_features=raw_feature_dim, out_features=embedding_dim, bias=False)

        # positional encoding layer
        self.pe = PositionalEncoding(d_model=embedding_dim, max_len=max_len)

        # encoder  layers
        enc_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nheads,
                                               dim_feedforward=hidden_size, dropout=dropout,
                                               batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=enc_layer, num_layers=n_layers)

        # final dense layer
        self.last_ln = nn.LayerNorm(
            hidden_size)
        self.dense = nn.Linear(embedding_dim, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.ndimension() <= 2:
            x = x.unsqueeze(0)
        x = x.permute(0, 2, 1)
        x = self.pre_embedder(x)
        x = self.pe(x)
        x = self.encoder(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x.permute(0, 2, 1)
