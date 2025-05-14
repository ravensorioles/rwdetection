import torch
from torch import nn


class CommandLevelLSTMContinuous(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_tokens = config['CommandLevelLSTMContinuous']['num_tokens']
        self.embedding = nn.Embedding(num_embeddings=config['CommandLevelLSTMContinuous']['token_dim'], embedding_dim=config['CommandLevelLSTMContinuous']['embedding_dim'])

        # encoder layers
        self.lstm = nn.LSTM(input_size=config['CommandLevelLSTMContinuous']['embedding_dim'], hidden_size=config['CommandLevelLSTMContinuous']['hidden_dim'], num_layers=config['CommandLevelLSTMContinuous']['n_layers'], batch_first=True,
                            dropout=config['CommandLevelLSTMContinuous']['dropout'])

        # final dense layer
        self.dense = nn.Linear(in_features=config['CommandLevelLSTMContinuous']['hidden_dim'], out_features=1)
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=config['CommandLevelLSTMContinuous']['n_bytes'], stride=config['CommandLevelLSTMContinuous']['n_bytes'], padding=0)
        if config['CommandLevelLSTMContinuous']['n_bytes'] == 1:
            self.conv1d = nn.Identity()
        self.sigmoid = nn.Sigmoid()
        self.h, self.c = None, None

    def forward(self, x):
        if isinstance(x, tuple):  # To be coherent with the TorchManager
            x, reuse = x
            x = x.to(torch.int32)
            x = self.embedding(x)
            x, (self.h, self.c) = self.lstm(x, (self.h, self.c)) if reuse else self.lstm(x)
        else:
            x = x.to(torch.int32)
            x = self.embedding(x)
            x, (self.h, self.c) = self.lstm(x)
        x = self.dense(x).transpose(dim0=1, dim1=2)
        x = self.conv1d(x).squeeze(dim=1)
        x = self.sigmoid(x)

        return x
