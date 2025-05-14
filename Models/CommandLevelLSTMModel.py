import torch
from torch import nn


class CommandLevelLSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_tokens = config['CommandLevelLSTM']['num_tokens']
        self.embedding = nn.Embedding(num_embeddings=config['CommandLevelLSTM']['token_dim'], embedding_dim=config['CommandLevelLSTM']['embedding_dim'])

        # encoder layers
        self.bilstm = nn.LSTM(input_size=config['CommandLevelLSTM']['embedding_dim'], hidden_size=config['CommandLevelLSTM']['hidden_dim'], num_layers=1, batch_first=True,
                              dropout=config['CommandLevelLSTM']['dropout'], bidirectional=True, proj_size=config['CommandLevelLSTM']['hidden_dim'] // 2)
        self.lstm = nn.LSTM(input_size=config['CommandLevelLSTM']['hidden_dim'], hidden_size=config['CommandLevelLSTM']['hidden_dim'], num_layers=config['CommandLevelLSTM']['n_layers'] - 1, batch_first=True,
                            dropout=config['CommandLevelLSTM']['dropout'])

        # final dense layer
        self.dense = nn.Linear(in_features=config['CommandLevelLSTM']['hidden_dim'], out_features=1)
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=config['CommandLevelLSTM']['n_bytes'], stride=config['CommandLevelLSTM']['n_bytes'], padding=0)
        if config['CommandLevelLSTM']['n_bytes'] == 1:
            self.conv1d = nn.Identity()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.to(torch.int32)
        x = self.embedding(x)
        x = self.bilstm(x)[0]
        x = self.lstm(x)[0]
        x = self.dense(x).transpose(dim0=1, dim1=2)
        x = self.conv1d(x).squeeze(dim=1)
        x = self.sigmoid(x)

        return x
