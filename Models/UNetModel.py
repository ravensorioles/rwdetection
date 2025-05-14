import torch
from torch import nn, sigmoid
import numpy as np


class EncoderMiniBlock(nn.Module):
    def __init__(self, filters_in, kernel_size=3):
        super().__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=filters_in, out_channels=filters_in * 2,
                                  kernel_size=kernel_size, padding='same')
        self.conv1d_2 = nn.Conv1d(in_channels=filters_in * 2, out_channels=filters_in * 2,
                                  kernel_size=kernel_size, padding='same')
        self.bn = nn.BatchNorm1d(filters_in * 2)
        self.mxpool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        x = nn.ReLU()(self.conv1d_1(x))
        skip_connection = self.bn(nn.ReLU()(self.conv1d_2(x)))
        next_layer = self.mxpool(skip_connection)

        return next_layer, skip_connection


class DecoderMiniBlock(nn.Module):
    def __init__(self, filters_in, output_padding, kernel_size=3):
        super().__init__()
        kernel_size_convt = max(kernel_size, 3)
        padding = (kernel_size_convt - 3) // 2 + output_padding
        self.convT1d = nn.ConvTranspose1d(in_channels=filters_in, out_channels=int(filters_in / 2),
                                          kernel_size=kernel_size_convt, stride=2, padding=padding,
                                          output_padding=output_padding)
        self.conv1d_1 = nn.Conv1d(in_channels=filters_in, out_channels=int(filters_in / 2),
                                  kernel_size=kernel_size, padding='same')
        self.conv1d_2 = nn.Conv1d(in_channels=int(filters_in / 2), out_channels=int(filters_in / 2),
                                  kernel_size=kernel_size, padding='same')

    def forward(self, x, x_skip):
        x = self.convT1d(x)
        x = torch.cat([x, x_skip], dim=1)
        x = nn.ReLU()(self.conv1d_1(x))
        x = nn.ReLU()(self.conv1d_2(x))

        return x


class UNetModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.no_tokens = config['UNet']['num_tokens']
        self.no_channels = config['UNet']['d_model']
        self.depth = min(3, np.floor(np.log2(self.no_tokens)).astype(int))
        self.kernel_size = config['UNet']['kernel_size']

        filters_in = self.no_channels

        self.convT_padding = [1]
        for step in range(self.depth + 1):
            self.convT_padding.append(int((self.no_tokens / (2 ** (step - 1)) + 1) % 2))  # The 1d transpose convolution that
            # should bring us to the skip dimensions correctly is 0 if the skip dimension is even and 1 if odd
            setattr(self, 'encoder_{}'.format(step), EncoderMiniBlock(filters_in=filters_in, kernel_size=self.kernel_size))
            filters_in *= 2

        for step in range(self.depth + 1, 0, -1):
            if step < self.depth + 1:
                filters_in /= 2
            setattr(self, 'decoder_{}'.format(step - 1), DecoderMiniBlock(filters_in=int(filters_in),
                                                                          output_padding=self.convT_padding[step],
                                                                          kernel_size=self.kernel_size))

        self.conv1d = nn.Conv1d(in_channels=2 * self.no_channels, out_channels=self.no_channels,
                                kernel_size=self.kernel_size, padding='same')

        self.dense = nn.Linear(self.no_channels, 2)

    def forward(self, x):
        skips = []
        for step in range(self.depth):
            x, skip = getattr(self, 'encoder_{}'.format(step))(x)
            skips.append(skip)

        # Bottom branch
        _, x = getattr(self, 'encoder_{}'.format(self.depth))(x)

        for step in range(self.depth, 0, -1):
            x = getattr(self, 'decoder_{}'.format(step))(x, skips[step - 1])

        x = nn.ReLU()(self.conv1d(x)).permute(0, 2, 1)
        x = self.dense(x).permute(0, 2, 1)

        x = sigmoid(x)

        return x
