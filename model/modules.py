""" adapted from https://github.com/r9y9/tacotron_pytorch """
""" with reference to https://github.com/CorentinJ/Real-Time-Voice-Cloning/tree/b5ba6d0371882dbab595c48deb2ff17896547de7/synthesizer """
""" adapted from https://github.com/NVIDIA/tacotron2 """

import torch
from torch import nn


class Prenet(nn.Module):
    """
    Prenet
        - Several linear layers with ReLU activation and dropout regularization
    """
    def __init__(self, in_dim, sizes=[256, 128], dropout=0.5):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size)
             for (in_size, out_size) in zip(in_sizes, sizes)])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        for linear in self.layers:
            inputs = self.dropout(self.relu(linear(inputs)))
        return inputs


class BatchNormConv1dStack(nn.Module):
    """
    BatchNormConv1dStack
        - A stack of 1-d convolution layers
        - Each convolution layer is followed by activation function (optional), Batch Normalization (BN) and dropout
    """
    def __init__(self, in_channel,
                 out_channels=[512, 512, 512], kernel_size=3, stride=1, padding=1,
                 activations=None, dropout=0.5):
        super(BatchNormConv1dStack, self).__init__()

        # Validation check
        if activations is None:
            activations = [None] * len(out_channels)
        assert len(activations) == len(out_channels)

        # 1-d convolutions with BN
        in_sizes = [in_channel] + out_channels[:-1]
        self.conv1ds = nn.ModuleList(
            [BatchNormConv1d(in_size, out_size, kernel_size=kernel_size, stride=stride,
                             padding=padding, activation=ac)
             for (in_size, out_size, ac) in zip(in_sizes, out_channels, activations)])

        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for conv1d in self.conv1ds:
            x = self.dropout(conv1d(x))
        return x


class BatchNormConv1d(nn.Module):
    """
    BatchNormConv1d
        - 1-d convolution layer with specific activation function, followed by Batch Normalization (BN)

    Batch Norm before activation or after the activation?
    Still in debation!
    In practace, applying batch norm after the activation yields bettr results.
        - https://blog.paperspace.com/busting-the-myths-about-batch-normalization/
        - https://medium.com/@nihar.kanungo/batch-normalization-and-activation-function-sequence-confusion-4e075334b4cc
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=None):
        super(BatchNormConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels,
                                kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv1d(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.bn(x)


class Highway(nn.Module):
    """
    Highway network
    """
    def __init__(self, size):
        super(Highway, self).__init__()
        self.H = nn.Linear(size, size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(size, size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)


class CBHG(nn.Module):
    """
    CBHG module: a recurrent neural network composed of:
        - 1-d convolution bank
        - Highway networks + residual connections
        - Bidirectional gated recurrent units (Bi-GRU)
    """

    def __init__(self, in_dim, K=16, conv_channels=128, pool_kernel_size=2,
                 proj_channels=[128, 128], proj_kernel_size=3,
                 num_highways=4, highway_units=128, rnn_units=128):
        super(CBHG, self).__init__()

        # List of all rnns to call `flatten_parameters()` on
        self._to_flatten = []

        # 1-d convolution bank
        self.relu = nn.ReLU()
        self.conv1d_bank = nn.ModuleList(
            [BatchNormConv1d(in_dim, conv_channels, kernel_size=k, stride=1,
                             padding=k // 2, activation=self.relu)
             for k in range(1, K + 1)])

        # Max-pooling
        self.max_pool1d = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=1, padding=1)

        # conv1d projections
        in_sizes = [K * conv_channels] + proj_channels[:-1]
        activations = [self.relu] * (len(proj_channels) - 1) + [None]
        self.conv1d_projections = nn.ModuleList(
            [BatchNormConv1d(in_size, out_size, kernel_size=proj_kernel_size, stride=1,
                             padding=1, activation=ac)
             for (in_size, out_size, ac) in zip(in_sizes, proj_channels, activations)])
        # for residual connection
        assert proj_channels[-1] == in_dim

        # Fix the highway input if necessary
        if proj_channels[-1] != highway_units:
            self.pre_highway = nn.Linear(proj_channels[-1], highway_units, bias=False)
        else:
            self.pre_highway = None

        # Highway networks
        self.highways = nn.ModuleList(
            [Highway(highway_units) for _ in range(num_highways)])

        # Bi-GRU
        self.gru = nn.GRU(highway_units, rnn_units, batch_first=True, bidirectional=True)
        self._to_flatten.append(self.gru)

        # Avoid fragmentation of RNN parameters and associated warning
        self._flatten_parameters()

    def forward(self, inputs):
        # Although we `_flatten_parameters()` on init, when using DataParallel
        # the model gets replicated, making it no longer guaranteed that the
        # weights are contiguous in GPU memory. Hence, we must call it again
        self._flatten_parameters()

        # (B, T_in, in_dim)
        x = inputs
        T = x.size(1)

        # Needed to perform conv1d on time-axis
        # (B, in_dim, T_in)
        x = x.transpose(1, 2)

        # (B, conv_channels*K, T_in)
        # Concat conv1d bank outputs along the channel axis
        x = torch.cat([conv1d(x)[:, :, :T] for conv1d in self.conv1d_bank], dim=1)

        # Max-pooling
        x = self.max_pool1d(x)[:, :, :T]

        # Conv1d projections
        for conv1d in self.conv1d_projections:
            x = conv1d(x)

        # Back to the original shape
        # (B, T_in, in_dim)
        x = x.transpose(1, 2)

        # Residual connection
        x += inputs

        # Pre-highway
        # (B, T_in, highway_units)
        if self.pre_highway is not None:
            x = self.pre_highway(x)

        # Through the Highways
        # (B, T_in, highway_units)
        for highway in self.highways:
            x = highway(x)

        # And then the Bi-GRU
        # (B, T_in, rnn_units*2)
        x, _ = self.gru(x)
        return x

    def _flatten_parameters(self):
        """Calls `flatten_parameters` on all the rnns. Used
        to improve efficiency and avoid PyTorch yelling at us."""
        [m.flatten_parameters() for m in self._to_flatten]
