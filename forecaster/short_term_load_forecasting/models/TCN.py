############################ CENTRALIZED APPROACH ############################
# libraries
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import matplotlib.pyplot as plt
import numpy as np


# Use graphs edition
from ..graficas import *
plot = True

"""
TCN Neural Network in a recursive and a MIMO approach.

-------------------------------References--------------------------------------
[1] Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks
for sequence modeling. arXiv preprint arXiv:1803.01271.
[2] unit8, “Temporal Convolutional Networks and Forecasting,” Unit8.
https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/
"""


# LSTM recursive neural network
class LSTMRecursive(nn.Module):
    def __init__(self):
        super(LSTMRecursive, self).__init__()
        # Capa LSTM
        self.lstm1 = nn.LSTM(192, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, batch_first=True)
        self.lstm3 = nn.LSTM(64, 192, batch_first=True)
        self.relu = nn.ReLU()
        # Capa completamente conectada
        self.fc1 = nn.Linear(65, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        results = torch.empty((0))
        x, d = x[:, :-1, :], x[:, -1, :]
        for i in range(24):
            # Obtener la salida de la capa LSTM
            x_, _ = self.lstm1(x)
            x_ = self.relu(x_)
            x_, _ = self.lstm2(x_, _)
            # lstm_out, _ = self.lstm3(x)
            # Tomar el último paso de la secuencia como representación
            # lstm_out = self.relu(lstm_out[:, -1, :])
            x_ = torch.cat((self.relu(x_[:, -1, :]), d), dim=1)
            # Aplicar la capa completamente conectada
            x_ = self.fc1(x_)
            output = self.fc2(x_)
            results = torch.cat((results, output), 1)
            x = torch.cat((x[:, 1:, :], output.unsqueeze(1)), 1)
        return results


# LSTM neural network MIMO
class LSTMMIMO(nn.Module):
    def __init__(self):
        super(LSTMMIMO, self).__init__()
        # Capa LSTM
        self.lstm1 = nn.LSTM(192, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, batch_first=True)
        self.lstm3 = nn.LSTM(64, 192, batch_first=True)
        self.relu = nn.ReLU()
        # Capa completamente conectada
        self.fc1 = nn.Linear(65, 64)
        self.fc2 = nn.Linear(64, 24)

    def forward(self, x):
        x, d = x[:, :-1, :], x[:, -1, :]
        # Obtener la salida de la capa LSTM
        x, _ = self.lstm1(x)
        x = self.relu(x)
        x, _ = self.lstm2(x, _)
        x = torch.cat((self.relu(x[:, -1, :]), d), dim=1)
        # Aplicar la capa completamente conectada
        x = self.fc1(x)
        output = self.fc2(x)
        return output


# TCN Neural Network
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1,
                                 self.relu1,
                                 self.dropout1,
                                 self.conv2, self.chomp2,
                                 # self.relu2,
                                 self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return (out + res)  # self.relu(out + res) #


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self):
        super(TCN, self).__init__()
        self.input_size = 1
        self.output_size = 24
        self.num_channels = [1] * 7
        self.kernel_size = 9
        self.tcn = TemporalConvNet(self.input_size, self.num_channels, self.kernel_size, dropout=0.25)
        self.linear1 = nn.Linear(193, 64)
        # self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, self.output_size)  # num_channels[-1], output_size)
        self.sig = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        self.linear1.weight.data.normal_(0, 0.01)
        self.linear3.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x = x.unsqueeze(1)
        x, d = x[:, :-1, :], x[:, -1, :]
        x = x.permute(0, 2, 1)
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(x)  # .transpose(1, 2).transpose(1, 2)
        output = torch.cat((self.sig(output), d.unsqueeze(dim=1)), dim=2)
        output = self.linear1(output)
        # output = self.linear2(output)
        output = self.linear3(output)  # .double()
        return output  # self.sig(output)


class TCNRecursive(nn.Module):
    def __init__(self):
        super(TCNRecursive, self).__init__()
        self.input_size = 1
        self.output_size = 24
        self.num_channels = [1] * 7
        self.kernel_size = 9
        self.tcn = TemporalConvNet(self.input_size, self.num_channels, self.kernel_size, dropout=0.25)
        self.linear1 = nn.Linear(193, 64)
        self.linear3 = nn.Linear(64, self.output_size)  # num_channels[-1], output_size)
        self.sig = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        self.linear1.weight.data.normal_(0, 0.01)
        self.linear3.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x, d = x[:, :-1, :], x[:, -1, :]
        x = x.permute(0, 2, 1)
        # return output
        results = torch.empty((0))
        for i in range(24):
            x_ = self.tcn(x)
            x_ = torch.cat((self.sig(x_), d.unsqueeze(dim=1)), dim=2)
            x_ = self.linear1(x_)
            output = self.linear3(x_)
            results = torch.cat((results, output), 1)
            x = torch.cat((x[:, :, 1:], output), 2)
        return results


def main():
    pass

if __name__ == "__main__":
    # main()
    pass