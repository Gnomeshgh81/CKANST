import torch.nn.functional as F
import torch.nn as nn
import torch
from module.basic_model import BasicModel
from kan_convs import KANConv2DLayer

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()

        self.conv3_3_1 = KANConv2DLayer(input_dim, hidden_dim, 3)
        self.conv3_3_2 = KANConv2DLayer(hidden_dim, 2 * hidden_dim, 3)
        self.conv3_3_3 = KANConv2DLayer(2 * hidden_dim, 4 * hidden_dim, 3)
        self.conv3_3_4 = KANConv2DLayer(4 * hidden_dim, 4 * hidden_dim, 3)
        self.conv3_3_5 = KANConv2DLayer(4 * hidden_dim, 2 * hidden_dim, 3)
        self.conv1_1_1 = KANConv2DLayer(2 * hidden_dim, hidden_dim, 1)

        self.MaxPooling = nn.MaxPool2d(2, 2)
        self.AvgPooling = nn.AvgPool2d(2, 2)
        self.act = nn.ReLU(inplace=True)
        self.BN_1 = nn.BatchNorm2d(hidden_dim)
        self.BN_2 = nn.BatchNorm2d(2 * hidden_dim)
        self.BN_4 = nn.BatchNorm2d(4 * hidden_dim)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))

    def forward(self, x):
        x = self.act(self.BN_1(self.conv3_3_1(self.pad(x))))
        x = self.MaxPooling(x)
        x = self.act(self.BN_2(self.conv3_3_2(self.pad(x))))
        x = self.MaxPooling(x)
        x = self.act(self.BN_4(self.conv3_3_3(self.pad(x))))
        x = self.MaxPooling(x)
        x = self.act(self.BN_4(self.conv3_3_4(self.pad(x))))
        x = self.act(self.BN_2(self.conv3_3_5(self.pad(x))))
        x = self.act(self.BN_1(self.conv1_1_1(x)))

        return x

class Decoder(nn.Module):
    #                     128
    def __init__(self, input_dim, num_layers):
        super(Decoder, self).__init__()
        self.conv3_3_1 = KANConv2DLayer(input_dim, 4 * input_dim, 3)
        self.conv3_3_2 = KANConv2DLayer(4 * input_dim, 4 * input_dim, 3)
        self.conv3_3_3 = KANConv2DLayer(4 * input_dim, 4 * input_dim, 3)
        self.conv3_3_4 = KANConv2DLayer(4 * input_dim, 4 * input_dim, 3)

        # F
        self.conv3_3_5 = KANConv2DLayer(4 * input_dim, 4 * input_dim, 3)
        self.conv3_3_6 = KANConv2DLayer(4 * input_dim, 2 * input_dim, 3)
        self.conv3_3_7 = KANConv2DLayer(2 * input_dim, 512, 3)

        self.act = nn.ReLU(inplace=True)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.BN2 = nn.BatchNorm2d(2 * input_dim)
        self.BN4 = nn.BatchNorm2d(4 * input_dim)
        self.BN8 = nn.BatchNorm2d(8 * input_dim)

    def forward(self, x):
        t1 = self.act(self.BN4(self.conv3_3_1(self.pad(x))))
        y = self.act(self.BN4(self.conv3_3_2(self.pad(t1))))
        t2 = self.act(self.BN4(self.conv3_3_3(self.pad(y))) + t1)
        y = self.act(self.BN4(self.conv3_3_4(self.pad(t2))))
        y = self.act(self.BN4(self.conv3_3_5(self.pad(y))) + t2)
        y = self.act(self.BN2(self.conv3_3_6(self.pad(y))))
        y = self.conv3_3_7(self.pad(y))
        return y
