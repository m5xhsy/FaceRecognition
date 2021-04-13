import os

import torch
from torch import nn
from torch.nn import functional as F


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        )  # verify bias false
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001,  # value found in tensorflow
            momentum=0.1,  # default pytorch value
            affine=True
        )
        self.relu = nn.ReLU(inplace=False)
        self.l2  = nn.MSELoss
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()
        self.conv1a = BasicConv2d(in_planes=3, out_planes=32, kernel_size=3, stride=2)
        self.conv2a = BasicConv2d(in_planes=32, out_planes=32, kernel_size=3, stride=1)
        self.conv3a = BasicConv2d(in_planes=32, out_planes=64, kernel_size=3, stride=1, padding=1)
        self.conv4a = BasicConv2d(in_planes=64, out_planes=96, kernel_size=3, stride=2)
        self.maxp4b = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv5a = nn.Sequential(
            BasicConv2d(in_planes=160, out_planes=64, kernel_size=1, stride=1, padding=3),
            BasicConv2d(in_planes=64, out_planes=64, kernel_size=(7, 1), stride=1),
            BasicConv2d(in_planes=64, out_planes=64, kernel_size=(1, 7), stride=1),
            BasicConv2d(in_planes=64, out_planes=96, kernel_size=3, stride=1)
        )
        self.conv5b = nn.Sequential(
            BasicConv2d(in_planes=160, out_planes=64, kernel_size=1, stride=1),
            BasicConv2d(in_planes=64, out_planes=96, kernel_size=3, stride=1)
        )
        self.conv6a = BasicConv2d(in_planes=192, out_planes=192, kernel_size=3, stride=2)
        self.maxp6b = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        out = self.conv1a(x)
        out = self.conv2a(out)
        out = self.conv3a(out)
        out4a = self.conv4a(out)
        out4b = self.maxp4b(out)
        out = torch.cat((out4b, out4a), 1)
        out5a = self.conv5a(out)
        out5b = self.conv5b(out)
        out = torch.cat((out5a, out5b), 1)
        out6a = self.conv6a(out)
        out6b = self.maxp6b(out)
        out = torch.cat((out6a, out6b), 1)
        return out


class InceptionResnetA(nn.Module):
    def __init__(self, scale=1.0):
        self.scale = scale
        super(InceptionResnetA, self).__init__()
        self.conv1a = BasicConv2d(in_planes=384, out_planes=32, kernel_size=1, stride=1)
        self.conv1b = nn.Sequential(
            BasicConv2d(in_planes=384, out_planes=32, kernel_size=1, stride=1, padding=1),
            BasicConv2d(in_planes=32, out_planes=32, kernel_size=3, stride=1)
        )
        self.conv1c = nn.Sequential(
            BasicConv2d(in_planes=384, out_planes=32, kernel_size=1, stride=1, padding=2),
            BasicConv2d(in_planes=32, out_planes=48, kernel_size=3, stride=1),
            BasicConv2d(in_planes=48, out_planes=64, kernel_size=3, stride=1)
        )
        self.conv2a = BasicConv2d(in_planes=128, out_planes=384, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out1a = self.conv1a(x)
        out1b = self.conv1b(x)
        out1c = self.conv1c(x)
        out = torch.cat((out1a, out1b, out1c), 1)
        out = self.conv2a(out)
        # print(out1a.size(),out1b.size(),out1c.size())
        out = x + out * self.scale
        out = self.relu(out)
        return out


class ReductionA(nn.Module):
    def __init__(self):
        super(ReductionA, self).__init__()
        self.conv1a = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv1b = BasicConv2d(in_planes=384, out_planes=256, kernel_size=3, stride=2)
        self.conv1c = nn.Sequential(
            BasicConv2d(in_planes=384, out_planes=192, kernel_size=1, stride=1, padding=1),
            BasicConv2d(in_planes=192, out_planes=192, kernel_size=3, stride=1),
            BasicConv2d(in_planes=192, out_planes=256, kernel_size=3, stride=2)
        )

    def forward(self, x):
        out1a = self.conv1a(x)
        out1b = self.conv1b(x)
        out1c = self.conv1c(x)
        out = torch.cat((out1a, out1b, out1c), 1)
        return out


class InceptionResnetB(nn.Module):
    def __init__(self, scale=1.0):
        self.scale = scale
        super(InceptionResnetB, self).__init__()
        self.conv1a = BasicConv2d(in_planes=896, out_planes=192, kernel_size=1, stride=1)
        self.conv1b = nn.Sequential(
            BasicConv2d(in_planes=896, out_planes=128, kernel_size=1, stride=1, padding=3),
            BasicConv2d(in_planes=128, out_planes=160, kernel_size=(1, 7), stride=1),
            BasicConv2d(in_planes=160, out_planes=192, kernel_size=(7, 1), stride=1)
        )
        self.conv2a = BasicConv2d(in_planes=384, out_planes=896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out1a = self.conv1a(x)
        out1b = self.conv1b(x)
        out = torch.cat((out1a, out1b), 1)
        out = self.conv2a(out)
        out = x + out * self.scale
        out = self.relu(out)
        return out


class ReductionB(nn.Module):
    def __init__(self):
        super(ReductionB, self).__init__()
        self.conv1a = nn.Sequential(
            BasicConv2d(in_planes=896, out_planes=256, kernel_size=1, stride=1, padding=1),
            BasicConv2d(in_planes=256, out_planes=288, kernel_size=3, stride=1),
            BasicConv2d(in_planes=288, out_planes=256, kernel_size=3, stride=2)
        )
        self.conv1b = nn.Sequential(
            BasicConv2d(in_planes=896, out_planes=256, kernel_size=1, stride=1),
            BasicConv2d(in_planes=256, out_planes=256, kernel_size=3, stride=2)
        )
        self.conv1c = nn.Sequential(
            BasicConv2d(in_planes=896, out_planes=256, kernel_size=1, stride=1),
            BasicConv2d(in_planes=256, out_planes=384, kernel_size=3, stride=2)
        )
        self.maxp1d = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        out1a = self.conv1a(x)
        out1b = self.conv1b(x)
        out1c = self.conv1c(x)
        out1d = self.maxp1d(x)
        out = torch.cat((out1a, out1b, out1c, out1d), 1)
        return out


class InceptionResnetC(nn.Module):
    def __init__(self, scale=1.0, noReLU=False):
        super(InceptionResnetC, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.conv1a = BasicConv2d(in_planes=1792, out_planes=192, kernel_size=1, stride=1)
        self.conv1b = nn.Sequential(
            BasicConv2d(in_planes=1792, out_planes=192, kernel_size=1, stride=1, padding=1),
            BasicConv2d(in_planes=192, out_planes=224, kernel_size=(1, 3), stride=1),
            BasicConv2d(in_planes=224, out_planes=256, kernel_size=(3, 1), stride=1)
        )
        self.conv2a = BasicConv2d(in_planes=448, out_planes=1792, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out1a = self.conv1a(x)
        out1b = self.conv1b(x)
        out = torch.cat((out1a, out1b), 1)
        out = self.conv2a(out)
        out = x + out * self.scale

        if not self.noReLU:
            out = self.relu(out)

        return out


class InceptionResnetV2(nn.Module):
    def __init__(self, classify=False, num_classes=None, dropout_prob=0.5):
        super(InceptionResnetV2, self).__init__()
        self.classify = classify
        self.num_classes = num_classes

        self.stem = Stem()
        self.ira_x5 = nn.Sequential(
            InceptionResnetA(scale=0.17),
            InceptionResnetA(scale=0.17),
            InceptionResnetA(scale=0.17),
            InceptionResnetA(scale=0.17),
            InceptionResnetA(scale=0.17),
        )
        self.ra = ReductionA()
        self.irb_x10 = nn.Sequential(
            InceptionResnetB(scale=0.1),
            InceptionResnetB(scale=0.1),
            InceptionResnetB(scale=0.1),
            InceptionResnetB(scale=0.1),
            InceptionResnetB(scale=0.1),
            InceptionResnetB(scale=0.1),
            InceptionResnetB(scale=0.1),
            InceptionResnetB(scale=0.1),
            InceptionResnetB(scale=0.1),
            InceptionResnetB(scale=0.1),
        )
        self.rb = ReductionB()

        self.irc_x5 = nn.Sequential(
            InceptionResnetC(scale=0.2),
            InceptionResnetC(scale=0.2),
            InceptionResnetC(scale=0.2),
            InceptionResnetC(scale=0.2),
            InceptionResnetC(scale=0.2)
        )
        self.irc_noRelu = InceptionResnetC(noReLU=True)
        self.avgp = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(1792, 512, bias=False)
        self.bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)
        # print(out1a.size(), out1b.size())
        if self.classify and self.num_classes is not None:
            self.logits = nn.Linear(512, self.num_classes)

    def forward(self, x):
        out = self.stem(x)
        out = self.ira_x5(out)
        out = self.ra(out)
        out = self.irb_x10(out)
        out = self.rb(out)
        out = self.irc_x5(out)
        out = self.irc_noRelu(out)
        out = self.avgp(out)
        out = self.dropout(out)
        out = self.linear(out.view(x.shape[0], -1))
        out = self.bn(out)
        # print(out.size())
        if self.classify:
            out = self.logits(out)
        else:
            out = F.normalize(out, p=2, dim=1)
        return out


if __name__ == "__main__":
    input = torch.rand(1, 3, 299, 299)
    # input = torch.rand(1, 3, 160, 160)
    net = InceptionResnetV2(classify=True, num_classes=3).eval()
    net(input)


