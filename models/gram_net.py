import torch
import torch.nn as nn


def gram_matrix(in_feature):
    batch, channel, height, width = in_feature.size()
    features = in_feature.view(batch, channel, height * width)
    gram = torch.bmm(features, torch.transpose(features, 1, 2))
    return gram.view(batch, 1, channel, channel)


class GramBlock(nn.Module):
    def __init__(self, in_channel):
        super(GramBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = gram_matrix(x1)
        x3 = self.conv2(x2)
        out = self.gap(x3)
        out = torch.flatten(out, 1)
        return out

