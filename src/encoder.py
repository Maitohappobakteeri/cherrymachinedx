import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_normal_(m.weight.data, 0.5)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


features = 64
in_channels = 3


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, 1, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            # 16x16
            nn.Conv2d(features, features, 3, 1, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            # 16x16
            nn.Conv2d(features, features, 3, 1, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            # 16x16
            nn.Conv2d(features, features * 2, 3, 1, 1, bias=False),
            nn.Dropout(0.05),
            nn.Tanh(),
            # 16x16
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # 8x8
            nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            # 4x4
            nn.Conv2d(features * 8, features * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            # 4x4
            nn.Conv2d(features * 8, 100, 4, 1, 0, bias=False),
            nn.Tanh(),
        )

        self.apply(weights_init)

    def forward(self, input):
        return self.main(input)
