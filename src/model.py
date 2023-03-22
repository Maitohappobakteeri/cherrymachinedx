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


features = 32
in_channels = 100
out_channels = 3


class Model(nn.Module):
    def __init__(self, ngpu):
        super(Model, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 1 x 1
            nn.ConvTranspose2d(in_channels, features * 4, 4, 1, 0, bias=False),
            nn.GroupNorm(4, features * 4),
            nn.ReLU(True),
            # 4 x 4
            nn.ConvTranspose2d(features * 4, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # 8 x 8
            nn.ConvTranspose2d(features * 4, features * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            # 8 x 8
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # 16 x 16
            nn.ConvTranspose2d(features * 4, features * 4, 3, 1, 1, bias=False),
            nn.Dropout(0.1),
            nn.Tanh(),
            # 16 x 16
            nn.ConvTranspose2d(features * 4, features * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # 16 x 16
            nn.ConvTranspose2d(features * 4, features * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # 16 x 16
            nn.ConvTranspose2d(features * 4, features, 3, 1, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            # 16 x 16
            nn.ConvTranspose2d(features, out_channels, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

        self.apply(weights_init)

    def forward(self, input):
        return self.main(input)
