import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_normal_(m.weight.data, 0.5)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


features = 32
in_channels = 3


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, 1, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x64
            nn.Conv2d(features, features * 2, 5, 1, 2, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x64
            nn.Conv2d(features * 2, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32
            nn.Conv2d(features * 2, features * 3, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 3),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16
            nn.Conv2d(features * 3, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8
            nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8
            nn.Conv2d(features * 8, features * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4
            nn.Conv2d(features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

        self.apply(weights_init)

    def forward(self, input):
        return self.main(input)
