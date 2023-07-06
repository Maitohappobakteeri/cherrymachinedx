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


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 64x64
            nn.Conv2d(in_channels, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8
            nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4
        )

        self.attention = nn.MultiheadAttention(
            features * 8, 4, batch_first=True
        )

        self.final = nn.Sequential(
            nn.Conv2d(features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

        self.apply(weights_init)

    def forward(self, input):
        batch_size = input.shape[0]
        output = self.main(input).reshape(batch_size, -1, features * 8)
        (output, _) = self.attention(output, output, output, need_weights=False)
        return self.final(output.reshape((batch_size, -1, 4, 4)))
