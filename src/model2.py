import torch
import torch.nn as nn
import torchvision.transforms as transforms


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_normal_(m.weight.data, 0.5)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


in_channels = 3
features = 32


class Model(nn.Module):
    def __init__(self, ngpu):
        super(Model, self).__init__()
        self.ngpu = ngpu
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, 1, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            # 16x16
            nn.Conv2d(features, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # 8x8
            nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            # 4x4
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # 8x8
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            # 16x16
            nn.Conv2d(features * 2, features, 3, 1, 1, bias=False),
            nn.Tanh(),
        )


        self.decoder = nn.Sequential(
            # 1 x 1
            nn.ConvTranspose2d(100, features * 4, 4, 1, 0, bias=False),
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
            nn.ConvTranspose2d(features * 8, features * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 16),
            nn.ReLU(True),
            # 16 x 16
            # nn.ConvTranspose2d(features * 4, features * 4, 3, 1, 1, bias=False),
            # nn.Dropout(0.1),
            # nn.Tanh(),
            # 16 x 16
            # nn.ConvTranspose2d(features * 4, features * 4, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(features * 4),
            # nn.ReLU(True),
            # 16 x 16
            # nn.ConvTranspose2d(features * 4, features * 4, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(features * 4),
            # nn.ReLU(True),
            # 16 x 16
            # nn.ConvTranspose2d(features * 4, features, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(features),
            # nn.ReLU(True),
            # 16 x 16
            nn.ConvTranspose2d(features * 16, features, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

        self.main = nn.Sequential(
            # 16 x 16
            nn.ConvTranspose2d(in_channels + features * 2, features, 5, 1, 2, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            # 16 x 16
            nn.ConvTranspose2d(features, features * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(),
            # 16 x 16
            nn.ConvTranspose2d(features * 2, features * 3, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 3),
            nn.ReLU(True),
            # 32 x 32
            nn.ConvTranspose2d(features * 3, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(),
            # 64 x 64
            nn.ConvTranspose2d(features * 4, features, 3, 1, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            # 64 x 64
            nn.ConvTranspose2d(features, in_channels, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

        self.apply(weights_init)

    def forward(self, input, encoded):
        feats = self.feature_extractor(input)
        decoded = self.decoder(encoded)
        return self.main(torch.cat((input, decoded, feats), dim=1))
