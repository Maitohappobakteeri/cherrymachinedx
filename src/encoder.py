import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_normal_(m.weight.data, 1.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


features = 16
in_channels = 3
out_channels = 500


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            # 64x64
            nn.Conv2d(in_channels, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            # 32x32
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            # 16x16
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # 8x8
            nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            # 4x4
        )

        self.main2 = nn.Sequential(
            # 64x64
            nn.Conv2d(in_channels, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            # 32x32
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # 16x16
            nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            # 8x8
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            # 4x4
        )

        self.fastest = nn.Sequential(
            # 64x64
            nn.Conv2d(in_channels, features * 2, 6, 4, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            # 16x16
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # 8x8
            nn.Conv2d(features * 4, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # 4x4
        )

        self.attention = nn.MultiheadAttention(
            features * 8, 4, batch_first=True
        )

        self.final = nn.Sequential(
            nn.Conv2d(features * 20, features * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 32),
            nn.ReLU(True),

            nn.Conv2d(features * 32, out_channels, 1, 1, 0, bias=False),
            nn.Tanh(),
        )

        self.final_mean = nn.Linear(out_channels, out_channels)
        self.final_var = nn.Linear(out_channels, out_channels)

        self.apply(weights_init)

    def forward(self, input):
        batch_size = input.shape[0]
        output = self.main(input).reshape(batch_size, -1, features * 8)
        (output, _) = self.attention(output, output, output, need_weights=False)
        output2 = self.main2(input)
        f = self.fastest(input)
        output = self.final(torch.cat((output.reshape((batch_size, -1, 4, 4)), output2, f), dim=1)).view(batch_size, -1)
        return self.final_mean(output).view(batch_size, -1, 1, 1), self.final_var(output).view(batch_size, -1, 1, 1)
        # return self.final_mean(output).view(batch_size, -1, 1, 1), self.final_var(output).view(batch_size, -1, 1, 1)

    def reparametrize(self, mean, var):
        noise = torch.randn(mean.shape[0], 100, 1, 1).to(mean.get_device())
        std = torch.exp(var * 0.5)
        eps = torch.randn_like(std)
        # return mean + eps * std
        return torch.cat((mean + eps * std, noise), dim=1)