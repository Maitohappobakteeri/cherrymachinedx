import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

from database_memory import DatabaseMemory
from database_image_memory import DatabaseImageMemory

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_normal_(m.weight.data, 1.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


features = 16
in_channels = 500 + 100
out_channels = 3


class Model(nn.Module):
    def __init__(self, ngpu):
        super(Model, self).__init__()
        self.ngpu = ngpu

        self.decoder = nn.Sequential(
            nn.Linear(in_channels, features * 16),
            nn.LayerNorm(features * 16),
            nn.ReLU(True),

            nn.Linear(features * 16, features * 32),
            nn.LayerNorm(features * 32),
            nn.ReLU(True),

            nn.Linear(features * 32, features * 32),
            nn.LayerNorm(features * 32),
            nn.ReLU(True),
        )

        self.attention = nn.MultiheadAttention(
            features, 1, batch_first=True
        )

        self.decoder_image = nn.Sequential(
            # 1 x 1
            nn.ConvTranspose2d(features * 32, features * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 16),
            nn.ReLU(True),
            # 4 x 4
            nn.ConvTranspose2d(features * 16, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            # 8 x 8
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # 16 x 16
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            # 32 x 32
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # 64 x 64
        )

        self.decoder_image2 = nn.Sequential(
            # 1 x 1
            nn.ConvTranspose2d(features * 32, features * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 32),
            nn.ReLU(True),
            # 4 x 4
            nn.ConvTranspose2d(features * 32, features * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 16),
            nn.ReLU(True),
            # 8 x 8
            nn.ConvTranspose2d(features * 16, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            # 16 x 16
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # 32 x 32
            nn.ConvTranspose2d(features * 4, features, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # 64 x 64
        )

        self.decoder_image_out = nn.Sequential(
            nn.Dropout(p=0.1),
            # 64 x 64
            nn.ConvTranspose2d(features * 4, features * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            # 64 x 64
            nn.ConvTranspose2d(features * 2, out_channels * 2, 3, 1, 1, bias=True),
            nn.Tanh(),
            # 64 x 64
        )

        self.decoder_features = nn.Sequential(
            # 1 x 1
            nn.ConvTranspose2d(features * 32, features * 4, 8, 1, 0, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # 8 x 8
            nn.ConvTranspose2d(features * 4, features * 2, 6, 4, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            # 32 x 32
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # 64 x 64
        )

        self.fastest = nn.Sequential(
            # 1 x 1
            nn.ConvTranspose2d(in_channels, features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            # 1 x 1
            nn.ConvTranspose2d(features, features * 2, 8, 1, 0, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            # 8 x 8
            nn.ConvTranspose2d(features * 2, features * 4, 6, 4, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # 32 x 32
            nn.ConvTranspose2d(features * 4, features, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # 64 x 64
        )

        self.fastest_to_16x16 = nn.Sequential(
            # 64x64
            nn.Conv2d(features, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            # 32x32
            nn.Conv2d(features, out_channels * 2, 4, 2, 1, bias=True),
            nn.Tanh(),
            # 16x16
        )

        self.decoder_image_simple = nn.Sequential(
            # 1 x 1
            nn.ConvTranspose2d(features * 32, features * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 16),
            nn.ReLU(True),
            # 4 x 4
            nn.ConvTranspose2d(features * 16, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            # 8 x 8
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # 16 x 16
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            # 32 x 32
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # 64 x 64
        )

        self.decoder_image_out_simple = nn.Sequential(
            # 64 x 64
            nn.ConvTranspose2d(features, features * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            # 64 x 64
            nn.ConvTranspose2d(features * 2, out_channels, 3, 1, 1, bias=True),
            nn.Tanh(),
            # 64 x 64
        )

        self.decoder_image_out = nn.Sequential(
            # 64 x 64
            nn.ConvTranspose2d(features * 5, features * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            # 64 x 64
            nn.ConvTranspose2d(features * 2, out_channels * 2, 3, 1, 1, bias=True),
            nn.Tanh(),
            # 64 x 64
        )

        self.final = nn.Sequential(
            # 64 x 64
            nn.ConvTranspose2d(out_channels, features, 3, 1, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            # 64 x 64
            nn.ConvTranspose2d(features, out_channels, 3, 1, 1, bias=True),
            nn.Tanh(),
            # 64 x 64
        )

        self.memory = DatabaseMemory(ngpu, features * 32, 3)
        self.memory_image = DatabaseImageMemory(ngpu, features, 3)

        self.dropoutp5 = nn.Dropout(p=0.05)
        self.dropoutp10 = nn.Dropout(p=0.1)
        self.dropoutp20 = nn.Dropout(p=0.2)
        self.dropoutp30 = nn.Dropout(p=0.3)
        self.dropoutp50 = nn.Dropout(p=0.5)

        self.apply(weights_init)

    def mix(self, input):
        image, factors = torch.split(input, 3, dim=1)
        factors = torch.divide(torch.add(factors, 1), 2)
        return torch.sub(torch.ones(image.shape).to(image.get_device()), torch.mul(image, factors))

    def forward(self, input, outputx16=False):
        batch_size = input.shape[0]
        input = self.dropoutp10(input)
        decoded = self.decoder(input.view((batch_size, -1)))
        output = decoded.reshape((batch_size, -1, features))
        (output, _) = self.attention(output, output, output, need_weights=False)
        output = self.memory(output.reshape((batch_size, features * 32)))
        output = output.reshape((batch_size, -1, features * 8))
        output = og = self.dropoutp5(output)
        output = self.decoder_image(og.view((batch_size, -1, 1, 1)))
        output = self.memory_image(output)
        output2 = self.decoder_image2(og.view((batch_size, -1, 1, 1)))

        output_simple = self.decoder_image_simple(og.view((batch_size, -1, 1, 1)))

        f = self.fastest(input)
        decoder_features = self.decoder_features(decoded.reshape((batch_size, -1, 1, 1)))
        output = torch.cat((output, output2, decoder_features, f, output_simple), dim=1)
        output = self.dropoutp10(output)
        output = self.mix(self.decoder_image_out(output))
        output = self.final(output)

        if outputx16:
            output_x16 = self.mix(self.fastest_to_16x16(f))
            output_simple = self.decoder_image_out_simple(output_simple)
            return output, output_x16, output_simple
        return output
