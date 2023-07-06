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

        self.decoder = nn.Sequential(
            nn.Linear(in_channels, features * 4),
            nn.LayerNorm(features * 4),
            nn.ReLU(True),

            nn.Linear(features * 4, features * 4),
            nn.LayerNorm(features * 4),
            nn.ReLU(True),

            nn.Linear(features * 4, features * 8),
            nn.LayerNorm(features * 8),
            nn.ReLU(True),
        )

        self.step1 = nn.Sequential(
            # 1 x 1
            nn.ConvTranspose2d(features * 8, features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            # 4 x 4
        )

        self.step1_attention = nn.MultiheadAttention(
            features * 8, 4, batch_first=True
        )

        self.main_to_16x16 = nn.Sequential(
            # 4 x 4 redistribute after attention
            nn.ConvTranspose2d(features * 8, features * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),

            # 4 x 4 increase resolution
            nn.ConvTranspose2d(features * 8, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            # 8 x 8
            nn.ConvTranspose2d(features * 8, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            # 16 x 16
        )

        self.main_to_64x64 = nn.Sequential(
            # 16 x 16
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # 32 x 32
            nn.ConvTranspose2d(features * 4, features, 4, 2, 1, bias=False),
            nn.Tanh(),
            # 64 x 64
        )

        self.decoder_image = nn.Sequential(
            # 1 x 1
            nn.ConvTranspose2d(features * 8, features * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # 4 x 4
            nn.ConvTranspose2d(features * 4, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # 8 x 8
            nn.ConvTranspose2d(features * 4, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # 16 x 16
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            # 32 x 32
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
            nn.Tanh(),
            # 64 x 64
        )

        self.small_decoder_image = nn.Sequential(
            # 1 x 1
            nn.ConvTranspose2d(features * 8, features * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # 4 x 4
            nn.ConvTranspose2d(features * 4, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # 8 x 8
            nn.ConvTranspose2d(features * 4, features, 4, 2, 1, bias=False),
            nn.Tanh(),
            # 16 x 16
        )

        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 3, features * 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(features * 2, features * 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(features * 4, out_channels, 1, 1, 0, bias=False),
            nn.Tanh(),
        )

        self.decoder_image_out = nn.Sequential(
            # 64 x 64
            nn.ConvTranspose2d(features, features * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            # 64 x 64
            nn.ConvTranspose2d(features * 2, out_channels, 3, 1, 1, bias=False),
            nn.Tanh(),
            # 64 x 64
        )

        self.small_decoder_image_out = nn.Sequential(
            # 64 x 64
            nn.ConvTranspose2d(features, features * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            # 64 x 64
            nn.ConvTranspose2d(features * 2, out_channels, 3, 1, 1, bias=False),
            nn.Tanh(),
            # 64 x 64
        )

        self.pre_attention1 = nn.ConvTranspose2d(features * 8, features, 1, 1, 0, bias=False)
        self.pre_attention2 = nn.ConvTranspose2d(features, features, 1, 1, 0, bias=False)
        self.pre_attention3 = nn.ConvTranspose2d(features, features, 1, 1, 0, bias=False)
        self.pre_attention4 = nn.ConvTranspose2d(features, features, 1, 1, 0, bias=False)

        self.mid_attention1 = nn.ConvTranspose2d(features, features * 8, 1, 1, 0, bias=False)
        self.mid_attention2 = nn.ConvTranspose2d(features, features, 1, 1, 0, bias=False)

        self.after_attention1 = nn.ConvTranspose2d(features * 8, features * 8, 3, 1, 1, bias=False)
        self.after_attention2 = nn.ConvTranspose2d(features, features, 3, 1, 1, bias=False)

        self.apply(weights_init)

    def forward(self, input):
        batch_size = input.shape[0]
        decoded = self.decoder(input.view((batch_size, -1)))

        output = self.step1(
            decoded.view((batch_size, -1, 1, 1))
        ).view((batch_size, -1, features * 8))
        (output, _) = self.step1_attention(output, output, output, need_weights=False)

        output_x16 = self.main_to_16x16(output.reshape((batch_size, -1, 4, 4)))
        small_decoder_output = self.small_decoder_image(decoded.view((batch_size, -1, 1, 1)))
        mul1 = nn.functional.softmax(torch.mul(self.pre_attention1(output_x16), self.pre_attention2(small_decoder_output)), dim=1)
        output_x16 = self.after_attention1(torch.mul(output_x16, self.mid_attention1(mul1)))

        original_output_x64 = output_x64 = self.main_to_64x64(output_x16)
        decoder_output = self.decoder_image(decoded.view((batch_size, -1, 1, 1)))
        mul2 = nn.functional.softmax(torch.mul(self.pre_attention3(output_x64), self.pre_attention4(decoder_output)), dim=1)
        output_x64 = self.after_attention2(torch.mul(output_x64, self.mid_attention2(mul2)))

        return self.decoder_image_out(decoder_output), self.small_decoder_image_out(small_decoder_output), self.final(torch.cat((original_output_x64, output_x64, decoder_output), dim=1))
