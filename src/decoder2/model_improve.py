import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_normal_(m.weight.data, 0.001)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


features = 16
in_channels = 3
out_channels = 3


class PositionalEncoding(nn.Module):
    def __init__(self, max_time_steps: int, embedding_size: int, n: int = 10000, device="cuda"):
        super().__init__()

        i = torch.arange(embedding_size // 2).to(device)
        k = torch.arange(max_time_steps).unsqueeze(dim=1).to(device)

        self.pos_embeddings = torch.zeros(max_time_steps, embedding_size, requires_grad=False).to(device)
        self.pos_embeddings[:, 0::2] = torch.sin(k / (n ** (2 * i / embedding_size)))
        self.pos_embeddings[:, 1::2] = torch.cos(k / (n ** (2 * i / embedding_size)))

    def forward(self, t):
        return self.pos_embeddings[t, :]


class DownUpModel(nn.Module):
    def __init__(self, ngpu, in_channels, out_channels):
        super(DownUpModel, self).__init__()
        self.ngpu = ngpu
        self.out_channels = out_channels
        self.dropoutp5 = nn.Dropout(p=0.05)

        self.main = nn.Sequential(
            # 64x64
            nn.ConvTranspose2d(in_channels, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.1, inplace=True),

            # 32x32
            nn.Conv2d(features, out_channels, 4, 2, 1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.main2 = nn.Sequential(
            # 64x64
            nn.ConvTranspose2d(in_channels, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.1, inplace=True),

            # 32x32
            nn.ConvTranspose2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.1, inplace=True),

            # 16x16
            nn.Conv2d(features * 2, features, 4, 2, 1, bias=True),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.1, inplace=True),

            # 32x32
            nn.Conv2d(features, out_channels, 4, 2, 1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.mix = nn.Sequential(
             # 64x64
            nn.ConvTranspose2d(in_channels + out_channels, out_channels, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.final = nn.Sequential(
            # 64 x 64
            nn.ConvTranspose2d(out_channels * 2, out_channels * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
            # 64 x 64
            nn.ConvTranspose2d(out_channels * 2, out_channels, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            # 64 x 64
        )


        self.apply(weights_init)

    def self_mix(self, input):
        image, factors = torch.split(input, self.out_channels, dim=1)
        factors = torch.divide(torch.add(factors, 1), 2)
        return torch.sub(torch.ones(image.shape).to(image.get_device()), torch.mul(image, factors))

    def forward(self, input):
        output = torch.cat((self.main(input), self.main2(input)), dim=1)
        output = self.dropoutp5(output)
        output = self.final(output)
        output = self.mix(torch.cat((output, input), dim=1))
        return output


class Model(nn.Module):
    def __init__(self, ngpu):
        super(Model, self).__init__()
        self.ngpu = ngpu

        self.embedding = PositionalEncoding(1000, 3)
        self.mix_embedding = nn.Sequential(
            nn.ConvTranspose2d(3 + features * 16, features * 16, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(features * 16),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.mix_encoded = nn.Sequential(
            nn.ConvTranspose2d(600 + features * 16, features * 32, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(features * 32),
            nn.LeakyReLU(0.1, inplace=True)
        )

        down_up_features = 8
        self.main = DownUpModel(ngpu, in_channels + features, down_up_features)
        self.main2 = DownUpModel(ngpu, down_up_features, down_up_features)
        self.main3 = DownUpModel(ngpu, down_up_features, down_up_features)
        self.main4 = DownUpModel(ngpu, down_up_features, down_up_features)
        self.main5 = DownUpModel(ngpu, down_up_features, down_up_features)
        self.main6 = DownUpModel(ngpu, down_up_features, down_up_features)

        self.encode = nn.Sequential(
            # 64x64
            nn.Conv2d(in_channels, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.1, inplace=True),
            # 32x32
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.1, inplace=True),
            # 16x16
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.1, inplace=True),
            # 8x8
            nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.1, inplace=True),
            # 4x4
            nn.Conv2d(features * 8, features * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 16),
            nn.LeakyReLU(0.1, inplace=True),
            # 1x1
        )

        self.decode = nn.Sequential(
            # 1 x 1
            nn.ConvTranspose2d(features * 32, features * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 16),
            nn.LeakyReLU(0.1, inplace=True),
            # 4 x 4
            nn.ConvTranspose2d(features * 16, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.1, inplace=True),
            # 8 x 8
            nn.ConvTranspose2d(features * 4, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.1, inplace=True),
            # 16 x 16
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.1, inplace=True),
            # 32 x 32
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.1, inplace=True),
            # 64 x 64
        )

        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(in_channels + down_up_features + features, features, 3, 1, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(features, out_channels, 3, 1, 1, bias=False),
        )
        
        self.attention = nn.MultiheadAttention(
            features, 1, batch_first=True
        )

        self.dropoutp5 = nn.Dropout(p=0.05)
        self.dropoutp10 = nn.Dropout(p=0.1)

        self.apply(weights_init)

    def forward(self, input, encoded, step):
        batch_size = input.shape[0]
        e = self.encode(input)
        pe = self.embedding(torch.tensor([step] * batch_size, dtype=torch.long)).reshape((batch_size, -1, 1, 1))
        e = self.mix_embedding(torch.cat((e, pe), dim=1))
        e = torch.cat((e, encoded), dim=1)
        e = self.mix_encoded(e)
        e = e.reshape((batch_size, -1, features))
        e, _ = self.attention(e, e, e, need_weights=False)
        e = e.reshape((batch_size, -1, 1, 1))
        e = self.dropoutp10(e)
        d = self.decode(e)
        d = self.dropoutp5(d)
        output = self.main(torch.cat((input, d), dim=1))
        output = self.main2(output)
        output = self.main3(output)
        output = self.main4(output)
        output = self.main5(output)
        output = self.main6(output)
        output = self.dropoutp10(output)
        output =  self.final(torch.cat((input, output, d), dim=1))
        # image, factors = torch.split(output, 3, dim=1)
        # factors = torch.divide(torch.add(factors, 1.0002), 2.0002)
        # factors = torch.divide(torch.add(factors, 1.02), 2.02)
        # inv_factors = torch.sub(torch.ones(factors.shape).to(input.get_device()), factors)
        return torch.tanh(torch.add(torch.atanh(torch.divide(input, 1 + 1e-6)), output))
        # return output
