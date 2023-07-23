import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight.data, 1.0)
    elif classname.find("Conv") != -1:
        nn.init.xavier_normal_(m.weight.data, 1.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


features = 8
in_channels = 3
out_channels = 3
max_steps = 100


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
    def __init__(self, ngpu, in_channels, out_channels, embedding):
        super(DownUpModel, self).__init__()
        self.ngpu = ngpu
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding = embedding

        self.attention = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Softmax(dim=1),
            nn.Linear(in_channels, in_channels)
        )

        self.attention2 = nn.Sequential(
            nn.Linear(features, features),
            nn.Softmax(dim=1),
            nn.Linear(features, features)
        )

        self.attention3 = nn.Sequential(
            nn.Linear(features, features),
            nn.Softmax(dim=1),
            nn.Linear(features, features)
        )

        self.main = nn.Sequential(
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
            nn.Dropout(p=0.1),
        )

        self.main2 = nn.Sequential(
            # 1 x 1
            nn.ConvTranspose2d(features * 16 + 10, features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.1, inplace=True),
            # 4 x 4
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.1, inplace=True),
            # 8 x 8
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.1, inplace=True),
            # 16 x 16
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.1, inplace=True),
            # 32 x 32
            nn.ConvTranspose2d(features, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.1, inplace=True),
            # 64 x 64
        )

        self.small_main = nn.Sequential(
            # 16x16
            nn.Conv2d(in_channels + features, features * 4, 4, 2, 1, bias=False),
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
            nn.Dropout(p=0.1),
        )

        self.small_main2 = nn.Sequential(
            # 1 x 1
            nn.ConvTranspose2d(features * 16 + 10, features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.1, inplace=True),
            # 4 x 4
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.1, inplace=True),
            # 8 x 8
            nn.ConvTranspose2d(features * 4, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # 16 x 16
        )

        self.medium_main = nn.Sequential(
            # 32x32
            nn.Conv2d(in_channels + features, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.1, inplace=True),
            # 16x16
            nn.Conv2d(features, features * 4, 4, 2, 1, bias=False),
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
            nn.Dropout(p=0.1),
        )

        self.medium_main2 = nn.Sequential(
            # 1 x 1
            nn.ConvTranspose2d(features * 16 + 10, features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.1, inplace=True),
            # 4 x 4
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.1, inplace=True),
            # 8 x 8
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.1, inplace=True),
            # 16 x 16
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.1, inplace=True),
            # 32 x 32
        )

        self.apply(weights_init)

    def get_patches(self, t, patch_size, channels):
        t = torch.nn.functional.unfold(t, patch_size, stride=patch_size)
        return t.permute((0, 2, 1)).reshape((-1, channels, patch_size, patch_size))

    def combine_patches(self, t, batch_size, image_size, patch_size, channels):
        t = t.reshape((batch_size, -1, patch_size ** 2 * channels)).permute((0, 2, 1))
        return torch.nn.functional.fold(t, (image_size, image_size), patch_size, stride=patch_size)

    def small(self, input, step):
        batch_size = input.shape[0]
        in_channels = input.shape[1]
        image_size = 64
        patch_size = 16
        patches_per_image = (image_size // patch_size) ** 2
        patches = self.get_patches(input, patch_size, in_channels)
        patches = self.small_main(patches)
        pe_patches = self.embedding(torch.tensor([step] * batch_size * patches_per_image, dtype=torch.long)).reshape((batch_size * patches_per_image, -1, 1, 1))
        patches = torch.cat((patches, pe_patches), dim=1)
        patches = self.small_main2(patches)
        return self.combine_patches(patches, batch_size, image_size, patch_size, self.out_channels)

    def medium(self, input, step):
        batch_size = input.shape[0]
        in_channels = input.shape[1]
        image_size = 64
        patch_size = 32
        patches_per_image = (image_size // patch_size) ** 2
        patches = self.get_patches(input, patch_size, in_channels)
        patches = self.medium_main(patches)
        pe_patches = self.embedding(torch.tensor([step] * batch_size * patches_per_image, dtype=torch.long)).reshape((batch_size * patches_per_image, -1, 1, 1))
        patches = torch.cat((patches, pe_patches), dim=1)
        patches = self.medium_main2(patches)
        return self.combine_patches(patches, batch_size, image_size, patch_size, features)

    def forward(self, input, prev_output, step):
        batch_size = input.shape[0]

        attention_weight = self.attention(input.reshape((-1, self.in_channels))).reshape(input.shape)
        input = input * attention_weight

        output = self.main(input)
        pe = self.embedding(torch.tensor([step] * batch_size, dtype=torch.long)).reshape((batch_size, -1, 1, 1))
        output = torch.cat((output, pe), dim=1)
        output = self.main2(output)
        
        attention_weight = self.attention2(output.reshape((-1, features))).reshape(output.shape)
        output = output * attention_weight

        folded2 = self.medium(torch.cat((input, output), dim=1), step)

        attention_weight = self.attention3(folded2.reshape((-1, features))).reshape(folded2.shape)
        folded2 = folded2 * attention_weight

        folded = self.small(torch.cat((input, folded2), dim=1), step)

        return prev_output + folded


class Model(nn.Module):
    def __init__(self, ngpu):
        super(Model, self).__init__()
        self.ngpu = ngpu

        self.embedding = PositionalEncoding(max_steps, 10)

        down_up_features = features
        self.main = DownUpModel(ngpu, in_channels, down_up_features, self.embedding)
        self.main_list = nn.ModuleList([
          DownUpModel(ngpu, in_channels + down_up_features, down_up_features, self.embedding) for x in range(7)  
        ])

        self.final = nn.Sequential(
            nn.ConvTranspose2d(down_up_features, features, 3, 1, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.ConvTranspose2d(features, out_channels, 1, 1, 0, bias=False),
            nn.Tanh(),
        )

        self.scale = nn.Linear(1, 1)

        self.apply(weights_init)

    def forward(self, input, step, returnDiff=False):
        output = self.main(input, torch.zeros((input.shape[0], features, *input.shape[2:]), device="cuda"), step)
        for m in self.main_list:
            output = m(torch.cat((input, output), dim=1), output, step)
        output =  self.final(output)
        # return torch.tanh(output)
        # return torch.tanh(torch.atanh(torch.clamp(input, -1.0 + 1e-12, 1.0 - 1e-12)) + output)
        diff = self.scale(output.reshape(-1, 1)).reshape(input.shape)
        if returnDiff:
            return input + diff, diff 
        return input + diff

