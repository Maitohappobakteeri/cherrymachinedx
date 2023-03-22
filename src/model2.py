from __future__ import print_function

# %matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os.path


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_normal_(m.weight.data, 0.5)
        # nn.init.normal_(m.weight.data, 1.0, 0.02)
        # torch.randn(m.weight.data.shape, out=m.weight.data)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("Linear") != -1:
        # nn.init.xavier_normal_(m.weight.data, 10.0)
        # nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.xavier_uniform_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0)


ngf = 32
nz = 100
nc = 3
features = 32


class Model(nn.Module):
    def __init__(self, ngpu):
        super(Model, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 16 x 16
            nn.ConvTranspose2d(3, features, 5, 1, 2, bias=False),
            # nn.BatchNorm2d(features),
            nn.Dropout(0.05),
            nn.Tanh(),
            # 16 x 16
            nn.ConvTranspose2d(features, features * 2, 3, 1, 1, bias=False),
            nn.Dropout(0.05),
            # nn.BatchNorm2d(features * 2),
            nn.Tanh(),
            # 16 x 16
            nn.ConvTranspose2d(features * 2, features * 3, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 3),
            nn.ReLU(True),
            # 32 x 32
            nn.ConvTranspose2d(features * 3, features * 4, 4, 2, 1, bias=False),
            nn.Dropout(0.05),
            nn.Tanh(),
            # 64 x 64
            nn.ConvTranspose2d(features * 4, features, 3, 1, 1, bias=False),
            nn.Dropout(0.05),
            nn.Tanh(),
            # 128 x 128
            nn.ConvTranspose2d(features, nc, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

        std = 0.1
        self.normalize = transforms.Normalize((0.0, 0.0, 0.0), (std, std, std))

        self.apply(weights_init)

    def forward(self, input):
        # input = self.normalize(input)
        return self.main(input)
