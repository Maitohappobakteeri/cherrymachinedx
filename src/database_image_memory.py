import torch
from torch import nn
import numpy as np
import os

# import sqlite3
# con = sqlite3.connect("../trained.db")

import log

max_batches = 8
features = 16
slots = 1000
image_size = 64
channels = 3

memory = np.zeros((
    max_batches,
    slots,
    channels,
    image_size,
    image_size
))

update_memory = np.zeros((
    max_batches,
    slots,
    channels,
    image_size,
    image_size
))

def update():
    global memory, update_memory, short_memory
    memory = np.copy(update_memory)

if os.path.isfile("./image_memory"):
    memory = np.load("./image_memory", allow_pickle=True)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_uniform_(m.weight.data, 0.1)
    elif classname.find("Linear") != -1:
        nn.init.xavier_uniform_(m.weight.data, 0.1)
    elif classname.find("LSTM") != -1:
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param.data, 1.0)
            elif "weight_hh" in name:
                torch.nn.init.xavier_uniform_(param.data, 1.0)
            elif "bias" in name:
                param.data.fill_(0)

class FastAttention(nn.Module):
    def __init__(self, config, in_channels):
        super(FastAttention, self).__init__()
        self.config = config

        self.main = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Softmax(dim=1),
            nn.Linear(in_channels, in_channels)
        )

        self.apply(weights_init)

    def forward(self, inputs):
        return inputs * self.main(inputs)
    
class FastAttentionImage(nn.Module):
    def __init__(self, config, in_channels):
        super(FastAttentionImage, self).__init__()
        self.config = config

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.Softmax(dim=2),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
        )

        self.apply(weights_init)

    def forward(self, inputs):
        return inputs * self.main(inputs)
    

class MemoryAccess(nn.Module):
    def __init__(self, config, in_channels, available_slots, read_slots=3):
        super(MemoryAccess, self).__init__()
        self.config = config
        self.read_slots = read_slots
        self.available_slots = available_slots

        self.encode = nn.Sequential(
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
            nn.Conv2d(features * 4, features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
        )

        self.read_weights = nn.Sequential(
            FastAttention(config, features * 8),
            nn.Linear(features * 8, self.read_slots * available_slots),
            nn.Tanh()
        )

        self.update_weights = nn.Sequential(
            FastAttention(config, features * 8),
            nn.Linear(features * 8, self.read_slots),
            nn.Sigmoid()
        )

        self.update_memory = nn.Sequential(
            FastAttentionImage(config, in_channels + channels),
            nn.Conv2d(in_channels + channels, channels, 3, 1, 1, bias=False),
            nn.ReLU(True)
        )

        self.apply_memory = nn.Sequential(
            FastAttentionImage(config, channels * 2),
            nn.Conv2d(channels * 2, channels, 3, 1, 1, bias=False),
            nn.ReLU(True)
        )

        self.apply(weights_init)

    def forward(self, inputs):
        global memory, update_memory
        batch_size = inputs.shape[0]
        encoded = self.encode(inputs).reshape(batch_size, -1)
        read = self.read_weights(encoded).reshape((batch_size, -1, self.available_slots))
        read_weights, read_indices = torch.max(read, dim=2)
        update_weights = self.update_weights(encoded)
        indices = read_indices.cpu().numpy()
        read_memory = []

        for i_batch in range(len(indices)):
            m = torch.zeros((channels, image_size, image_size), device="cuda")
            for i_slot, i in enumerate(indices[i_batch]):
                r = torch.tensor(memory[i_batch][i], dtype=torch.float, device="cuda")
                r = update_weights[i_batch, i_slot] * self.update_memory(torch.cat((r, inputs[i_batch]), dim=0).unsqueeze(dim=0)).squeeze(dim=0) + (1.0 - update_weights[i_batch, i_slot]) * r
                #update start
                if update_weights[i_batch, i_slot] > 0.5:
                    update = r.detach().cpu().numpy()
                    for update_i, update_value in enumerate(update):
                        update_memory[i_batch][i][update_i] = 0.5 * update_value + 0.5 * update_memory[i_batch][i][update_i]
                #update end
                m = read_weights[i_batch, i_slot] * self.apply_memory(torch.cat((r, m), dim=0).unsqueeze(dim=0)).squeeze(dim=0)
            read_memory.append(torch.tanh(m))

        return torch.stack(read_memory, dim=0)


class DatabaseImageMemory(nn.Module):
    def __init__(self, config, in_channels, read_slots):
        super(DatabaseImageMemory, self).__init__()
        self.read_slots = read_slots
        self.in_channels = in_channels
        self.config = config
        self.slots = slots

        batch_size = 16

        self.memory_access = MemoryAccess(config, in_channels, self.slots, self.read_slots)

        self.main_logic = nn.Sequential(
            FastAttentionImage(config, channels + in_channels),
            nn.Conv2d(in_channels + channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),

            nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )

        self.apply(weights_init)

    def forward(self, inputs):
        og_shape = inputs.shape
        m = self.memory_access(inputs)
        return self.main_logic(torch.cat((inputs, m), dim=1)).reshape(og_shape)

    def save_db(self):
        global memory
        memory.dump("./image_memory")
