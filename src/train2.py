import argparse
from log import (
    log,
    important,
    pretty_format,
    set_status_state,
    set_substeps,
    ProgressStatus,
    LogTypes,
)
from model2 import Model
from encoder import Encoder
from discriminator2 import Discriminator
from config import Configuration
from plot import plot_simple_array, show_fakes

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
from statistics import mean, median
from itertools import tee
import os
import numpy as np
import math
import json

device = "cuda"

important("cherrymachinedx")
important("Parsing args")

parser = argparse.ArgumentParser()
parser.add_argument("--max-epochs", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--max-lr", type=float, default=0.00001)

args = parser.parse_args()
log(pretty_format(args.__dict__))
config = Configuration(args)

state = {
    "lr_step": -1,
    "loss_history": [],
    "loss_history_real": [],
    "loss_history_discriminator": [],
}
if os.path.isfile("./state.json"):
    with open("./state.json", "r") as f:
        state = json.load(f)

image_transforms1 = transforms.RandomChoice(
    [
        transforms.RandomAdjustSharpness(1.00, p=1.0),
        transforms.RandomAdjustSharpness(1.05, p=1.0),
        transforms.RandomAdjustSharpness(1.1, p=1.0),
        transforms.RandomAdjustSharpness(1.15, p=1.0),
    ]
)
image_transforms2 = nn.Sequential(
    transforms.ColorJitter(0.05, 0.01, (0.9, 1.0), 0.1),
    # transforms.RandomHorizontalFlip(),
)

# dataset = Dataset(config)
dataset = dset.ImageFolder(
    root="dataset/images/",
    transform=transforms.Compose(
        [
            transforms.RandomResizedCrop((64, 64), scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
model = Model(config).to(device)
encoder = Encoder(config).to(device)
discriminator = Discriminator(config).to(device)
model.train()
encoder.eval()

criterion = nn.BCELoss(reduction="mean")
criterion_mse = nn.MSELoss(reduction="mean")
lr_start_div_factor = 3
lr_d = 0.001
lr_m = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr_m, betas=(0.5, 0.9))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.9))
total_steps = 100_000
pct_start = 0.0001
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr_d,
    div_factor=lr_start_div_factor,
    total_steps=total_steps,
    pct_start=pct_start,
    anneal_strategy="linear",
    three_phase=True,
    base_momentum=0.5,
    max_momentum=0.5,
)
scheduler_d = torch.optim.lr_scheduler.OneCycleLR(
    optimizer_d,
    max_lr=lr_d,
    div_factor=lr_start_div_factor,
    total_steps=total_steps,
    pct_start=pct_start,
    anneal_strategy="linear",
    three_phase=True,
    base_momentum=0.5,
    max_momentum=0.5,
)

if os.path.isfile("./trained_model"):
    trained_model = torch.load("./trained_model", map_location=torch.device(device))
    encoder.load_state_dict(trained_model["encoder"])

if os.path.isfile("./trained_model2"):
    trained_model = torch.load("./trained_model2", map_location=torch.device(device))
    model.load_state_dict(trained_model["model"])
    optimizer.load_state_dict(trained_model["model_optimizer"])
    scheduler.load_state_dict(trained_model["model_scheduler"])
    discriminator.load_state_dict(trained_model["discriminator"])
    optimizer_d.load_state_dict(trained_model["discriminator_optimizer"])
    scheduler_d.load_state_dict(trained_model["discriminator_scheduler"])


def pack_loss_history(loss_history):
    new_list = []
    for i in range(0, (len(loss_history) // 2 * 2), 2):
        current = loss_history[i]
        nxt = loss_history[i + 1]

        if current[1] <= nxt[1]:
            new_list.append([(current[0] + nxt[0]) / 2, current[1] + nxt[1]])
        else:
            new_list.append(current)
            new_list.append(nxt)

    return new_list


important("Starting training")
loss_history = state["loss_history"]
loss_history_real = state["loss_history_real"]
loss_history_discriminator = state["loss_history_discriminator"]
set_status_state(ProgressStatus(args.max_epochs))
m_loss_adjust = 1.0
d_loss_adjust = 1.0
for epoch in range(args.max_epochs):
    # dataset.load_batches()
    epoch_losses = []

    set_substeps(min(len(dataloader), 10))
    for batch, x in enumerate(dataloader):
        if batch > 10:
            break

        noise = torch.randn(x[0].shape[0], 100, 1, 1, device=device).to(device)

        x = x[0].to(device)
        x_source = image_transforms2(image_transforms1(x))

        x_interpolated = nn.functional.interpolate(x_source, size=16, mode="bilinear")
        encoded = encoder(x_interpolated).detach()
        optimizer_d.zero_grad()
        disc_real = discriminator(x_source)
        real_labes = torch.ones(x.shape[0]).to(device)
        disc_real_loss = criterion(disc_real.view(-1), real_labes)
        disc_real_loss.backward()
        # optimizer_d.step()
        # optimizer_d.zero_grad()
        pred = model(x_interpolated, encoded)
        fake_labels = torch.zeros(x.shape[0]).to(device)
        disc_fake = discriminator(pred)
        disc_fake_loss = criterion(disc_fake.view(-1), fake_labels) * (
            1.0 / max(disc_real_loss.item(), 1.0)
        )
        disc_fake_loss.backward()
        optimizer_d.step()

        optimizer.zero_grad()
        y_pred = model(x_interpolated, encoded)
        disc_pred = discriminator(y_pred)
        real_labes = torch.ones(x.shape[0]).to(device)
        loss_d = criterion(disc_pred.view(-1), real_labes)
        loss_d_factor = 1.0 / max(disc_real_loss.item(), 1.0)
        loss_real = 10 * criterion_mse(
            y_pred,
            x,
        )
        loss_d_scaled = loss_d * loss_d_factor
        loss = loss_real + loss_d_scaled

        loss_history.append([loss.item(), 1])
        loss_history_discriminator.append([loss_d_scaled.item(), 1])
        loss_history_real.append([loss_real.item(), 1])

        loss.backward()
        optimizer.step()

        log(
            f"{epoch}:{batch} - loss: {round(loss.item(), 2)} ({round(loss.item() - loss_d_scaled.item(), 2)} + {round(loss_d_scaled.item(), 2)}) - loss real: {round(disc_real_loss.item(), 2)}, - loss fake: {round(disc_fake_loss.item(), 2)}, lr: {round(math.log10(scheduler.get_last_lr()[0]), 3)}",
            repeating_status=True,
            substep=True,
        )
        # epoch_losses.append(round(math.log10(loss.item()), 2))
    log(
        "",
        repeating_status=True,
    )
    state["lr_step"] += 1
    scheduler_d.step()
    scheduler.step()
    state["loss_history"] = loss_history
    state["loss_history_real"] = loss_history_real
    state["loss_history_discriminator"] = loss_history_discriminator
    with open("./state.json", "w") as f:
        json.dump(state, f)
    # torch.save(model.state_dict(), "../trained_model")
    loss_history = pack_loss_history(loss_history)
    loss_history_real = pack_loss_history(loss_history_real)
    loss_history_discriminator = pack_loss_history(loss_history_discriminator)
    # plot_simple_array([round(math.log10(x[0]), 2) for x in loss_history], "../loss_history.png")
    # log(predict(model, input_text), multiline=True, type=LogTypes.DATA)
    model.train()

trained_model = {
    "model": model.state_dict(),
    "model_optimizer": optimizer.state_dict(),
    "model_scheduler": scheduler.state_dict(),
    "discriminator": discriminator.state_dict(),
    "discriminator_optimizer": optimizer_d.state_dict(),
    "discriminator_scheduler": scheduler_d.state_dict(),
}
torch.save(trained_model, "./trained_model2")
plot_simple_array(
    [
        [round(math.log10(x[0] + 0.000_1), 2) for x in loss_history],
        [round(math.log10(x[0] + 0.000_1), 2) for x in loss_history_real],
        [round(math.log10(x[0] + 0.000_1), 2) for x in loss_history_discriminator],
    ],
    "./loss_history.png",
)

model.eval()
with torch.no_grad():
    it = iter(dataloader)
    real = torch.cat((next(it)[0], next(it)[0]))
    images = nn.functional.interpolate(real, size=16, mode="bilinear").to(device)
    encoded = encoder(images)
    fakes = model(images, encoded)
show_fakes(real.cpu(), fakes.cpu(), "output.png")

important("Done")
