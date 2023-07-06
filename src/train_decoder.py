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
from model_decoder import Model
from encoder import Encoder
from is_same_discriminator import IsSameDiscriminator
from config import Configuration
from plot import plot_simple_array, show_fakes_gen1_with_fakes, save_generation_snapshot

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
import random
import sys

# sys.exit(0)

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

image_size = 64

image_transforms1 = transforms.RandomChoice(
    [
        transforms.RandomAdjustSharpness(1.00, p=1.0),
        transforms.RandomAdjustSharpness(1.05, p=1.0),
        # transforms.RandomAdjustSharpness(1.1, p=1.0),
        # transforms.RandomAdjustSharpness(1.15, p=1.0),
    ]
)

image_transforms2 = transforms.RandomChoice([
        transforms.ColorJitter(0, 0, 0, 0),
        transforms.ColorJitter(0.05, 0.01, 0.01, 0.01),
        # transforms.ColorJitter(0.1, 0.03, 0.03, 0.03),
        # transforms.ColorJitter(0.15, 0.04, 0.04, 0.04),
])

image_transforms3 = nn.Sequential(
    transforms.RandomResizedCrop(64, scale=(0.95, 1.0), ratio=(1.0, 1.0)),
)

# dataset = Dataset(config)
dataset = dset.ImageFolder(
    root="dataset/images/",
    transform=transforms.Compose(
        [ 
            transforms.Resize(image_size),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
model = Model(config).to(device)
encoder = Encoder(config).to(device)
discriminator = IsSameDiscriminator(config).to(device)
model.train()
encoder.train()
discriminator.train()

criterion = nn.BCELoss(reduction="mean")
criterion_mse = nn.MSELoss(reduction="none")
criterion_mse_mean = nn.MSELoss(reduction="mean")
lr_start_div_factor = 10
# should start at div by / 100
lr_mul = 10
lr_d = 0.001 * lr_mul
lr_m = 0.001 * lr_mul
optimizer = optim.Adam(model.parameters(), lr=lr_m, betas=(0.5, 0.5))
optimizer_e = optim.Adam(encoder.parameters(), lr=lr_m, betas=(0.5, 0.5))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.5))
total_steps = 100_000_000
pct_start = 0.000001
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr_m,
    div_factor=lr_start_div_factor,
    total_steps=total_steps,
    pct_start=pct_start,
    #anneal_strategy="linear",
    three_phase=False,
    base_momentum=0.5,
    max_momentum=0.5,
)
scheduler_e = torch.optim.lr_scheduler.OneCycleLR(
    optimizer_e,
    max_lr=lr_m,
    div_factor=lr_start_div_factor,
    total_steps=total_steps,
    pct_start=pct_start,
    #anneal_strategy="linear",
    three_phase=False,
    base_momentum=0.5,
    max_momentum=0.5,
)
scheduler_d = torch.optim.lr_scheduler.OneCycleLR(
    optimizer_d,
    max_lr=lr_d,
    div_factor=lr_start_div_factor,
    total_steps=total_steps,
    pct_start=pct_start,
    #anneal_strategy="linear",
    three_phase=False,
    base_momentum=0.5,
    max_momentum=0.5,
)


if os.path.isfile("./trained_model"):
    trained_model = torch.load("./trained_model", map_location=torch.device(device))
    model.load_state_dict(trained_model["model"])
    optimizer.load_state_dict(trained_model["model_optimizer"])
    scheduler.load_state_dict(trained_model["model_scheduler"])
    encoder.load_state_dict(trained_model["encoder"])
    optimizer_e.load_state_dict(trained_model["encoder_optimizer"])
    scheduler_e.load_state_dict(trained_model["encoder_scheduler"])
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

def mse_loss_per_image(image, target):
    loss = criterion_mse(image, target)
    loss = torch.mean(loss, dim=(1, 2, 3))
    return loss

def mse_loss_for_channel(image, target):
    loss = criterion_mse(image, target)
    loss = torch.mean(loss, dim=(2, 3))
    loss = torch.pow(loss, 2)
    loss = torch.mean(loss, dim=(1))
    loss = torch.pow(loss, 2)
    return 100 * torch.mean(loss)

def mse_loss(image, target):
    loss = criterion_mse(image, target)
    return torch.mean(loss)

def save_predictions():
    torch.manual_seed(112)
    dataloader = DataLoader(dataset, batch_size=8)
    real = next(iter(dataloader))[0]
    # noise = torch.randn(16, 100, 1, 1, device=device)
    encoder.eval()
    model.eval()
    with torch.no_grad():
        encoder_output = encoder(real.to(device))
        decoder_fakes = model(encoder.reparametrize(*encoder_output))
        real_mod = image_transforms3(image_transforms2(image_transforms1(real)))
        
        true_fakes = []
        for x in range(4):
            noise = [torch.mul(torch.randn(1, 600, 1, 1, device=device).to(device), ((x * 8 + n + 1) * 0.10)) for n in range(8)]
            noise = torch.cat(noise, dim=0)
            true_fakes.append(model(noise))

        show_fakes_gen1_with_fakes(real.cpu(), decoder_fakes.cpu(), torch.cat(true_fakes, dim=0).cpu(), real_mod.cpu(), "output.png")
        save_generation_snapshot("v1", decoder_fakes.cpu())
    model.train()
    encoder.train()
    torch.manual_seed(random.randrange(0,10_000))

important("Starting training")
loss_history = state["loss_history"]
loss_history_real = state["loss_history_real"]
loss_history_discriminator = state["loss_history_discriminator"]
set_status_state(ProgressStatus(args.max_epochs))
m_loss_adjust = 1.0
d_loss_adjust = 1.0
target_batch = max(32, config.batch_size)
staleness_for_batch = 0
for epoch in range(args.max_epochs):
    # dataset.load_batches()
    epoch_losses = []

    # set_substeps(min(len(dataloader), 64 * 2))
    set_substeps(len(dataloader))
    for batch, x in enumerate(dataloader):
        # if batch >= 64 * 2:
        #     break
        if x[0].shape[0] != config.batch_size:
           continue
        noise = torch.randn(x[0].shape[0], 100, 1, 1, device=device).to(device)

        x = x[0].to(device)
        x_source = x
        x_modified_no_resize = image_transforms2(image_transforms1(x_source))
        x_modified = image_transforms3(x_modified_no_resize)

        fake_labels = torch.ones(x.shape[0]).to(device)
        real_labels = torch.zeros(x.shape[0]).to(device)

        model.requires_grad_(False)
        encoder.requires_grad_(False)
        discriminator.requires_grad_(True)

        disc_real = discriminator(torch.stack((x, x_modified)).view(x.shape[0], 6, 64, 64))
        loss_per_image = mse_loss_per_image(x_modified_no_resize, x)
        disc_real_loss = criterion(disc_real[:, 0].view(-1), real_labels) + criterion(disc_real[:, 1].view(-1), real_labels)
        # disc_real_loss = criterion(disc_real[:, 0].view(-1), real_labels)
        # disc_real_loss = disc_real_loss / 2
        (disc_real_loss / (target_batch // config.batch_size)).backward()

        loss_per_image = mse_loss_per_image(x, x)
        disc_real_identity = discriminator(torch.stack((x, x)).view(x.shape[0], 6, 64, 64))
        disc_real_identity_loss = criterion(disc_real_identity[:, 0].view(-1), real_labels) + criterion(disc_real_identity[:, 1].view(-1), real_labels)
        (disc_real_identity_loss / (target_batch // config.batch_size)).backward()

        rev_x = torch.flip(x, dims=(0,))
        rev_loss_per_image = mse_loss_per_image(x_modified_no_resize, rev_x)
        disc_fake2 = discriminator(torch.stack((rev_x, x_modified)).view(x.shape[0], 6, 64, 64))
        disc_fake_loss2 = criterion(disc_fake2[:, 0].view(-1), fake_labels) + criterion(disc_fake2[:, 1].view(-1), real_labels)
        # disc_fake_loss2 = criterion(disc_fake2[:, 0].view(-1), fake_labels)
        # disc_fake_loss2 = disc_fake_loss2 / 2
        disc_real_loss = torch.divide(disc_real_loss + disc_real_identity_loss + disc_fake_loss2, 3)
        (disc_fake_loss2 / (target_batch // config.batch_size)).backward()
        # optimizer_d.step()
        # optimizer_d.zero_grad()
        encoder_output = encoder(x_source)
        pred = model(encoder.reparametrize(*encoder_output))
        loss_per_image = mse_loss_per_image(pred, x)
        disc_fake = discriminator(torch.stack((x, pred)).view(x.shape[0], 6, 64, 64))
        disc_fake_loss = criterion(disc_fake[:, 0].view(-1), fake_labels) + criterion(disc_fake[:, 1].view(-1), fake_labels)
        # disc_fake_loss = criterion(disc_fake[:, 0].view(-1), fake_labels)
        # disc_fake_loss = disc_fake_loss / 2
        loss_d_factor = 1.0 / max(disc_real_loss.item(), 0.05)
        disc_fake_loss = disc_fake_loss
        (disc_fake_loss / (target_batch // config.batch_size)).backward()

        rev_pred = torch.flip(pred, dims=(0,))
        rev_loss_per_image = mse_loss_per_image(rev_pred, x)
        disc_fake2 = discriminator(torch.stack((x, rev_pred)).view(x.shape[0], 6, 64, 64))
        disc_fake_loss2 = criterion(disc_fake2[:, 0].view(-1), fake_labels) + criterion(disc_fake2[:, 1].view(-1), fake_labels)
        # disc_fake_loss2 = criterion(disc_fake2[:, 0].view(-1), fake_labels)
        # disc_fake_loss2 = disc_fake_loss2 / 2
        disc_fake_loss = torch.divide(disc_fake_loss + disc_fake_loss2, 2)
        
        (disc_fake_loss2 / (target_batch // config.batch_size)).backward()
        # disc_fake_loss.backward()

        # if (1 + batch) % (target_batch // config.batch_size) == 0 or (batch + 1) == len(dataloader):
        #     optimizer_d.step()

        #     optimizer_e.zero_grad()
        #     optimizer.zero_grad()
        #     optimizer_d.zero_grad()

        #     scheduler_d.step()

        model.requires_grad_(True)
        encoder.requires_grad_(True)
        discriminator.requires_grad_(False)

        encoder_output = encoder(x_source)
        (pred, predx16, pred_simple) = model(encoder.reparametrize(*encoder_output), outputx16=True)
        disc_pred = discriminator(torch.stack((x, pred)).view(x.shape[0], 6, 64, 64))
        loss_d = criterion(disc_pred[:, 0].view(-1), real_labels) + criterion(disc_pred[:, 1].view(-1), real_labels)
        # loss_d = criterion_mse_mean(disc_pred[:, 0].view(-1), real_labels) + criterion(disc_pred[:, 0].view(-1), real_labels)
        # loss_d = criterion(disc_pred[:, 0].view(-1), real_labels)
        # disc_pred = discriminator(quick_pred)
        # loss_d = loss_d + criterion(disc_pred.view(-1), real_labes) 
        # loss_d_factor = 1.0 / max(disc_real_loss.item(), 1.0)
        loss_d_scaled = loss_d * loss_d_factor * 0.1
        loss = loss_d_scaled

        beta = 1
        kl = -0.5 * torch.sum(1 + encoder_output[1] - encoder_output[0].pow(2) - encoder_output[1].exp(), -1)
        loss_e = torch.mean(beta*kl)
        # diff_mean_encoder = nn.functional.relu(torch.add(torch.mul(torch.mean(torch.abs(encoder_output[0])), -1), 0.5))
        # staleness = torch.mean(disc_pred, dim=2)
        # staleness = torch.var(staleness)
        # staleness = nn.functional.relu(torch.add(torch.mul(staleness, -1), 0.2))
        # staleness_for_batch = (staleness_for_batch * (10 * target_batch - 1) + staleness.item()) / (10 * target_batch)
        # staleness = staleness * staleness_for_batch * 10
        # staleness = nn.functional.relu(torch.sub(staleness, 0.1))
        # loss = (loss_e + loss_d_scaled + diff_mean_encoder + staleness) 
        # loss_mse = torch.mean(torch.mean(criterion_mse(pred, x_source), dim=1) / torch.add(torch.var(x_source, dim=1), 0.1))
        loss_mse = torch.mean(criterion_mse(pred, x_source))
        x_source_x16 = nn.functional.interpolate(x_source, 16)
        # loss_msex16 = torch.mean(torch.mean(criterion_mse(predx16, x_source_x16), dim=1) / torch.add(torch.var(x_source_x16, dim=1), 0.1))
        loss_msex16 = torch.mean(criterion_mse(predx16, x_source_x16))
        loss_simple = torch.mean(criterion_mse(pred_simple, x_source))
        loss_d_scaled = (loss_d_scaled * loss_mse ** 2) * 0.1

        loss = loss_e + 10 * loss_mse + loss_msex16 * 10 + loss_simple * 1 + loss_d_scaled
        (loss / (target_batch // config.batch_size)).backward()

        # pred = model(torch.randn(config.batch_size, 600, 1, 1, device=device).to(device))
        # disc_pred = discriminator(torch.stack((x, pred)).view(x.shape[0], 6, 64, 64))
        # loss_d_random = criterion(disc_pred[:, 1].view(-1), real_labels)
        # loss_d_random_scaled = loss_d_random * loss_d_factor * 0.1
        # (loss_d_random_scaled / (target_batch // config.batch_size)).backward()

        if (1 + batch) % (target_batch // config.batch_size) == 0 or (batch + 1) == len(dataloader):
        # if (1 + batch) % (32) == 0 or (batch + 1) == len(dataloader):
            # optimizer.step()
            # optimizer_e.step()

            # optimizer.zero_grad()
            # optimizer_d.zero_grad()
            # optimizer_e.zero_grad()

            # scheduler.step()
            # scheduler_e.step()

            optimizer.step()
            optimizer_d.step()
            optimizer_e.step()
            optimizer.zero_grad()
            optimizer_d.zero_grad()
            optimizer_e.zero_grad()
            scheduler.step()
            scheduler_d.step()
            scheduler_e.step()

        if (1 + batch) % (128) == 0 or (batch + 1) == len(dataloader):
            save_predictions()

        # optimizer.zero_grad()
        # optimizer_e.zero_grad()
        # encoder_output = encoder(x_source)
        # decoder_pred = model(encoder.reparametrize(*encoder_output))
        # kl = -0.5 * torch.sum(1 + encoder_output[1] - encoder_output[0].pow(2) - encoder_output[1].exp(), -1)
        # loss_e = mse_loss(decoder_pred, x) + torch.mean(beta*kl)
        # # loss_e = loss_e + (mse_loss_for_channel(quick_pred, x) + (mse_loss_for_channel(y_pred, x))) / loss_d_scaled.item()
        # loss = loss + loss_e

        # loss_e.backward()
        # optimizer.step()
        # optimizer_e.step()

        loss_history.append([loss.item(), 1])

        encoder_range = (torch.max(encoder_output[0]) - torch.min(encoder_output[0])).item()
        encoder_median = (torch.median(encoder_output[0])).item()
        var_encoder_range = (torch.max(encoder_output[1]) - torch.min(encoder_output[1])).item()
        var_encoder_median = (torch.median(encoder_output[1])).item()
        log(
            f"{epoch}:{batch} - loss: {round(loss.item(), 2)} ({round(loss.item() - loss_d_scaled.item(), 2)} + {round(loss_d_scaled.item(), 2)}) - loss real: {round(disc_real_loss.item(), 2)}, - loss fake: {round(disc_fake_loss.item(), 2)}, lr: {round(math.log10(scheduler.get_last_lr()[0]), 3)}, input range: {round(encoder_range, 3)}, input med: {round(encoder_median, 3)}, var range: {round(var_encoder_range, 3)}, var med: {round(var_encoder_median, 3)}",
            repeating_status=True,
            substep=True,
        )
        # epoch_losses.append(round(math.log10(loss.item()), 2))
    log(
        "",
        repeating_status=True,
    )
    state["lr_step"] += 1
    state["loss_history"] = loss_history
    state["loss_history_real"] = loss_history_real
    state["loss_history_discriminator"] = loss_history_discriminator
    # with open("./state.json", "w") as f:
    # json.dump(state, f)
    # torch.save(model.state_dict(), "../trained_model")
    loss_history = pack_loss_history(loss_history)
    loss_history_real = pack_loss_history(loss_history_real)
    loss_history_discriminator = pack_loss_history(loss_history_discriminator)
    # plot_simple_array([round(math.log10(x[0]), 2) for x in loss_history], "../loss_history.png")
    # log(predict(model, input_text), multiline=True, type=LogTypes.DATA)

trained_model = {
    "model": model.state_dict(),
    "model_optimizer": optimizer.state_dict(),
    "model_scheduler": scheduler.state_dict(),
    "encoder": encoder.state_dict(),
    "encoder_optimizer": optimizer_e.state_dict(),
    "encoder_scheduler": scheduler_e.state_dict(),
    "discriminator": discriminator.state_dict(),
    "discriminator_optimizer": optimizer_d.state_dict(),
    "discriminator_scheduler": scheduler_d.state_dict(),
}
torch.save(trained_model, "./trained_model")
# plot_simple_array(
#     [
#         [round(math.log10(x[0] + 0.01), 2) for x in loss_history],
#     ],
#     "../loss_history.png",
# )

save_predictions()
important("Done")
