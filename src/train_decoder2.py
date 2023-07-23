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
from decoder2.model_improve import Model as Model2, max_steps
from encoder import Encoder
from is_same_discriminator import IsSameDiscriminator
from config import Configuration
from plot import plot_simple_array, show_fakes_gen1, show_fakes_gen1_with_fakes, save_generation_snapshot, show_fakes_gen1_with_fakes_with_steps

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
model.requires_grad_(False)
encoder.requires_grad_(False)
model.eval()
encoder.eval()

model2 = Model2(config).to(device)
model2.train()

criterion = nn.BCELoss(reduction="mean")
criterion_mse = nn.MSELoss(reduction="none")
criterion_mse_mean = nn.MSELoss(reduction="mean")
lr_start_div_factor = 10
# should start at div by / 100
lr_mul = 10
lr_m = 0.001 * lr_mul
optimizer = optim.Adam(model2.parameters(), lr=lr_m, betas=(0.99, 0.99))
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
    base_momentum=0.99,
    max_momentum=0.99,
)

if os.path.isfile("./trained_model"):
    trained_model = torch.load("./trained_model", map_location=torch.device(device))
    model.load_state_dict(trained_model["model"])
    encoder.load_state_dict(trained_model["encoder"])

if os.path.isfile("./trained_model2"):
    trained_model = torch.load("./trained_model2", map_location=torch.device(device))
    model2.load_state_dict(trained_model["model"])
    optimizer.load_state_dict(trained_model["model_optimizer"])
    scheduler.load_state_dict(trained_model["model_scheduler"])

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

def save_predictions(max_steps=max_steps, filename="output.png"):
    torch.manual_seed(112)
    dataloader = DataLoader(dataset, batch_size=8)
    real = next(iter(dataloader))[0]
    # noise = torch.randn(16, 100, 1, 1, device=device)
    model2.eval()
    with torch.no_grad():
        encoder_output = encoder(real.to(device))
        encoder_reparam = encoder.reparametrize(*encoder_output)
        model_output = model(encoder_reparam)
        noise_masks = create_noise_masks(model_output.shape)
        decoder_fakes = add_noise(model_output, 0, noise_masks)
        # example_steps = [(x / 1000) * max_steps // 1 for x in [0, 120, 240, 360, 480, 600, 720, 999]]
        example_steps = [x for x in range(0, max_steps, 50)]
        if max_steps <= 10:
            example_steps = [x for x in range(max_steps)]
        elif max_steps <= 100:
            example_steps = [x for x in range(0, max_steps, 10)]
        noise_masks2 = create_noise_masks(model_output[0:1].shape)
        data_example = torch.cat([add_noise(mix_images(real[0:1], model_output[0:1], n, device="cuda"), n, noise_masks2) for n in example_steps], dim=0)
        decoder_fakes_steps = []
        for step in range(max_steps):
            decoder_fakes1, decoder_fakes2 = torch.split(decoder_fakes, 4, dim=0)
            decoder_fakes1 = model2(decoder_fakes1, step)
            decoder_fakes2 = model2(decoder_fakes2, step)
            decoder_fakes = torch.cat((decoder_fakes1, decoder_fakes2), dim=0)
            if step in example_steps:
                decoder_fakes_steps.append(decoder_fakes[0:1])

            log(
                f"generating 0:{step}",
                repeating_status=True,
                substep=True,
                no_step=True
            )

        noise = torch.randn(8, 600, 1, 1, device=device).to(device)
        true_fakes = add_noise(model(noise), 0, noise_masks)
        for step in range(max_steps):
            true_fakes1, true_fakes2 = torch.split(true_fakes, 4, dim=0)
            true_fakes1 = model2(true_fakes1, step)
            true_fakes2 = model2(true_fakes2, step)
            true_fakes = torch.cat((true_fakes1, true_fakes2), dim=0)
            log(
                f"generating 1:{step}",
                repeating_status=True,
                substep=True,
                no_step=True
            )
                
        show_fakes_gen1_with_fakes_with_steps(real.cpu(), torch.cat((decoder_fakes_steps), dim=0).cpu(), decoder_fakes.cpu(), true_fakes.cpu(), data_example.cpu(), filename)
        save_generation_snapshot("v1", decoder_fakes.cpu())
    model2.train()
    torch.manual_seed(random.randrange(0,10_000))


def create_noise_masks(shape):
    return [shape, torch.randn(max_steps, device=device)]
    # return torch.normal(0.0, 50.0 / max_steps, size=(max_steps, *shape), device=device)


def add_noise(tensor, step, noise_masks):
    # noise = torch.sum(torch.permute(noise_masks[step:], (1, 2, 3, 4, 0)), dim=-1)

    # shape = noise_masks[0]
    # noise = torch.zeros(shape, device=device)
    # for seed in noise_masks[1][step:]:
    #     torch.manual_seed(seed)
    #     noise = noise + torch.normal(0.0, 1.0 / max_steps, size=shape, device=device)
    # return torch.add(tensor, noise).clamp(-1.0, 1.0)

    return tensor


def minmax_normalization(tensor, new_min, new_max):
    old_max = torch.max(tensor)
    old_min = torch.min(tensor)
    old_range = (old_max - old_min)
    if old_range > 0:
        return (tensor - old_min) / old_range * (new_max - new_min) + new_min
    return tensor * ((new_max + new_min) / 2)

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def mix_images(fake, real, step, device=None):
    device = device or fake.get_device()
    n = step / max_steps
    image_size = fake.shape[-1]
    gaussian = (torch.tensor(makeGaussian(image_size, image_size), dtype=torch.float)).reshape(1, 1, image_size, image_size).repeat(fake.shape[0], fake.shape[1], 1, 1).to(device)
    gaussian = minmax_normalization(gaussian, n, 1.0)
    noise = n * gaussian
    return torch.add(torch.mul(real.to(device), torch.ones(fake.shape).to(device) - noise), torch.mul(fake.to(device), noise))

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
    steps = 2
    set_substeps(len(dataloader) * steps)
    for batch, x in enumerate(dataloader):
        # if batch >= 64 * 2:
        #     break
        if x[0].shape[0] != config.batch_size:
           continue
        x = x[0].to(device)

        x_source = x
        x_modified_no_resize = image_transforms2(image_transforms1(x_source))
        x_modified = image_transforms3(x_modified_no_resize)
        encoder_output = encoder(x_source)
        encoder_reparam = encoder.reparametrize(*encoder_output)
        pred_og = model(encoder_reparam)
        noise = create_noise_masks(x.shape)
        for step in range(steps):
            adjusted_step = int(step * (max_steps / steps) + batch % (max_steps / steps))

            fake_labels = torch.ones(x.shape[0]).to(device)
            real_labels = torch.zeros(x.shape[0]).to(device)

            # disc_real = discriminator(torch.stack((x, x_modified)).view(x.shape[0], 6, 64, 64))
            # loss_per_image = mse_loss_per_image(x_modified_no_resize, x)
            # disc_real_loss = criterion(disc_real[:, 1].view(-1), real_labels) + criterion(disc_real[:, 2].view(-1), real_labels)
            # # disc_real_loss = criterion(disc_real[:, 1].view(-1), real_labels)
            # # disc_real_loss = disc_real_loss / 2
            # (disc_real_loss / (target_batch // config.batch_size) / max_steps).backward()

            # loss_per_image = mse_loss_per_image(x, x)
            # disc_real_identity = discriminator(torch.stack((x, x)).view(x.shape[0], 6, 64, 64))
            # disc_real_identity_loss = criterion(disc_real_identity[:, 1].view(-1), real_labels) + criterion(disc_real_identity[:, 2].view(-1), real_labels)
            # (disc_real_identity_loss / (target_batch // config.batch_size) / max_steps).backward()

            # rev_x = torch.flip(x, dims=(0,))
            # rev_loss_per_image = mse_loss_per_image(x_modified_no_resize, rev_x)
            # disc_fake2 = discriminator(torch.stack((rev_x, x_modified)).view(x.shape[0], 6, 64, 64))
            # disc_fake_loss2 = criterion(disc_fake2[:, 1].view(-1), fake_labels) + criterion(disc_fake2[:, 2].view(-1), real_labels)
            # # disc_fake_loss2 = criterion(disc_fake2[:, 1].view(-1), fake_labels)
            # # disc_fake_loss2 = disc_fake_loss2 / 2
            # disc_real_loss = torch.divide(disc_real_loss + disc_real_identity_loss + disc_fake_loss2, 3)
            # (disc_fake_loss2 / (target_batch // config.batch_size) / max_steps).backward()
            # # optimizer_d.step()
            # # optimizer_d.zero_grad()
            # encoder_output = encoder(x_source)
            # encoder_reparam = encoder.reparametrize(*encoder_output)
            # pred = model2(mix_images(model(encoder_reparam), x_source, step), encoder_reparam, step)
            # loss_per_image = mse_loss_per_image(pred, x)
            # disc_fake = discriminator(torch.stack((x, pred)).view(x.shape[0], 6, 64, 64))
            # disc_fake_loss = criterion(disc_fake[:, 1].view(-1), fake_labels) + criterion(disc_fake[:, 2].view(-1), fake_labels)
            # # disc_fake_loss = criterion(disc_fake[:, 1].view(-1), fake_labels)
            # # disc_fake_loss = disc_fake_loss / 2
            # loss_d_factor = 1.0 / max(disc_real_loss.item(), 0.05)
            # disc_fake_loss = disc_fake_loss
            # (disc_fake_loss / (target_batch // config.batch_size) / max_steps).backward()

            # rev_pred = torch.flip(pred, dims=(0,))
            # rev_loss_per_image = mse_loss_per_image(rev_pred, x)
            # disc_fake2 = discriminator(torch.stack((x, rev_pred)).view(x.shape[0], 6, 64, 64))
            # disc_fake_loss2 = criterion(disc_fake2[:, 1].view(-1), fake_labels) + criterion(disc_fake2[:, 2].view(-1), fake_labels)
            # # disc_fake_loss2 = criterion(disc_fake2[:, 1].view(-1), fake_labels)
            # # disc_fake_loss2 = disc_fake_loss2 / 2
            # disc_fake_loss = torch.divide(disc_fake_loss + disc_fake_loss2, 2)
            
            # (disc_fake_loss2 / (target_batch // config.batch_size) / max_steps).backward()
            # disc_fake_loss.backward()

            # model2.requires_grad_(True)
            mix1 = add_noise(mix_images(pred_og, x_source, adjusted_step), adjusted_step, noise)
            mix2 = add_noise(mix_images(pred_og, x_source, adjusted_step + 1), adjusted_step + 1, noise)
            pred, pred_diff  = model2(mix1, adjusted_step, returnDiff=True)
            # disc_pred = discriminator(torch.stack((x, pred)).view(x.shape[0], 6, 64, 64))
            # loss_d = criterion(disc_pred[:, 1].view(-1), real_labels) + criterion(disc_pred[:, 2].view(-1), real_labels)
            # loss_d = criterion_mse_mean(disc_pred[:, 0].view(-1), real_labels) + criterion(disc_pred[:, 1].view(-1), real_labels)
            # loss_d = criterion(disc_pred[:, 1].view(-1), real_labels)
            # disc_pred = discriminator(quick_pred)
            # loss_d = loss_d + criterion(disc_pred.view(-1), real_labes) 
            # loss_d_factor = 1.0 / max(disc_real_loss.item(), 1.0)
            # diff_mean_encoder = nn.functional.relu(torch.add(torch.mul(torch.mean(torch.abs(encoder_output[0])), -1), 0.5))
            # staleness = torch.mean(disc_pred, dim=2)
            # staleness = torch.var(staleness)
            # staleness = nn.functional.relu(torch.add(torch.mul(staleness, -1), 0.2))
            # staleness_for_batch = (staleness_for_batch * (10 * target_batch - 1) + staleness.item()) / (10 * target_batch)
            # staleness = staleness * staleness_for_batch * 10
            # staleness = nn.functional.relu(torch.sub(staleness, 0.1))
            # loss = (loss_e + loss_d_scaled + diff_mean_encoder + staleness) 
            

            target = mix2 - mix1
            target_max = torch.max(torch.abs(target))
            target_max = target_max if target_max.item() > 0 else 1.0
            loss_mse = criterion_mse_mean(pred_diff / target_max, target / target_max)
            loss = loss_mse
            # ((loss_mult * loss) / (target_batch // config.batch_size) / steps).backward()
            (loss / (target_batch // config.batch_size) / steps).backward()


            # noise = torch.randn(config.batch_size, 600, 1, 1, device=device).to(device)
            # pred = model2(model(noise), noise, step)
            # disc_pred = discriminator(torch.stack((x, pred)).view(x.shape[0], 6, 64, 64))
            # loss_d_random = criterion(disc_pred[:, 2].view(-1), real_labels)
            # loss_d_random_scaled = loss_d_random * loss_d_factor * 0.1
            # (loss_d_random_scaled / (target_batch // config.batch_size)).backward()

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

            mean_diff_data = criterion_mse_mean(mix1, mix2)
            mean_diff_pred = (criterion_mse_mean(pred, mix2) - mean_diff_data) / (mean_diff_data + 1e-12)
            log(
                f"{epoch}:{batch}:{step} - loss: {round(math.log10(1e-12 + loss.item()), 2)}, lr: {round(math.log10(1e-12 + scheduler.get_last_lr()[0]), 3)}, step_is_worse: {mean_diff_pred.item() >= 0}, diff: {round(100 * mean_diff_pred.item(), 5)}%, diff_data: {round(math.log10(1e-12 + mean_diff_data.item()), 2)}",
                repeating_status=True,
                substep=True,
            )
            # epoch_losses.append(round(math.log10(loss.item()), 2))
        if (1 + batch) % (target_batch // config.batch_size) == 0 or (batch + 1) == len(dataloader):
        # if (1 + batch) % (32) == 0 or (batch + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        if (1 + batch) % (128) == 0 or (batch + 1) == len(dataloader):
            save_predictions(10, "quick")
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
    "model": model2.state_dict(),
    "model_optimizer": optimizer.state_dict(),
    "model_scheduler": scheduler.state_dict(),
}
torch.save(trained_model, "./trained_model2")
# plot_simple_array(
#     [
#         [round(math.log10(x[0] + 0.01), 2) for x in loss_history],
#     ],
#     "../loss_history.png",
# )

save_predictions()
important("Done")
