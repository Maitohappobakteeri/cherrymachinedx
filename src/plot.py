import matplotlib.pyplot as plt
import numpy as np
import math
import torchvision.utils as vutils
import torch.nn as nn
import os
from torchvision.utils import save_image
import torch

import common


def plot_simple_array(x_list, save_as_filename):
    fig, axs = plt.subplots(1, 1)
    start, end = axs.get_xlim()
    axs.xaxis.set_ticks(np.arange(start, end, 10))
    axs.set_title(save_as_filename)
    for x in x_list:
        y = [y for y in range(len(x))]
        axs.plot(y, x)
        p5, p4, a, b, c, d = np.polyfit(y, x, 5)
        predict_amount = min(500, math.floor((len(x) ** 0.5)))
        axs.plot(
            [y for y in range(len(x) + predict_amount)],
            [
                p5 * i**5 + p4 * i**4 + a * i**3 + b * i**2 + c * i + d
                for i in range(len(x) + predict_amount)
            ],
        )
    axs.grid()
    plt.savefig(save_as_filename)
    plt.close()


def show_fakes_simple(real, fakes, filename):
    plt.subplot(2, 1, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real, padding=2, normalize=True), (1, 2, 0))
    )

    plt.subplot(2, 1, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(
        np.transpose(vutils.make_grid(fakes, padding=2, normalize=True), (1, 2, 0))
    )
    plt.savefig(filename)
    plt.close()


def show_fakes_gen1(real, fakes, real_mod, filename):
    diff = torch.divide(torch.subtract(real, fakes), 2.0)
    mean_diff = torch.mean(torch.abs(diff))
    stacked = torch.stack((real, real_mod, fakes, diff), dim=1).view((-1, 3, 64, 64))

    plt.subplot(1, 1, 1)
    plt.axis("off")
    plt.title(f"Real/Edit/Fake/Diff (Error {round(100 * mean_diff.item(), 2)}%)")
    plt.imshow(
        np.transpose(vutils.make_grid(stacked, padding=2, normalize=True, nrow=8), (1, 2, 0))
    )

    # plt.subplot(4, 1, 2)
    # plt.axis("off")
    # plt.title("Fake Images (small decode)")
    # plt.imshow(
    #     np.transpose(vutils.make_grid(nn.functional.interpolate(
    #                 small_decoder_fakes,
    #                 size=64,
    #                 mode="bilinear",
    #             ), padding=2, normalize=True), (1, 2, 0))
    # )

    # plt.subplot(4, 1, 3)
    # plt.axis("off")
    # plt.title("Fake Images (decode)")
    # plt.imshow(
    #     np.transpose(vutils.make_grid(decoder_fake, padding=2, normalize=True), (1, 2, 0))
    # )

    # plt.subplot(4, 1, 4)
    # plt.axis("off")
    # plt.title("Fake Images (final)")
    # plt.imshow(
    #     np.transpose(vutils.make_grid(fakes, padding=2, normalize=True), (1, 2, 0))
    # )

    plt.savefig(filename)
    plt.close()

def show_fakes_gen1_with_fakes(real, fakes, real_fakes, real_mod, filename):
    diff = torch.divide(torch.subtract(real, fakes), 2.0)
    mean_diff = torch.mean(torch.abs(diff))
    stacked = torch.stack((real, real_mod, fakes, diff), dim=1).view((-1, 3, 64, 64))

    plt.subplot(2, 1, 1)
    plt.axis("off")
    plt.title(f"Real/Edit/Fake/Diff (Error {round(100 * mean_diff.item(), 2)}%)")
    plt.imshow(
        np.transpose(vutils.make_grid(stacked, padding=2, normalize=True, nrow=8), (1, 2, 0))
    )

    plt.subplot(2, 1, 2)
    plt.axis("off")
    plt.title("Pure fakes")
    plt.imshow(
        np.transpose(vutils.make_grid(real_fakes, padding=2, normalize=True, nrow=8), (1, 2, 0))
    )

    # plt.subplot(4, 1, 3)
    # plt.axis("off")
    # plt.title("Fake Images (decode)")
    # plt.imshow(
    #     np.transpose(vutils.make_grid(decoder_fake, padding=2, normalize=True), (1, 2, 0))
    # )

    # plt.subplot(4, 1, 4)
    # plt.axis("off")
    # plt.title("Fake Images (final)")
    # plt.imshow(
    #     np.transpose(vutils.make_grid(fakes, padding=2, normalize=True), (1, 2, 0))
    # )

    plt.savefig(filename)
    plt.close()


def show_fakes_gen1_with_fakes_with_steps(real, step1, step2, fakes, real_fakes, filename):
    diff = torch.divide(torch.subtract(real, fakes), 2.0)
    mean_diff = torch.mean(torch.abs(diff))
    stacked = torch.stack((real, step1, step2, fakes), dim=1).view((-1, 3, 64, 64))

    plt.subplot(2, 1, 1)
    plt.axis("off")
    plt.title(f"Real/Edit/Fake/Diff (Error {round(100 * mean_diff.item(), 2)}%)")
    plt.imshow(
        np.transpose(vutils.make_grid(stacked, padding=2, normalize=True, nrow=8), (1, 2, 0))
    )

    plt.subplot(2, 1, 2)
    plt.axis("off")
    plt.title("Pure fakes")
    plt.imshow(
        np.transpose(vutils.make_grid(real_fakes, padding=2, normalize=True, nrow=8), (1, 2, 0))
    )

    # plt.subplot(4, 1, 3)
    # plt.axis("off")
    # plt.title("Fake Images (decode)")
    # plt.imshow(
    #     np.transpose(vutils.make_grid(decoder_fake, padding=2, normalize=True), (1, 2, 0))
    # )

    # plt.subplot(4, 1, 4)
    # plt.axis("off")
    # plt.title("Fake Images (final)")
    # plt.imshow(
    #     np.transpose(vutils.make_grid(fakes, padding=2, normalize=True), (1, 2, 0))
    # )

    plt.savefig(filename)
    plt.close()


def show_predict(first, fakes, filename):
    plt.subplot(2, 1, 1)
    plt.axis("off")
    plt.title("First generation")
    plt.imshow(
        np.transpose(vutils.make_grid(first, padding=2, normalize=True), (1, 2, 0))
    )

    plt.subplot(2, 1, 2)
    plt.axis("off")
    plt.title("Second generation")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                fakes,
                padding=2,
                normalize=True,
            ),
            (1, 2, 0),
        )
    )

    plt.savefig(filename)
    plt.close()


def show_fakes(real, fakes, filename):
    plt.subplot(3, 1, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real, padding=2, normalize=True), (1, 2, 0))
    )

    plt.subplot(3, 1, 2)
    plt.axis("off")
    plt.title("Interpolated Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                nn.functional.interpolate(
                    nn.functional.interpolate(real, size=16, mode="bilinear"),
                    size=64,
                    mode="nearest",
                ),
                padding=2,
                normalize=True,
            ),
            (1, 2, 0),
        )
    )

    plt.subplot(3, 1, 3)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(
        np.transpose(vutils.make_grid(fakes, padding=2, normalize=True), (1, 2, 0))
    )
    plt.savefig(filename)
    plt.close()


def save_generation_snapshot(prefix, fakes):
    prefix_dir = os.path.join(common.cache_dir, prefix)
    common.ensure_dir(prefix_dir)
    fakes = torch.mul(torch.add(fakes, 1.0), 0.5)

    for i in range(fakes.shape[0]):
        dirname = os.path.join(prefix_dir, str(i))
        common.ensure_dir(dirname)
        num_files = len(common.list_all_files(dirname))
        filename = os.path.join(dirname, f"{num_files:012}.png")
        image = fakes[i]
        save_image(image, filename)
