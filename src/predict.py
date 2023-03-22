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
from model import Model
from model2 import Model as Model2
from config import Configuration
from plot import plot_simple_array, show_predict, save_generation_snapshot

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

device = "cpu"

important("cherrymachinedx")
important("Parsing args")

parser = argparse.ArgumentParser()
args = parser.parse_args()
log(pretty_format(args.__dict__))
config = Configuration(args)

model = Model(config).to(device)
model.eval()
model2 = Model2(config).to(device)
model2.eval()

if os.path.isfile("./trained_model"):
    trained_model = torch.load("./trained_model", map_location=torch.device(device))
    model.load_state_dict(trained_model["model"])

if os.path.isfile("./trained_model2_no_gan"):
    trained_model = torch.load(
        "./trained_model2_no_gan", map_location=torch.device(device)
    )
    model2.load_state_dict(trained_model["model"])
elif os.path.isfile("./trained_model2"):
    trained_model = torch.load("./trained_model2", map_location=torch.device(device))
    model2.load_state_dict(trained_model["model"])

with torch.no_grad():
    important("Generating seed")
    torch.manual_seed(112)
    noise = torch.randn(32, 100, 1, 1, device=device)
    important("Generating first generation")
    x = model(noise)
    important("Generating second generation")
    fakes = model2(x)
    save_generation_snapshot("predict", fakes)
important("Saving images")
show_predict(x, fakes, "predict.png")

important("Done")
