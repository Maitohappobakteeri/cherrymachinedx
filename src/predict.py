import argparse
from log import (
    log,
    important,
    pretty_format,
)
from model import Model
from model2 import Model as Model2
from config import Configuration
from plot import show_predict, save_generation_snapshot

import torch
import os

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

if os.path.isfile("./trained_model2"):
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
