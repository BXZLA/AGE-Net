import random

import numpy as np
import torch


class Normalizer:

    def __init__(self, tensor=None):
        self.mean = torch.mean(tensor) if tensor is not None else torch.tensor(0.0)
        self.std = torch.std(tensor) if tensor is not None else torch.tensor(1.0)

    def norm(self, values):
        return (values - self.mean) / self.std

    def denorm(self, values):
        return values * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state):
        self.mean = state["mean"]
        self.std = state["std"]


def set_seed(seed, cuda=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def infer_n_conv(state_dict):
    indices = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("convs.") and key.split(".")[1].isdigit()
    }
    return max(indices) + 1 if indices else 4


def load_checkpoint(path):

    return torch.load(path, map_location="cpu", weights_only=False)


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def move_inputs(inputs, device):
    return (
        inputs[0].to(device),
        inputs[1].to(device),
        inputs[2].to(device),
        [index.to(device) for index in inputs[3]],
        inputs[4].to(device),
        inputs[5].to(device),
        inputs[6].to(device),
        inputs[7].to(device),
        inputs[8].to(device),
    )
