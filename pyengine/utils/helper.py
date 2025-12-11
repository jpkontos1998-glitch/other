import os
from typing import Any
import torch
import numpy as np
import random
from tabulate import tabulate
import typeguard
import matplotlib.pyplot as plt
import psutil
from pynvml import *

nvmlInit()

from .constants import N_ACTION, N_BOARD_CELL
from .load_pystratego import get_pystratego

pystratego = get_pystratego()


def get_weighted_uniform_policy(legal_action_mask: torch.Tensor) -> torch.Tensor:
    assert legal_action_mask.ndim == 2
    assert legal_action_mask.shape[1] == N_ACTION
    assert legal_action_mask.dtype == torch.bool
    device = legal_action_mask.device
    origin_squares = torch.arange(N_ACTION, device=device) % N_BOARD_CELL
    counts = torch.zeros(legal_action_mask.size(0), N_BOARD_CELL, device=device, dtype=torch.int32)
    counts.scatter_add_(
        1,
        origin_squares.unsqueeze(0).expand_as(legal_action_mask),
        legal_action_mask.to(torch.int32),
    )
    expanded_counts = counts.gather(
        1, origin_squares.unsqueeze(0).expand_as(legal_action_mask)
    ).clamp(min=1)
    unnorm = legal_action_mask.float() / expanded_counts.float()
    return unnorm / unnorm.sum(dim=1, keepdim=True)


def mem2str(num_bytes):
    assert num_bytes >= 0
    if num_bytes >= 2**30:  # GB
        val = float(num_bytes) / (2**30)
        result = "%.3f GB" % val
    elif num_bytes >= 2**20:  # MB
        val = float(num_bytes) / (2**20)
        result = "%.3f MB" % val
    elif num_bytes >= 2**10:  # KB
        val = float(num_bytes) / (2**10)
        result = "%.3f KB" % val
    else:
        result = "%d bytes" % num_bytes
    return result


def get_mem_usage(msg=""):
    mem = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    used = mem2str(process.memory_info().rss)
    return (
        f"Mem {msg}: used: {used}, avail: {mem2str(mem.available)}, total: {(mem2str(mem.total))}"
    )


def get_gpumem_usage_gb():
    handle = nvmlDeviceGetHandleByIndex(0)
    mem = nvmlDeviceGetMemoryInfo(handle)
    return mem.used / (2**30)


def get_mem_usage_gb():
    process = psutil.Process(os.getpid())
    used = process.memory_info().rss / (2**30)
    return used


def generate_grid(cols, rows, figsize=5, fig_h=None, fig_w=None, squeeze=True):
    if fig_h is None:
        fig = plt.figure(figsize=(cols * figsize, rows * figsize))
    else:
        fig = plt.figure(figsize=(cols * fig_w, rows * fig_h))
    ax = fig.subplots(rows, cols, squeeze=squeeze)
    return fig, ax


def get_all_files(root, file_extension, contain=None) -> list[str]:
    files = []
    for folder, _, fs in os.walk(root):
        for f in fs:
            if file_extension is not None:
                if f.endswith(file_extension):
                    if contain is None or contain in os.path.join(folder, f):
                        files.append(os.path.join(folder, f))
            else:
                if contain in f:
                    files.append(os.path.join(folder, f))
    return files


def wrap_ruler(text: str, max_len=40):
    text_len = len(text)
    if text_len > max_len:
        return text_len

    left_len = (max_len - text_len) // 2
    right_len = max_len - text_len - left_len
    return ("=" * left_len) + text + ("=" * right_len)


def set_seed_everywhere(seed):
    # set the random seed for torch, numpy, and python
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def count_parameters(model):
    rows = []
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        rows.append([name, params])
        total_params += params

    for row in rows:
        row.append(row[-1] / total_params * 100)

    rows.append(["Total", total_params, 100])
    table = tabulate(
        rows, headers=["Module", "#Params", "%"], intfmt=",d", floatfmt=".2f", tablefmt="orgtbl"
    )
    print(table)


def extended_isinstance(obj: Any, type_: Any) -> bool:
    try:
        typeguard.check_type(obj, type_)
        return True
    except typeguard.TypeCheckError:
        return False


def pprint_infostate_channels(channels):
    abbreviated = []
    sequences = {}

    for item in channels:
        if "[" in item and "]" in item:
            var, index = item.split("[")
            index = index.strip("]")
            if var not in sequences:
                sequences[var] = []
            sequences[var].append(int(index))
        else:
            abbreviated.append(item)

    for var, indices in sequences.items():
        if len(indices) > 1:
            # Assuming all indices are negative and continuous
            positive_max = abs(min(indices))  # e.g., -86 becomes 86
            abbreviated.append(f"{var}[{positive_max}]")
        else:
            # Single index entry
            positive_index = abs(indices[0])
            abbreviated.append(f"{var}[{positive_index}]")

    print("infostate channels")
    for i, item in enumerate(abbreviated):
        print(f"\t {i}: {item}")


class VerboseMethod:
    def __init__(self, method):
        self.method = method

    def __call__(self, *args, **kwargs):
        print(f">>> [method] {self.method.__name__}(pos: {args}; kw: {kwargs})")
        out = self.method(*args, **kwargs)
        print(f"<<< {out}")
        return out


class VerboseShim:
    def __init__(self, obj):
        self.obj = obj

    def __getattr__(self, name):
        attr = self.obj.__getattribute__(name)
        if callable(attr):
            return VerboseMethod(attr)
        else:
            print(">>> [attribute]", name)
            print("<<<", attr)
            return attr
