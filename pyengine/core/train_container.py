from copy import deepcopy
from typing import Optional

import torch
from torch import nn

from pyengine.utils.loading import load_state_dict


class TrainContainer:
    def __init__(
        self,
        network: nn.Module,
        lr: float,
        weight_decay: float,
        from_checkpoint: Optional[str] = None,
    ):
        network = deepcopy(network)
        if from_checkpoint:
            load_state_dict(network, torch.load(from_checkpoint))
        self.net = network
        param_groups = [
            {
                "params": [
                    param for name, param in self.net.named_parameters() if "bias" not in name
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [param for name, param in self.net.named_parameters() if "bias" in name],
                "weight_decay": 0.0,
            },
        ]
        self.optim = torch.optim.AdamW(param_groups, lr=lr)
        if from_checkpoint:
            optim_path = from_checkpoint.replace(".pthw", ".ptho")
            print(f"Loading optimizer state from: {optim_path}")
            self.optim.load_state_dict(torch.load(optim_path))
