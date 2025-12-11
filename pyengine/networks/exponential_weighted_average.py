import copy

import torch

from pyengine.utils.loading import load_state_dict


class EMA:
    def __init__(self, model, decay, resume_from=None):
        self.original_model = model
        if resume_from:
            load_state_dict(self.original_model, torch.load(resume_from))
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        if resume_from:
            ema_path = resume_from.replace(".pthw", ".pthm")
            load_state_dict(self.ema_model, torch.load(ema_path))
        for param in self.ema_model.parameters():
            param.detach_()

    def update(self):
        with torch.no_grad():
            for ema_param, orig_param in zip(
                self.ema_model.parameters(), self.original_model.parameters()
            ):
                ema_param.mul_(self.decay).add_((1.0 - self.decay) * orig_param)

    def apply_shadow(self):
        for ema_param, orig_param in zip(
            self.ema_model.parameters(), self.original_model.parameters()
        ):
            orig_param.data.copy_(ema_param.data)
