import os

import torch


class Saver:
    def __init__(self, save_dir: str, model_name: str):
        self.save_dir = save_dir
        self.model_name = model_name

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def save(
        self,
        state_dict: dict,
        ema_state_dict: dict,
        opt_state_dict: dict,
        model_id: str | int,
    ) -> None:
        weight_name = f"{self.model_name}{model_id}.pthw"
        fn = os.path.join(self.save_dir, weight_name)
        torch.save(state_dict, fn)
        ema_name = weight_name.replace("pthw", "pthm")
        ema_fn = os.path.join(self.save_dir, ema_name)
        torch.save(ema_state_dict, ema_fn)
        opt_name = weight_name.replace("pthw", "ptho")
        opt_fn = os.path.join(self.save_dir, opt_name)
        torch.save(opt_state_dict, opt_fn)
