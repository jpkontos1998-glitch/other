import os
import pickle
from collections import defaultdict
from datetime import datetime
import wandb


class ValueStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.counter = 0
        self.summation = 0.0
        self.max_value = -1e38
        self.min_value = 1e38
        self.max_idx = -1
        self.min_idx = -1

    def append(self, v, count=1):
        self.counter += count

        self.summation += v
        if v > self.max_value:
            self.max_value = v
            self.max_idx = self.counter
        if v < self.min_value:
            self.min_value = v
            self.min_idx = self.counter

    def mean(self):
        if self.counter == 0:
            print("Counter %s is 0")
            assert False
        return self.summation / self.counter

    def sum(self):
        return self.summation

    def summary(self, info=None):
        info = "" if info is None else info
        if self.counter > 1:
            return "%s[%4d]: avg: %8.4f, min: %8.4f[%4d], max: %8.4f[%4d]" % (
                info,
                self.counter,
                self.summation / self.counter,
                self.min_value,
                self.min_idx,
                self.max_value,
                self.max_idx,
            )
        elif self.counter == 1:
            prefix = f"{info}: "
            if isinstance(self.min_value, int):
                return prefix + f"{self.min_value}"
            else:
                return prefix + f"{self.min_value:.2f}"
        else:
            return "%s[0]" % (info)


class MultiCounter:
    def __init__(
        self,
        root,
        use_wandb=False,
        *,
        wb_exp_name=None,
        wb_run_name=None,
        wb_group_name=None,
        config=None,
        rank=0,
    ):
        self.stats = defaultdict(lambda: ValueStats())
        self.last_time = datetime.now()
        self.max_key_len = 0
        if rank == 0:
            self.pikl_path = os.path.join(root, "log.pkl")
        else:
            self.pikl_path = os.path.join(root, f"log_rank{rank}.pkl")
        self.history = []

        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project=wb_exp_name,
                name=wb_run_name,
                group=wb_group_name[-128:],  # wandb group name has a max length of 128
                config={} if config is None else config,
                dir=root,
            )

    def __getitem__(self, key):
        if len(key) > self.max_key_len:
            self.max_key_len = len(key)

        return self.stats[key]

    def reset(self):
        for k in self.stats.keys():
            self.stats[k].reset()

        self.last_time = datetime.now()

    def summary(self, global_counter, *, reset=True, prefix="", print_msg=True):
        assert self.last_time is not None
        time_elapsed = (datetime.now() - self.last_time).total_seconds()

        self.history.append({k: v.mean() for k, v in self.stats.items() if v.counter > 0})
        with open(self.pikl_path, "ab") as f:
            pickle.dump(self.history[-1], f)

        if print_msg:
            print("[%d] Time spent = %.2f s" % (global_counter, time_elapsed))

            sorted_keys = sorted(
                [k for k, v in self.stats.items() if v.counter > 0], key=logger_sort
            )
            for k in sorted_keys:
                v = self.stats[k]
                if v.counter > 1:
                    continue
                info = str(global_counter) + ": " + k.ljust(self.max_key_len + 2)
                print(v.summary(info=info))

            for k in sorted_keys:
                v = self.stats[k]
                if v.counter == 1:
                    continue
                info = str(global_counter) + ": " + k.ljust(self.max_key_len + 2)
                print(v.summary(info=info))

        if self.use_wandb:
            to_log = {f"{prefix}{k}": v.mean() for k, v in self.stats.items() if v.counter > 0}
            wandb.log(to_log)

        if reset:
            self.reset()


def logger_sort(item):
    if not item.startswith("eval/"):
        return (0, item)
    try:
        checkpoint_num = int(item.split("model")[1])
        return (2, checkpoint_num, item)
    except (IndexError, ValueError):
        return (1, item)
