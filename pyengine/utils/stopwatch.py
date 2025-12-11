from contextlib import contextmanager
from collections import defaultdict
import time
import tabulate

import numpy as np
import torch


class Stopwatch:
    """stop watch in MS"""

    def __init__(self):
        self.times = defaultdict(list)
        self.reset_time = time.time()

        self.init_time = time.time()

    @property
    def total_time(self):
        return time.time() - self.init_time

    @property
    def elapsed_time_since_reset(self):
        return time.time() - self.reset_time

    def reset(self):
        self.times = defaultdict(list)
        self.reset_time = time.time()

    @contextmanager
    def time(self, key, sync=False):
        if sync:
            torch.cuda.synchronize()
        t = time.time()
        yield
        if sync:
            torch.cuda.synchronize()

        self.times[key].append(1000 * (time.time() - t))  # record in ms

    def summary(self, reset: bool = True):
        """
        Print a tabulated summary of recorded timings.

        Adds a 'total (ms)' column and formats millisecond values with
        thousands separators for better readability.
        """
        headers = ["name", "num", "total (ms)", "t/call (ms)", "%"]
        total = 0.0
        times = {}

        # Aggregate stats for each timer
        for k, v in self.times.items():
            sum_t = float(np.sum(v))  # total time for this key (ms)
            mean_t = sum_t / len(v)  # average time per call (ms)
            times[k] = (len(v), sum_t, mean_t)
            if "/" not in k:  # exclude nested timers from grand total
                total += sum_t

        print("Timer Info:")
        rows = []
        data = {}

        for k, (num, sum_t, mean_t) in times.items():
            pct = (100.0 * sum_t / total) if total else 0.0
            rows.append(
                [
                    k,
                    f"{num:,}",  # num with thousands separator
                    f"{sum_t:,.1f}",  # total time with thousands separator
                    f"{mean_t:,.1f}",  # mean time with thousands separator
                    f"{pct:.1f}",
                ]
            )
            data[f"time/{k}"] = sum_t / 1000  # convert to seconds for return dict

        # Append overall total row
        rows.append(
            [
                "total",
                "",
                f"{total:,.1f}",
                "",
                "100.0",
            ]
        )

        print(tabulate.tabulate(rows, headers=headers, tablefmt="orgtbl"))

        if reset:
            self.reset()

        return data
