import os
import glob
from importlib.machinery import ExtensionFileLoader
import argparse

import torch
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

root = os.path.dirname(os.path.abspath(__file__))
pystratego_path = glob.glob(f"{root}/../build/pystratego*.so")[0]
pystratego = ExtensionFileLoader("pystratego", pystratego_path).load_module()


parser = argparse.ArgumentParser()
parser.add_argument("--variant", default="classic", choices=["classic", "barrage"])
parser.add_argument("--sample", default="random", choices=["random", "first"])
args = parser.parse_args()

if args.variant == "classic":
    reset_behavior = pystratego.ResetBehavior.RANDOM_JB_CLASSIC_BOARD
else:
    assert args.variant == "barrage"
    reset_behavior = pystratego.ResetBehavior.RANDOM_JB_BARRAGE_BOARD


def run(num_envs, num_rows=400, num_reps=10):
    env = pystratego.StrategoRolloutBuffer(
        num_rows,
        num_envs,
        reset_behavior=reset_behavior,
        move_memory=32,
        two_square_rule=True,
    )
    action_tensor = torch.zeros(num_envs, dtype=torch.int32, device="cuda")

    dps = []
    for _ in range(num_reps):
        start = time.time_ns()
        env.reset()
        num_steps = 5 * num_rows
        for t in range(num_steps):
            if args.sample == "random":
                env.sample_random_legal_action(action_tensor)
            else:
                assert args.sample == "first"
                env.sample_first_legal_action(action_tensor)

            # FIXME: Optionally double check that the actions selected are indeed valid
            # assert(legal_action_mask[range(num_envs), action_tensor].all())

            env.apply_actions(action_tensor)

            # env.compute_infostate_tensor(t)
            # env.compute_reward_pl0(t)

        end = time.time_ns()
        dp = (end - start) * 1e-3 / (num_envs * num_steps)
        dps.append(dp)

    return {"mean": np.mean(dps), "max": np.max(dps), "min": np.min(dps)}


if __name__ == "__main__":
    grid = [2**x for x in range(1, 12)]
    xs = []
    ys_mean = []
    ys_min = []
    ys_max = []
    for num_envs in grid:
        latency = run(num_envs)
        xs += [num_envs]
        ys_mean += [latency["mean"]]
        ys_min += [latency["min"]]
        ys_max += [latency["max"]]
        print(
            f"Num envs = {num_envs:3}  latency mean {latency['mean']} (min {latency['min']}  max {latency['max']}) µs / sample"
        )

    fig, ax = plt.subplots()
    ax.set_title(f"µstratego benchmark (sample={args.sample})")
    ax.fill_between(xs, ys_min, ys_max, alpha=0.4)
    ax.loglog(xs, ys_mean, "o-")
    ax.set_xticks(grid)
    ax.set_xticklabels(grid)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel("Num environments")
    ax.set_ylabel("Latency [µs / sample]")
    ax.grid()
    fig.savefig("benchmark.pdf", bbox_inches="tight")
