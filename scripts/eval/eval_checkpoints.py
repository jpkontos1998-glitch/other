import argparse
import os

import numpy as np

from pyengine.core.evaluation import evaluate
from pyengine.utils.loading import get_train_info, load_rl_model, load_env
from pyengine import utils


def get_ordered_models(log_dir):
    models = {}
    print(os.listdir(log_dir))
    for f in os.listdir(log_dir):
        if "model" in f and "init_model" not in f and "ptho" not in f:
            # find indices for start and end of checkpoint number
            b = f.index("l") + 1
            e = f.index(".")
            models[int(f[b:e])] = log_dir + "/" + f
    return {k: models[k] for k in sorted(models.keys())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", default=100, type=int)
    parser.add_argument("--log_dir")
    parser.add_argument("--save_dir", help="Dummy arg for submit script")
    args = parser.parse_args()

    train_info = get_train_info(args.log_dir)
    ordered_models = get_ordered_models(args.log_dir)
    env = load_env(next(iter(ordered_models.values())), num_envs=args.num_envs)

    outcomes = np.zeros((len(ordered_models), len(ordered_models)))
    for i, (row_id, row_model) in enumerate(ordered_models.items()):
        row_net, row_arrangements = load_rl_model(row_model)
        for j, (col_id, col_model) in enumerate(ordered_models.items()):
            col_net, col_arrangements = load_rl_model(col_model)
            with utils.eval_mode(row_net, col_net):
                outcomes[i, j] = (
                    evaluate(row_net, row_arrangements, col_net, col_arrangements, env)
                    .mean()
                    .item()
                )
    np.save(f"{args.log_dir}/checkpoint_head_to_heads.npy", outcomes)
    with open(f"{args.log_dir}/checkpoint_head_to_heads.txt", "w") as f:
        f.write(np.array2string(outcomes.round(2)))
