import argparse
import numpy as np
import pickle

from pyengine.core.evaluation import evaluate
from pyengine.utils.loading import get_train_info, log_dir_from_fn, load_rl_model, load_env
from pyengine import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", default=100, type=int)
    parser.add_argument("--model_paths", required=True)
    parser.add_argument("--arrangement_paths", required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--save_dir", help="Dummy arg for submit script")
    args = parser.parse_args()

    fns = args.model_paths.split(",")
    arr_fns = args.arrangement_paths.split(",")
    assert len(fns) == len(arr_fns)
    train_infos = [get_train_info(log_dir_from_fn(fn)) for fn in fns]
    assert all([train_infos[0]["full_info"] == info["full_info"] for info in train_infos])
    assert all([train_infos[0]["barrage"] == info["barrage"] for info in train_infos])
    has_arrangement = ["arrangement_rng_state" in info for info in train_infos]
    if any(has_arrangement):
        assert all(has_arrangement)
        for info in train_infos:
            assert train_infos[0]["arrangement_rng_state"] == info["arrangement_rng_state"]
    env = load_env(fns[0], args.num_envs)
    print(fns)

    outcomes = np.zeros((len(train_infos), len(train_infos)))
    for i, (fn_i, arr_i) in enumerate(zip(fns, arr_fns)):
        model_i, _ = load_rl_model(fn_i)
        with open(arr_i, "rb") as f_i:
            arrangements_i = pickle.load(f_i)
        for j, (fn_j, arr_j) in enumerate(zip(fns, arr_fns)):
            model_j, _ = load_rl_model(fn_j)
            with open(arr_j, "rb") as f_j:
                arrangements_j = pickle.load(f_j)
            with utils.eval_mode(model_i, model_j):
                if i < j or args.full:
                    outcomes[i, j] = (
                        evaluate(model_i, arrangements_i, model_j, arrangements_j, env)
                        .mean()
                        .item()
                    )
    print(np.array2string(outcomes.round(2) - outcomes.round(2).T))
    print("average performance of each policy")
    print(
        np.array2string(
            (outcomes.round(2) - outcomes.round(2).T).sum(axis=-1) / (outcomes.shape[0] - 1)
        )
    )
