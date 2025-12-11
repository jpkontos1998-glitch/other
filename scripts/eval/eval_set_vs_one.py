import argparse

from pyengine.core.evaluation import evaluate
from pyengine.utils.loading import get_train_info, log_dir_from_fn, load_rl_model, load_env
from pyengine import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", default=100, type=int)
    parser.add_argument("--model_paths", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--save_dir", help="Dummy arg for submit script")
    args = parser.parse_args()

    fns = args.model_paths.split(",")
    train_infos = [get_train_info(log_dir_from_fn(fn)) for fn in fns]
    opponent_info = get_train_info(log_dir_from_fn(args.opponent))
    assert all([opponent_info["full_info"] == info["full_info"] for info in train_infos])
    assert all([opponent_info["barrage"] == info["barrage"] for info in train_infos])
    has_arrangement = ["arrangement_rng_state" in info for info in train_infos]
    if any(has_arrangement):
        assert all(has_arrangement)
        for info in train_infos:
            assert opponent_info["arrangement_rng_state"] == info["arrangement_rng_state"]
    env = load_env(fns[0], args.num_envs)
    print(fns)

    opponent, opp_arrangements = load_rl_model(args.opponent)
    for i, fn_i in enumerate(fns):
        model_i, arrangements_i = load_rl_model(fn_i)
        with utils.eval_mode(model_i, opponent):
            outcome = (
                evaluate(model_i, arrangements_i, opponent, opp_arrangements, env).mean().item()
            )
        print(f"{fn_i}: {outcome}")
