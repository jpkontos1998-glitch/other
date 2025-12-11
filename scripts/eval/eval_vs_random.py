import argparse

from pyengine.core.evaluation import evaluate, uniform_random
from pyengine.utils.loading import get_train_info, load_rl_model, log_dir_from_fn, load_env
from pyengine import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", default=100, type=int)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_dir", help="Dummy arg for submit script")
    args = parser.parse_args()

    train_info = get_train_info(log_dir_from_fn(args.model_path))
    env = load_env(args.model_path, num_envs=args.num_envs)
    model, arrangements = load_rl_model(args.model_path)
    default_arrangements = env.conf.initial_arrangements
    with utils.eval_mode(model):
        perf = (
            evaluate(model, arrangements, uniform_random, default_arrangements, env).mean().item()
        )
    print(f"Average performance over {2 * args.num_envs} games: {round(perf, 2)}")
