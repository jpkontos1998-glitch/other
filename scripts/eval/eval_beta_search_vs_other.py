from argparse import ArgumentParser

import torch

from pyengine import utils
from pyengine.utils.loading import (
    load_rl_model,
    log_dir_from_fn,
    get_train_info,
    load_env,
)
from pyengine.core.evaluation import evaluate_one_sided
from pyengine.core.search import SearchBot
from pyengine.belief.naive import get_beta_belief

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_positions", default=1000, type=int)
    parser.add_argument("--rl_model_path1", required=True)
    parser.add_argument("--search_iterations", default=1, type=int)
    parser.add_argument("--depth", default=10, type=int)
    parser.add_argument("--stepsize", default=10, type=int)
    parser.add_argument("--temperature", default=0.001, type=float)
    parser.add_argument("--max_num_samples", default=200, type=int)
    parser.add_argument("--rl_model_path2", required=True)
    parser.add_argument("--num_envs", default=1024, type=int)
    parser.add_argument("--save_dir", help="Dummy arg for submit script")
    args = parser.parse_args()

    train_info = get_train_info(log_dir_from_fn(args.rl_model_path1))
    model, arrangements = load_rl_model(args.rl_model_path1)
    opponent_model, opponent_arrangements = load_rl_model(args.rl_model_path2)
    env = load_env(args.rl_model_path1, num_envs=1)
    search_env = load_env(args.rl_model_path1, num_envs=args.num_envs, verbose=False)
    searcher = SearchBot(
        model,
        search_env,
        depth=args.depth,
        stepsize=args.stepsize,
        temperature=args.temperature,
        td_lambda=1.0,
        max_num_samples=args.max_num_samples,
        belief_model=get_beta_belief,
    )
    outcomes = torch.zeros(args.num_positions, 2)
    for i in range(args.num_positions):
        utils.set_seed_everywhere(i)
        outcomes[i, 0] = evaluate_one_sided(
            searcher, arrangements[0], opponent_model, opponent_arrangements[1], env
        )
        utils.set_seed_everywhere(i)
        outcomes[i, 1] = -evaluate_one_sided(
            opponent_model, opponent_arrangements[0], searcher, arrangements[1], env
        )
        print(f"Num positions so far: {i+1}; search performance: {outcomes[: i + 1].mean().item()}")
