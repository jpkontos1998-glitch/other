from argparse import ArgumentParser

import wandb
import torch
import numpy as np

from pyengine.belief.naive import get_beta_belief
from pyengine import utils
from pyengine.utils.loading import (
    load_rl_model,
    load_belief_model,
    log_dir_from_fn,
    get_train_info,
    load_env,
)
from pyengine.core.evaluation import evaluate_one_sided
from pyengine.core.search import SearchBot

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_positions", default=500, type=int)
    parser.add_argument("--rl_model_path1", required=True)
    parser.add_argument("--belief_model_path")
    parser.add_argument("--depth", default=10, type=int)
    parser.add_argument("--stepsize", default=10, type=int)
    parser.add_argument("--num_envs", default=1024, type=int)
    parser.add_argument("--temperature", default=0.001, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--rl_model_path2", required=True)
    parser.add_argument("--max_num_moves", type=int, default=4000)
    parser.add_argument("--td_lambda", default=1.0, type=float)
    parser.add_argument("--max_num_samples", default=100, type=int)
    parser.add_argument("--uniform_magnet", default=0, type=int)
    parser.add_argument("--use_wb", type=int, default=0)
    parser.add_argument("--wb_exp_name", default="eval_search_vs_other")
    parser.add_argument("--save_dir", help="Dummy arg for submit script")
    args = parser.parse_args()

    train_info = get_train_info(log_dir_from_fn(args.rl_model_path1))
    model, arrangements = load_rl_model(args.rl_model_path1)
    opponent_model, opponent_arrangements = load_rl_model(args.rl_model_path2)
    env = load_env(
        args.rl_model_path1, num_envs=1, traj_len_per_player=2000, max_num_moves=args.max_num_moves
    )
    search_env = load_env(
        args.rl_model_path1, num_envs=args.num_envs, quiet=1, max_num_moves=args.max_num_moves
    )
    if args.belief_model_path == "beta":
        belief_model = get_beta_belief
    elif args.belief_model_path:
        belief_model = load_belief_model(args.belief_model_path)
    else:
        belief_model = None
    searcher = SearchBot(
        model,
        search_env,
        depth=args.depth,
        stepsize=args.stepsize,
        temperature=args.temperature,
        td_lambda=args.td_lambda,
        max_num_samples=args.max_num_samples,
        uniform_magnet=bool(args.uniform_magnet),
        dtype=torch.bfloat16,
        belief_model=belief_model,
    )
    outcomes = torch.zeros(args.num_positions, 2)
    if args.use_wb:
        run = wandb.init(project=args.wb_exp_name, config=vars(args))
    for i in range(args.num_positions):
        utils.set_seed_everywhere(i + args.seed * 10000)
        outcomes[i, 0] = evaluate_one_sided(
            searcher, arrangements[0], opponent_model, opponent_arrangements[1], env
        )
        utils.set_seed_everywhere(i + args.seed * 10000)
        outcomes[i, 1] = -evaluate_one_sided(
            opponent_model, opponent_arrangements[0], searcher, arrangements[1], env
        )
        print(
            f"Num games so far: {2 * (i + 1)}; search performance: {outcomes[: i + 1].mean().item()}"
        )
        data = {
            "running_average_performance": outcomes[: i + 1].mean().item(),
            "num_games": 2 * (i + 1),
            **{"search_policy_info/" + k: np.mean(v) for k, v in searcher.stats.items()},
        }
        time_data = searcher.stopwatch.summary()
        for k, v in time_data.items():
            data[k] = v
        if args.use_wb:
            wandb.log(data)
        print(data)
