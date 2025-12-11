from argparse import ArgumentParser

import torch

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
    parser.add_argument("--num_positions", default=1000, type=int)
    parser.add_argument("--rl_model_path", required=True)
    parser.add_argument("--belief_model_path")
    parser.add_argument("--depth", default=10, type=int)
    parser.add_argument("--stepsize1", default=10, type=int)
    parser.add_argument("--stepsize2", default=10, type=int)
    parser.add_argument("--temperature", default=0.001, type=float)
    parser.add_argument("--max_num_samples", default=200, type=int)
    parser.add_argument("--save_dir", help="Dummy arg for submit script")
    args = parser.parse_args()

    train_info = get_train_info(log_dir_from_fn(args.rl_model_path))
    eval_env = load_env(args.rl_model_path, num_envs=1)
    model, arrangments = load_rl_model(args.rl_model_path)
    search_env = load_env(args.rl_model_path, num_envs=1000, verbose=False)
    if args.belief_model_path:
        belief_model = load_belief_model(args.belief_model_path)
    else:
        belief_model = None
    searcher1 = SearchBot(
        model,
        search_env,
        iterations=args.search_iterations,
        depth=args.depth,
        stepsize=args.stepsize1,
        temperature=args.temperature,
        td_lambda=1.0,
        max_num_samples=args.max_num_samples,
        belief_model=belief_model,
    )
    searcher2 = SearchBot(
        model,
        search_env,
        iterations=args.search_iterations,
        depth=args.depth,
        stepsize=args.stepsize2,
        temperature=args.temperature,
        td_lambda=1.0,
        max_num_samples=args.max_num_samples,
        belief_model=belief_model,
    )
    outcomes = torch.zeros(args.num_positions, 2)
    for i in range(args.num_positions):
        utils.set_seed_everywhere(i)
        outcomes[i, 0] = evaluate_one_sided(
            searcher1, arrangments[0], searcher2, arrangments[1], eval_env
        )
        utils.set_seed_everywhere(i)
        outcomes[i, 1] = -evaluate_one_sided(
            searcher2, arrangments[0], searcher1, arrangments[1], eval_env
        )
        print(f"Num positions so far: {i+1}; search performance: {outcomes[: i + 1].mean().item()}")
