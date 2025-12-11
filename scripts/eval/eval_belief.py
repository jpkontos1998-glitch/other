from argparse import ArgumentParser

import torch
from torch.amp import autocast

from pyengine.belief.evaluation import EvalManager
from pyengine.utils.loading import load_env, load_rl_model, load_belief_model
from pyengine.utils import get_pystratego
from pyengine import utils

pystratego = get_pystratego()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--belief_model_path", type=str)
    parser.add_argument("--rl_model_path", type=str)
    args = parser.parse_args()
    belief_model = load_belief_model(args.belief_model_path)
    rl_model, arr = load_rl_model(args.rl_model_path)
    env = load_env(args.rl_model_path, num_envs=1000, traj_len_per_player=2000)
    env.change_reset_behavior_to_random_initial_arrangement(arr)
    utils.set_seed_everywhere(0)
    env.reset()
    with torch.no_grad(), autocast("cuda", torch.bfloat16):
        with utils.eval_mode(rl_model):
            manager = EvalManager(env, rl_model, torch.bfloat16)
        with utils.eval_mode(belief_model):
            data = manager.evaluate(belief_model)
    for key, value in data.items():
        print(f"{key}: {value}")
