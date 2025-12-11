#!/usr/bin/env python3

import os
import argparse
import torch
from torch.amp import autocast

from pyengine.arrangement.utils import to_string, filter_terminal
from pyengine.arrangement.sampling import generate_arrangements
from pyengine.utils.loading import load_rl_model, load_arrangement_model
from pyengine.core.env import Stratego
from pyengine import utils


def collect_games(model_path, num_steps, save_dir):
    model, _ = load_rl_model(model_path)
    init_model = load_arrangement_model(model_path.replace("model", "init_model"))
    # Create environment
    env = Stratego(num_envs=1_000, traj_len_per_player=100, max_num_moves_between_attacks=100)

    # Setup save directory
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "collected_games.msgpack")
    env.save_games(save_path)

    for t in range(num_steps):
        if t % 2_000 == 0:
            print(f"Updating arrangements; {t} of {num_steps} complete")
            arrangements = []
            for _ in range(2):
                (
                    arrs,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = generate_arrangements(1_000, init_model)
                string_arrs = filter_terminal(to_string(arrs))
                arrangements.append(string_arrs)
            env.change_reset_behavior_to_random_initial_arrangement(arrangements)

        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16), utils.eval_mode(
            model
        ):
            actions, _, _ = model(
                env.current_infostate_tensor,
                env.current_piece_ids,
                env.current_legal_action_mask,
            )
        env.sample_random_legal_action(actions)

        env.apply_actions(actions)

    env.stop_saving_games()
    print(f"Saved games to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect games using an RL model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the RL model")
    parser.add_argument("--num_steps", type=int, required=True, help="Number of steps to run")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the games")

    args = parser.parse_args()

    collect_games(
        model_path=args.model_path,
        num_steps=args.num_steps,
        save_dir=args.save_dir,
    )
