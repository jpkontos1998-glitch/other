import argparse

import torch
from torch.amp import autocast
import numpy as np

from pyengine.arrangement.utils import to_string, filter_terminal
from pyengine.arrangement.sampling import generate_arrangements
from pyengine.utils.loading import load_rl_model, load_arrangement_model
from pyengine.core.env import Stratego
from pyengine import utils
from pyengine.utils.init_helpers import CHAR_TO_VAL, CHAR_TO_VAL_BLUE
from numpy.linalg import lstsq

RED_CHARS = [ch for ch in CHAR_TO_VAL if ch not in ("_", "A", "M")]
BLUE_CHARS = [ch for ch in CHAR_TO_VAL_BLUE if ch not in ("Y")]

def extract_counts_from_board_str(board_str):
    counts = []
    for rch, bch in zip(RED_CHARS, BLUE_CHARS):
        counts.append(board_str.count(rch))
        counts.append(board_str.count(bch))
        counts.append(board_str.count(rch.lower()))
        counts.append(board_str.count(bch.lower()))
    return np.array(counts)


def run_games(model_path, num_steps, save_dir):
    model, _ = load_rl_model(model_path)
    init_model = load_arrangement_model(model_path.replace("model", "init_model"))
    # Create environment
    num_envs = 1000
    env = Stratego(num_envs=num_envs, traj_len_per_player=100, max_num_moves_between_attacks=100)

    # Setup save directory
    # os.makedirs(save_dir, exist_ok=True)
    all_feat = []
    all_values = []

    current_env = 0

    for t in range(num_steps):
        if t % 2_000 == 0:
            arrs = filter_terminal(to_string(generate_arrangements(1000, init_model)[0]))
            env.change_reset_behavior_to_random_initial_arrangement((arrs[::2], arrs[1::2]))

        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16), utils.eval_mode(
            model
        ):
            tensor_dict = model(
                env.current_infostate_tensor,
                env.current_piece_ids,
                env.current_legal_action_mask,
            )
            values = tensor_dict["value"].softmax(dim=-1)[current_env]
            scalar_values = values[-1] - values[0]
            if env.current_acting_player == 1:
                scalar_values = -scalar_values

        if not env.current_is_terminal[current_env]:
            features = extract_counts_from_board_str(env.current_board_strs[current_env])
            all_feat.append(features)
            all_values.append(scalar_values.unsqueeze(-1).cpu().numpy())
            current_env = (current_env + 1) % num_envs

        env.apply_actions(tensor_dict["action"])

        if t % (num_steps // 10) == 0:
            print(f"Collected {t} steps.")
            np.save(f"{save_dir}/all_feat.npy", np.stack(all_feat, axis=0))
            np.save(f"{save_dir}/all_values.npy", np.stack(all_values, axis=0))

    np.save(f"{save_dir}/all_feat.npy", np.stack(all_feat, axis=0))
    np.save(f"{save_dir}/all_values.npy", np.stack(all_values, axis=0))

    # print(f"Collected {len(all_feat)} features and values.")
    # x = lstsq(
    #     np.stack(all_feat, axis=0),
    #     np.stack(all_values, axis=0),
    # )[0]
    # res = np.stack(all_values, axis=0) - np.dot(np.stack(all_feat, axis=0), x)
    # print(x.T)
    # res_abs = np.abs(res)
    # print(f"Mean absolute error: {res_abs.mean()}")
    # print(f"Max absolute error: {res_abs.max()}")
    # print(f"Min absolute error: {res_abs.min()}")
    # print(f"Std absolute error: {res_abs.std()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect games using an RL model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the RL model")
    parser.add_argument("--num_steps", type=int, required=True, help="Number of steps to run")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save the results")

    args = parser.parse_args()

    run_games(
        model_path=args.model_path,
        num_steps=args.num_steps,
        save_dir=args.save_dir,
    )
