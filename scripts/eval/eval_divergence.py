from argparse import ArgumentParser

import torch
from torch.amp import autocast

from pyengine.utils import eval_mode
from pyengine.core.env import Stratego
from pyengine.utils.loading import load_rl_model


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path1", type=str, required=True)
    parser.add_argument("--model_path2", type=str, required=True)
    parser.add_argument("--temperature2", type=float, default=1.0)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--num_envs", type=int, default=500)
    args = parser.parse_args()

    model1, arr1 = load_rl_model(args.model_path1)
    model2, _ = load_rl_model(args.model_path2)

    env = Stratego(num_envs=args.num_envs, traj_len_per_player=100)
    action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
    kl = []
    ell1 = []

    for _ in range(args.num_steps):
        with torch.no_grad(), eval_mode(model1), autocast(device_type="cuda", dtype=torch.bfloat16):
            tensor_dict1 = model1(
                env.current_infostate_tensor,
                env.current_piece_ids,
                env.current_legal_action_mask,
            )
            tensor_dict2 = model2(
                env.current_infostate_tensor,
                env.current_piece_ids,
                env.current_legal_action_mask,
            )
        nonterm = ~env.current_is_terminal
        logpi2 = torch.log_softmax(tensor_dict2["action_log_probs"] / args.temperature2, dim=-1)
        kl.append(
            (tensor_dict1["action_log_probs"].exp() * (tensor_dict1["action_log_probs"] - logpi2))[
                nonterm
            ]
            .sum(dim=-1)
            .mean()
            .item()
        )
        ell1.append(
            torch.abs(tensor_dict1["action_log_probs"].exp() - logpi2.exp())[nonterm]
            .sum(dim=-1)
            .mean()
            .item()
        )
        action_tensor = tensor_dict1["action"]
        env.apply_actions(action_tensor)

    print(f"kl: {torch.tensor(kl).mean()}")
    print(f"ell1: {torch.tensor(ell1).mean()}")


if __name__ == "__main__":
    main()
