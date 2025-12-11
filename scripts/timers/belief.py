from argparse import ArgumentParser
import time

import torch

from pyengine.belief.sampling import generate
from pyengine.core.env import Stratego
from pyengine.utils.loading import load_belief_model

parser = ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--num_samples", type=int, default=100)
args = parser.parse_args()

model = load_belief_model(args.model_path)
env = Stratego(num_envs=1, traj_len_per_player=100, barrage=True)
# burn in
for i in range(10):
    generate(
        env.current_infostate_tensor.squeeze(0),
        env.current_piece_ids.squeeze(0),
        env.current_num_moves.squeeze(0),
        env.current_unknown_piece_position_onehot.squeeze(0),
        env.current_unknown_piece_counts.squeeze(0),
        env.current_unknown_piece_has_moved.squeeze(0),
        args.num_samples,
        model,
    )
times = []
action_tensor = torch.zeros(1, dtype=torch.int32, device="cuda")
for i in range(100):
    torch.cuda.synchronize()
    start = time.time()
    generate(
        env.current_infostate_tensor.squeeze(0),
        env.current_piece_ids.squeeze(0),
        env.current_num_moves.squeeze(0),
        env.current_unknown_piece_position_onehot.squeeze(0),
        env.current_unknown_piece_counts.squeeze(0),
        env.current_unknown_piece_has_moved.squeeze(0),
        args.num_samples,
        model,
    )
    torch.cuda.synchronize()
    times.append(time.time() - start)
    env.sample_random_legal_action(action_tensor)
    env.apply_actions(action_tensor)
print(f"Mean time: {sum(times) / len(times)}")
