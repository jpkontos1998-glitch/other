from pyengine.core.env import Stratego
from pyengine.networks.legacy_rl import TransformerRL, TransformerRLConfig
from pyengine import utils
import torch
import time

piece_counts = torch.tensor(utils.init_helpers.COUNTERS["classic"] + [0, 0], device="cpu")
rl_network = TransformerRL(piece_counts, TransformerRLConfig())
batch_size = 1
env = Stratego(num_envs=batch_size, traj_len_per_player=100)
n_iter = 20
for _ in range(5):
    rl_network(
        env.current_infostate_tensor.cpu(),
        env.current_piece_ids.cpu(),
        env.current_legal_action_mask.cpu(),
        env.current_num_moves.cpu(),
    )
start = time.time()
for i in range(n_iter):
    rl_network(
        env.current_infostate_tensor.cpu(),
        env.current_piece_ids.cpu(),
        env.current_legal_action_mask.cpu(),
        env.current_num_moves.cpu(),
    )
stop = time.time()
total = (stop - start) / n_iter
print(f"Time: {total}")
