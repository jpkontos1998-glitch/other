from pyengine.core.env import Stratego
from pyengine.networks.legacy_rl import TransformerRL, TransformerRLConfig
from pyengine.networks.rl_temporal import RLTemporal, RLTemporalConfig
from pyengine import utils
import torch
from torch.amp import autocast
import time
from pyengine.utils.constants import N_OCCUPIABLE_CELL

piece_counts = torch.tensor(utils.init_helpers.COUNTERS["classic"] + [0, 0], device="cuda")
rl_network = TransformerRL(piece_counts, TransformerRLConfig()).cuda()
batch_size = 600
temporal_network = RLTemporal(piece_counts, RLTemporalConfig()).cuda()
env = Stratego(num_envs=batch_size, traj_len_per_player=100)
n_iter = 20
hidden_states = (
    torch.randn(
        temporal_network.cfg.n_layer,
        batch_size,
        N_OCCUPIABLE_CELL + len(temporal_network.special_token_indices),
        temporal_network.cfg.embed_dim,
        device="cuda",
    ),
    torch.randn(
        temporal_network.cfg.n_layer,
        batch_size,
        N_OCCUPIABLE_CELL + len(temporal_network.special_token_indices),
        temporal_network.cfg.embed_dim,
        device="cuda",
    ),
)

for _ in range(5):
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        rl_network(
            env.current_infostate_tensor,
            env.current_piece_ids,
            env.current_legal_action_mask,
            env.current_num_moves,
        )
t = time.time()
torch.cuda.synchronize()
for i in range(n_iter):
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        rl_network(
            env.current_infostate_tensor,
            env.current_piece_ids,
            env.current_legal_action_mask,
            env.current_num_moves,
        )
torch.cuda.synchronize()
rl_network_train_time = (time.time() - t) / n_iter
print(f"RL network train time: {rl_network_train_time}")

for _ in range(5):
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        temporal_network(
            env.current_infostate_tensor,
            env.current_piece_ids,
            env.current_legal_action_mask,
            2 * torch.arange(batch_size, device="cuda"),
        )
t = time.time()
torch.cuda.synchronize()
for i in range(n_iter):
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        temporal_network(
            env.current_infostate_tensor,
            env.current_piece_ids,
            env.current_legal_action_mask,
            2 * torch.arange(batch_size, device="cuda"),
        )
torch.cuda.synchronize()
temporal_network_train_time = (time.time() - t) / n_iter
print(f"Temporal network train time: {temporal_network_train_time}")

for _ in range(5):
    with autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad(), utils.eval_mode(
        rl_network
    ):
        rl_network(
            env.current_infostate_tensor,
            env.current_piece_ids,
            env.current_legal_action_mask,
            env.current_num_moves,
        )
t = time.time()
torch.cuda.synchronize()
for i in range(n_iter):
    with autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad(), utils.eval_mode(
        rl_network
    ):
        rl_network(
            env.current_infostate_tensor,
            env.current_piece_ids,
            env.current_legal_action_mask,
            env.current_num_moves,
        )
torch.cuda.synchronize()
rl_network_eval_time = (time.time() - t) / n_iter
print(f"RL network eval time: {rl_network_eval_time}")

for _ in range(5):
    with autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad(), utils.eval_mode(
        temporal_network
    ):
        temporal_network(
            env.current_infostate_tensor,
            env.current_piece_ids,
            env.current_legal_action_mask,
            torch.zeros(batch_size, device="cuda"),
            hidden_states,
        )
torch.cuda.synchronize()
t = time.time()
for i in range(n_iter):
    with autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad(), utils.eval_mode(
        temporal_network
    ):
        temporal_network(
            env.current_infostate_tensor,
            env.current_piece_ids,
            env.current_legal_action_mask,
            torch.zeros(batch_size, device="cuda"),
            hidden_states,
        )
torch.cuda.synchronize()
temporal_network_eval_time = (time.time() - t) / n_iter
print(f"Temporal network eval time: {temporal_network_eval_time}")
print("-" * 80)
print(f"Train slowdown: {100 * (temporal_network_train_time / rl_network_train_time - 1):.2f}%")
print(f"Eval slowdown: {100 * (temporal_network_eval_time / rl_network_eval_time - 1):.2f}%")
