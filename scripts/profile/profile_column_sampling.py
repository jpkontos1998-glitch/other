import time

import torch

from pyengine.core.env import Stratego
from pyengine.utils import get_pystratego

pystratego = get_pystratego()

snaptimes = []
stepenvtimes = []
for _ in range(10):
    env = Stratego(num_envs=1, traj_len_per_player=2000)
    action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
    while True:
        env.sample_random_legal_action(action_tensor)
        env.apply_actions(action_tensor)
        if env.current_is_terminal:
            break

    torch.cuda.synchronize()
    start = time.time()
    even, odd = env.snapshot_env_history(env.current_step - 1, 0)
    torch.cuda.synchronize()
    end = time.time()
    snaptimes.append(end - start)
    torch.cuda.synchronize()
    start = time.time()
    for state in [even, odd]:
        seq_env = Stratego(
            state.num_envs,
            2,
            quiet=2,
            reset_state=state,
            reset_behavior=pystratego.ResetBehavior.CUSTOM_ENV_STATE,
            max_num_moves_between_attacks=env.conf.max_num_moves_between_attacks,
            max_num_moves=env.conf.max_num_moves,
            nonsteppable=False,
        )
        del seq_env
        # infostates = seq_env.current_infostate_tensor
        # piece_ids = seq_env.current_piece_ids
        # legal_actions = seq_env.current_legal_action_mask
    torch.cuda.synchronize()
    end = time.time()
    stepenvtimes.append(end - start)
nostepenvtimes = []
for _ in range(10):
    env = Stratego(num_envs=1, traj_len_per_player=2000)
    action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
    while True:
        env.sample_random_legal_action(action_tensor)
        env.apply_actions(action_tensor)
        if env.current_is_terminal:
            break

    torch.cuda.synchronize()
    start = time.time()
    even, odd = env.snapshot_env_history(env.current_step - 1, 0)
    torch.cuda.synchronize()
    end = time.time()
    snaptimes.append(end - start)
    torch.cuda.synchronize()
    start = time.time()
    for state in [even, odd]:
        seq_env = Stratego(
            state.num_envs,
            2,
            quiet=2,
            reset_state=state,
            reset_behavior=pystratego.ResetBehavior.CUSTOM_ENV_STATE,
            max_num_moves_between_attacks=env.conf.max_num_moves_between_attacks,
            max_num_moves=env.conf.max_num_moves,
            nonsteppable=True,
        )
        del seq_env
        # infostates = seq_env.current_infostate_tensor
        # piece_ids = seq_env.current_piece_ids
        # legal_actions = seq_env.current_legal_action_mask
    torch.cuda.synchronize()
    end = time.time()
    nostepenvtimes.append(end - start)

print(f"Snapshot time: {sum(snaptimes) / len(snaptimes)} seconds")
print(f"Steppable time: {sum(stepenvtimes) / len(stepenvtimes)} seconds")
print(f"Nonsteppable time: {sum(nostepenvtimes) / len(nostepenvtimes)} seconds")
