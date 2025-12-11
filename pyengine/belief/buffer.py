from dataclasses import dataclass
import math

import torch

from pyengine.utils.constants import N_PLAYER
from pyengine.core.env import Stratego
from pyengine.utils import get_pystratego

pystratego = get_pystratego()


@dataclass
class Batch:
    infostate_tensor: torch.Tensor
    piece_ids: torch.Tensor
    num_moves: torch.Tensor
    unknown_piece_position_onehot: torch.Tensor
    unknown_piece_type_onehot: torch.Tensor
    unknown_piece_counts: torch.Tensor
    unknown_piece_has_moved: torch.Tensor


class CircularBuffer:
    def __init__(
        self,
        num_envs: int,
        num_row: int,
        max_num_moves: int,
        max_num_moves_between_attacks: int,
        device: torch.device,
    ):
        self.num_col = num_envs
        self.num_row = num_row
        self.device = device

        self.auxiliary_env = Stratego(
            max_num_moves // N_PLAYER,
            N_PLAYER,
            quiet=2,
            max_num_moves_between_attacks=max_num_moves_between_attacks,
            max_num_moves=max_num_moves,
            nonsteppable=True,
            cuda_device=device,
        )

        self.curr_idx = 0
        self.steps = [-1 for _ in range(self.num_row)]
        self.is_newly_terminal = torch.zeros(
            self.num_row, self.num_col, dtype=torch.bool, device=device
        )

    def add(
        self,
        obs_step: int,
        is_newly_terminal: torch.Tensor,
    ):
        self.steps[self.curr_idx] = obs_step
        self.is_newly_terminal[self.curr_idx] = is_newly_terminal
        self.curr_idx += 1

    def ready_to_train(self):
        return self.curr_idx == self.num_row

    def reset(self):
        assert self.ready_to_train()
        self.curr_idx = 0
        self.is_newly_terminal.zero_()

    def sample(
        self,
        env: Stratego,
    ):
        assert self.ready_to_train()
        boundaries = torch.nonzero(self.is_newly_terminal, as_tuple=False)
        if boundaries.numel() == 0:
            return
        perm = torch.randperm(boundaries.size(0), device=self.device)
        boundaries = boundaries[perm]
        for end_row_t, col_t in boundaries:
            end_row, col = end_row_t.item(), col_t.item()
            end_step = self.steps[(end_row - 1) % self.num_row]
            for state in env.snapshot_env_history(end_step, col):
                if state.num_envs == 0:
                    continue
                state_num_envs = state.num_envs
                state.tile(math.ceil(self.auxiliary_env.num_envs / state.num_envs))
                state = state.slice(0, self.auxiliary_env.num_envs)
                self.auxiliary_env.change_reset_behavior_to_env_state(state)
                infostates = self.auxiliary_env.current_infostate_tensor[:state_num_envs]
                piece_ids = self.auxiliary_env.current_piece_ids[:state_num_envs]
                num_moves = self.auxiliary_env.current_num_moves[:state_num_envs]
                unknown_piece_position_onehot = (
                    self.auxiliary_env.current_unknown_piece_position_onehot[:state_num_envs]
                )
                unknown_piece_type_onehot = self.auxiliary_env.current_unknown_piece_type_onehot[
                    :state_num_envs
                ]
                unknown_piece_counts = self.auxiliary_env.current_unknown_piece_counts[
                    :state_num_envs
                ]
                unknown_piece_has_moved = self.auxiliary_env.current_unknown_piece_has_moved[
                    :state_num_envs
                ]

                yield Batch(
                    infostate_tensor=infostates,
                    piece_ids=piece_ids,
                    num_moves=num_moves,
                    unknown_piece_position_onehot=unknown_piece_position_onehot,
                    unknown_piece_type_onehot=unknown_piece_type_onehot,
                    unknown_piece_counts=unknown_piece_counts,
                    unknown_piece_has_moved=unknown_piece_has_moved,
                )
